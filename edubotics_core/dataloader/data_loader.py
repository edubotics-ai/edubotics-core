import os
from pprint import pprint
import re
import traceback
import requests
import pysrt
from langchain_community.document_loaders import (
    Docx2txtLoader,
    YoutubeLoader,
    TextLoader,
)
from langchain.schema import Document
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
import html2text
import bs4
from edubotics_core.dataloader.pdf_readers.base import PDFReader
from edubotics_core.dataloader.pdf_readers.llama import LlamaParser
from edubotics_core.dataloader.pdf_readers.gpt import GPTParser
from edubotics_core.dataloader.repo_readers.github import GithubReader
from edubotics_core.dataloader.repo_readers.helpers import read_notebook_from_file
from edubotics_core.dataloader.metadata_extractor import LLMMetadataExtractor
from edubotics_core.dataloader.helpers import get_metadata
from edubotics_core.config.constants import TIMEOUT

logger = logging.getLogger(__name__)
BASE_DIR = os.getcwd()


class HTMLReader:
    def __init__(self):
        pass

    def read_url(self, url):
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            return response.text
        else:
            logger.warning(f"Failed to download HTML from URL: {url}")
            return None

    def check_links(self, base_url, html_content):
        soup = bs4.BeautifulSoup(html_content, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")

            if not href or href.startswith("#"):
                continue
            elif not href.startswith("https"):
                href = href.replace("http", "https")

            absolute_url = urljoin(base_url, href)
            link["href"] = absolute_url

            try:
                resp = requests.head(absolute_url, timeout=TIMEOUT)
                if resp.status_code != 200:
                    pass
            except Exception as e:
                pass

        return str(soup)

    def html_to_md(self, url, html_content):
        html_processed = self.check_links(url, html_content)
        markdown_content = html2text.html2text(html_processed)
        return markdown_content

    def read_html(self, url):
        html_content = self.read_url(url)
        if html_content:
            return self.html_to_md(url, html_content)
        else:
            return None


class FileReader:
    def __init__(self, logger, config, kind):
        self.logger = logger
        self.config = config
        self.kind = kind

        if kind == "llama":
            self.pdf_reader = LlamaParser()
        elif kind == "gpt":
            self.pdf_reader = GPTParser()
        else:
            self.pdf_reader = PDFReader()

        self.web_reader = HTMLReader()
        self.github_reader = GithubReader()
        self.logger.info(
            f"Initialized FileReader with {kind} PDF reader and HTML reader"
        )

    def read_pdf(self, temp_file_path: str):
        documents = self.pdf_reader.parse(temp_file_path)
        return documents

    def read_txt(self, temp_file_path: str):
        if temp_file_path.startswith("http"):
            return self.read_txt_from_url(temp_file_path)
        else:
            loader = TextLoader(temp_file_path, autodetect_encoding=True)
            return loader.load()

    def read_txt_from_url(self, url: str):
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            return [Document(page_content=response.text)]
        else:
            return None

    def read_docx(self, temp_file_path: str):
        loader = Docx2txtLoader(temp_file_path)
        return loader.load()

    def read_srt(self, temp_file_path: str):
        subs = pysrt.open(temp_file_path)
        text = ""
        for sub in subs:
            text += sub.text
        return [Document(page_content=text)]

    def read_youtube_transcript(self, url: str):
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=True, language=["en"], translation="en"
        )
        return loader.load()

    def read_html(self, url: str):
        return [Document(page_content=self.web_reader.read_html(url), metadata={"source": url})]

    def read_tex_from_url(self, tex_url):
        response = requests.get(tex_url, timeout=TIMEOUT)
        if response.status_code == 200:
            return [Document(page_content=response.text)]
        else:
            self.logger.error(f"Failed to fetch .tex file from URL: {tex_url}")
            return None

    def read_github_repo(self, github_url: str):
        repo_contents = self.github_reader.get_repo_contents(github_url)
        docs = [
            Document(page_content=content, metadata={"source": file})
            for file, content in repo_contents.items()
            if content is not None
        ]
        for i, doc in enumerate(docs):
            doc.metadata["page"] = i

        return docs

    def read_notebook(self, notebook_path):
        if "github.com" in notebook_path and "blob" in notebook_path:
            notebook_path = notebook_path.replace(
                "github.com", "raw.githubusercontent.com"
            )
            notebook_path = notebook_path.replace("/blob/", "/")
            self.logger.info(f"Changed notebook path to {notebook_path}")

        notebook_content = read_notebook_from_file(
            notebook_path,
            headers_to_split_on=self.config["content"]["notebookheaders_to_split_on"],
        )

        return [Document(page_content=notebook_content)]


class ChunkProcessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.document_data = []
        self.document_metadata = []
        self.document_chunks_full = []

        # TODO: Fix when reparse_files is False
        if not config["vectorstore"]["reparse_files"]:
            self.load_document_data()

        if config["splitter_options"]["use_splitter"]:
            if config["splitter_options"]["chunking_mode"] == "fixed":
                if config["splitter_options"]["split_by_token"]:
                    self.splitter = (
                        RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                            chunk_size=config["splitter_options"]["chunk_size"],
                            chunk_overlap=config["splitter_options"]["chunk_overlap"],
                            separators=config["splitter_options"]["chunk_separators"],
                            disallowed_special=(),
                        )
                    )
                else:
                    self.splitter = RecursiveCharacterTextSplitter(
                        chunk_size=config["splitter_options"]["chunk_size"],
                        chunk_overlap=config["splitter_options"]["chunk_overlap"],
                        separators=config["splitter_options"]["chunk_separators"],
                        disallowed_special=(),
                    )
            else:
                self.splitter = SemanticChunker(
                    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
                )

        else:
            self.splitter = None
        self.logger.info("ChunkProcessor instance created")

    def remove_delimiters(self, document_chunks: list):
        for chunk in document_chunks:
            for delimiter in self.config["splitter_options"]["delimiters_to_remove"]:
                chunk.page_content = re.sub(delimiter, " ", chunk.page_content)
        return document_chunks

    def remove_chunks(self, document_chunks: list):
        front = self.config["splitter_options"]["front_chunk_to_remove"]
        end = self.config["splitter_options"]["last_chunks_to_remove"]
        for _ in range(front):
            del document_chunks[0]
        for _ in range(end):
            document_chunks.pop()
        return document_chunks

    def process_chunks(
        self, documents, file_type="txt", source="", page=0, metadata={}
    ):
        # TODO: Clear up this pipeline of re-adding metadata
        # documents = [Document(page_content=documents, source=source, page=page)]
        if (
            file_type == "pdf"
            and self.config["splitter_options"]["chunking_mode"] == "fixed"
        ):
            document_chunks = documents
        else:
            document_chunks = self.splitter.split_documents(documents)

        # add the source and page number back to the metadata
        for i, chunk in enumerate(document_chunks):
            # print(
            #     f"Chunk {i}: {chunk.page_content[:100] + '...' +chunk.page_content[-100:]}"
            # )
            # add the metadata extracted from the document
            for key, value in metadata.items():
                chunk.metadata[key] = value

            chunk_name = (
                os.path.basename(source) if os.path.basename(source) != "" else source
            )
            if "README" in chunk_name:
                chunk_name = "/".join(source.split("/")[-3:])

            chunk.metadata["source"] = source
            chunk.metadata["page"] = i
            chunk.metadata["chunk_id"] = f"{chunk_name}_{i}"

        if self.config["splitter_options"]["remove_leftover_delimiters"]:
            document_chunks = self.remove_delimiters(document_chunks)
        if self.config["splitter_options"]["remove_chunks"]:
            document_chunks = self.remove_chunks(document_chunks)

        return document_chunks

    def chunk_docs(self, file_reader, uploaded_files, weblinks):
        addl_metadata = get_metadata(
            *self.config["metadata"]["metadata_links"], self.config
        )  # For any additional metadata'''

        # remove already processed files if reparse_files is False
        if not self.config["vectorstore"]["reparse_files"]:
            total_documents = len(uploaded_files) + len(weblinks)
            uploaded_files = [
                file_path
                for file_path in uploaded_files
                if file_path not in self.document_data
            ]
            weblinks = [link for link in weblinks if link not in self.document_data]
            print(
                f"Total documents to process: {total_documents}, Documents already processed: {total_documents - len(uploaded_files) - len(weblinks)}"
            )

        with ThreadPoolExecutor() as executor:
            executor.map(
                self.process_file,
                uploaded_files,
                range(len(uploaded_files)),
                [file_reader] * len(uploaded_files),
                [addl_metadata] * len(uploaded_files),
            )
            executor.map(
                self.process_weblink,
                weblinks,
                range(len(weblinks)),
                [file_reader] * len(weblinks),
                [addl_metadata] * len(weblinks),
            )
        documents = [
            "".join([chunk.page_content for chunk in doc["chunks"]])
            for doc in self.document_data
        ]
        chunks = [
            chunk
            for doc in self.document_data
            for chunk in doc["chunks"]
        ]
        document_names = [doc["document_name"] for doc in self.document_data]
        document_metadata = [doc["metadata"] for doc in self.document_data]

        self.document_data_dict = [
            {
                "document_name": doc["document_name"],
                "metadata": doc["metadata"],
                "chunks": [
                    {
                        "content": chunk.page_content,
                        "source": chunk.metadata["source"],
                        "page": chunk.metadata["page"],
                        "chunk_id": chunk.metadata["chunk_id"],
                    }
                    for chunk in doc["chunks"]
                ],
            }
            for doc in self.document_data
        ]

        self.save_document_data()

        total_chunks = sum([len(doc["chunks"]) for doc in self.document_data])
        self.logger.info(
            f"Total document chunks extracted: {(total_chunks)}"
        )

        return chunks, document_names, documents, document_metadata

    def process_documents(
        self, documents, file_path, file_type, metadata_source, addl_metadata
    ):

        doc_name = os.path.basename(file_path)
        if "README" in doc_name:
            doc_name = file_path

        if doc_name == "":
            doc_name = file_path

        doc_metadata = {
            "source": file_path,
        }

        # Processing metadata for documents
        if self.config["metadata"]["lectures_pattern"] in file_path:
            addl_metadata_copy = addl_metadata.copy()
            doc_metadata.update(addl_metadata_copy)
            doc_metadata["content_type"] = "lecture"
        elif self.config["metadata"]["assignments_pattern"] in file_path:
            addl_metadata = LLMMetadataExtractor(
                fields=self.config["metadata"]["assignment_metadata_fields"]
            ).extract_metadata(file_path)

            doc_metadata.update(addl_metadata)
            doc_metadata["content_type"] = "assignment"
        else:
            doc_metadata["content_type"] = "other"

        # Chunking
        if self.config["vectorstore"]["db_option"] not in ["RAGatouille"]:
            document_chunks = self.process_chunks(
                documents,
                file_type,
                source=file_path,
                metadata=doc_metadata,
            )
            self.document_chunks_full.extend(document_chunks)

        file_data = {
            "document_name": doc_name,
            "metadata": doc_metadata,
            "chunks": document_chunks,
        }

        self.document_data.append(file_data)
        # self.document_metadata[file_path] = doc_metadata

    def process_file(self, file_path, file_index, file_reader, addl_metadata):
        self.logger.info(f"Processing file {file_index + 1} : {file_path}")
        file_name = os.path.basename(file_path)

        file_type = file_name.split(".")[-1]

        read_methods = {
            "pdf": file_reader.read_pdf,
            "txt": file_reader.read_txt,
            "docx": file_reader.read_docx,
            "srt": file_reader.read_srt,
            "tex": file_reader.read_tex_from_url,
            "ipynb": file_reader.read_notebook,
            "md": file_reader.read_txt,
        }
        if file_type not in read_methods:
            self.logger.warning(f"Unsupported file type: {file_type}")
            return

        try:
            if file_path in self.document_data:
                self.logger.warning(f"File {file_name} already processed")
                documents = [
                    Document(page_content=content)
                    for content in self.document_data[file_path].values()
                ]
            else:
                documents = read_methods[file_type](file_path)

            self.process_documents(
                documents, file_path, file_type, "file", addl_metadata
            )
        except Exception as e:
            self.logger.error(f"Error processing file {file_name}: {str(e)}")

    def process_weblink(self, link, link_index, file_reader, addl_metadata):
        self.logger.info(f"Reading link {link_index + 1} : {link}")

        if link in self.document_data:
            return

        try:
            if "youtube" in link:
                documents = file_reader.read_youtube_transcript(link)
            elif "github.com" in link:
                documents = file_reader.read_github_repo(link)
            else:
                documents = file_reader.read_html(link)

            if len(set([doc.metadata["source"] for doc in documents])) > 1:
                self.logger.warning(
                    f"Documents from link {link_index + 1} : {link} have multiple sources")
                for doc in documents:
                    self.process_documents(
                        [doc], doc.metadata["source"], "txt", "link", addl_metadata)
            else:
                self.process_documents(documents, link, "txt", "link", addl_metadata)
        except Exception as e:
            self.logger.error(f"Error Reading link {link_index + 1} : {link}: {str(e)}")
            self.logger.error(f"Error traceback: {traceback.format_exc()}")

    def save_document_data(self):
        if not os.path.exists(f"{self.config['log_chunk_dir']}"):
            os.makedirs(f"{self.config['log_chunk_dir']}")
            self.logger.info(f"Creating directory {self.config['log_chunk_dir']}")
        self.logger.info(
            f"Saving document content to {self.config['log_chunk_dir']}/doc_content.json"
        )
        with open(f"{self.config['log_chunk_dir']}/doc_content.json", "w") as json_file:
            json.dump(self.document_data_dict, json_file, indent=4)

    def load_document_data(self):
        try:
            with open(
                f"{self.config['log_chunk_dir']}/doc_content.json", "r"
            ) as json_file:
                self.document_data_dict = json.load(json_file)
            self.logger.info(
                f"Loaded document content from {self.config['log_chunk_dir']}/doc_content.json. Total documents: {len(self.document_data)}"
            )
        except FileNotFoundError:
            self.logger.warning(
                f"Document content not found in {self.config['log_chunk_dir']}/doc_content.json"
            )
            self.document_data_dict = {}


class DataLoader:
    def __init__(self, config, logger=None):
        self.file_reader = FileReader(
            logger=logger, config=config, kind=config["llm_params"]["pdf_reader"]
        )
        self.chunk_processor = ChunkProcessor(config, logger=logger)

    def get_chunks(self, uploaded_files, weblinks):
        return self.chunk_processor.chunk_docs(
            self.file_reader, uploaded_files, weblinks
        )


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Data Loader")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the main config file",
        # required=True,
        default="config/config.yml",
    )
    parser.add_argument(
        "--project_config_file",
        type=str,
        help="Path to the project config file",
        # required=True,
        default="config/project_config.yml",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    with open(args.project_config_file, "r") as f:
        project_config = yaml.safe_load(f)

    # Combine project config with the main config
    config.update(project_config)

    STORAGE_DIR = os.path.join(BASE_DIR, config["vectorstore"]["data_path"])
    uploaded_files = [
        os.path.join(STORAGE_DIR, file)
        for file in os.listdir(STORAGE_DIR)
        if file != "urls.txt"
    ]

    urls_file = os.path.join(STORAGE_DIR, "urls.txt")
    with open(urls_file, "r") as f:
        weblinks = f.readlines()

    weblinks = [link.strip() for link in weblinks]

    print(f"Uploaded files: {uploaded_files}")
    print(f"Web links: {weblinks}")

    data_loader = DataLoader(config, logger=logger)
    # Just for testing
    (
        document_chunks,
        document_names,
        documents,
        document_metadata,
    ) = data_loader.get_chunks(
        uploaded_files,
        weblinks,
    )

    print(document_names[:5])
    print(len(document_chunks))
    print(document_chunks[2].page_content[:100])
