import os
import re
import requests
import pysrt
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    YoutubeLoader,
    WebBaseLoader,
    TextLoader,
)
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from llama_parse import LlamaParse
from langchain.schema import Document
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragatouille import RAGPretrainedModel
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain import PromptTemplate
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
import html2text
import bs4
import tempfile
import PyPDF2
from modules.dataloader.pdf_readers.base import PDFReader
from modules.dataloader.pdf_readers.llama import LlamaParser

try:
    from modules.dataloader.helpers import get_metadata, download_pdf_from_url
    from modules.config.constants import OPENAI_API_KEY, LLAMA_CLOUD_API_KEY
except:
    from dataloader.helpers import get_metadata, download_pdf_from_url
    from config.constants import OPENAI_API_KEY, LLAMA_CLOUD_API_KEY

logger = logging.getLogger(__name__)
BASE_DIR = os.getcwd()


class HTMLReader:
    def __init__(self):
        pass

    def read_url(self, url):
        response = requests.get(url)
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
            link['href'] = absolute_url

            resp = requests.head(absolute_url)
            if resp.status_code != 200:
                logger.warning(f"Link {absolute_url} is broken")
                logger.warning(f"Status code: {resp.status_code}")

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
    def __init__(self, logger, kind):
        self.logger = logger
        self.kind = kind
        if kind == "llama":
            self.pdf_reader = LlamaParser()
        else:
            self.pdf_reader = PDFReader()
        self.web_reader = HTMLReader()


    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    def read_pdf(self, temp_file_path: str):
        if self.kind == "llama":
            documents = self.pdf_reader.parse(temp_file_path) # asyncio.run(self.pdf_reader.parse(temp_file_path)) if using async
        else:
            loader = self.pdf_reader.get_loader(temp_file_path)
            documents = self.pdf_reader.get_documents(loader)
        return documents

    def read_txt(self, temp_file_path: str):
        loader = TextLoader(temp_file_path, autodetect_encoding=True)
        return loader.load()

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
        loader = WebBaseLoader(url)
        return loader.load()

    def read_tex_from_url(self, tex_url):
        response = requests.get(tex_url)
        if response.status_code == 200:
            return [Document(page_content=response.text)]
        else:
            self.logger.error(f"Failed to fetch .tex file from URL: {tex_url}")
            return None


class ChunkProcessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.document_data = {}
        self.document_metadata = {}
        self.document_chunks_full = []

        if config["splitter_options"]["use_splitter"]:
            if config["splitter_options"]["split_by_token"]:
                self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=config["splitter_options"]["chunk_size"],
                    chunk_overlap=config["splitter_options"]["chunk_overlap"],
                    separators=config["splitter_options"]["chunk_separators"],
                    disallowed_special=(),
                )
            else:
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config["splitter_options"]["chunk_size"],
                    chunk_overlap=config["splitter_options"]["chunk_overlap"],
                    separators=config["splitter_options"]["chunk_separators"],
                    disallowed_special=(),
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
        documents = [Document(page_content=documents, source=source, page=page)]
        if (
            file_type == "txt"
            or file_type == "docx"
            or file_type == "srt"
            or file_type == "tex"
        ):
            document_chunks = self.splitter.split_documents(documents)
        elif file_type == "pdf":
            document_chunks = documents  # Full page for now

        # add the source and page number back to the metadata
        for chunk in document_chunks:
            chunk.metadata["source"] = source
            chunk.metadata["page"] = page

            # add the metadata extracted from the document
            for key, value in metadata.items():
                chunk.metadata[key] = value

        if self.config["splitter_options"]["remove_leftover_delimiters"]:
            document_chunks = self.remove_delimiters(document_chunks)
        if self.config["splitter_options"]["remove_chunks"]:
            document_chunks = self.remove_chunks(document_chunks)

        return document_chunks

    def chunk_docs(self, file_reader, uploaded_files, weblinks):
        addl_metadata = get_metadata(
            "https://dl4ds.github.io/sp2024/lectures/",
            "https://dl4ds.github.io/sp2024/schedule/",
        )  # For any additional metadata
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

        document_names = [
            f"{file_name}_{page_num}"
            for file_name, pages in self.document_data.items()
            for page_num in pages.keys()
        ]
        documents = [
            page for doc in self.document_data.values() for page in doc.values()
        ]
        document_metadata = [
            page for doc in self.document_metadata.values() for page in doc.values()
        ]

        self.save_document_data()

        self.logger.info(
            f"Total document chunks extracted: {len(self.document_chunks_full)}"
        )

        return self.document_chunks_full, document_names, documents, document_metadata

    def process_documents(
        self, documents, file_path, file_type, metadata_source, addl_metadata
    ):
        file_data = {}
        file_metadata = {}

        for doc in documents:
            # if len(doc.page_content) <= 400: # better approach to filter out non-informative documents
            #     continue

            page_num = doc.metadata.get("page", 0)
            file_data[page_num] = doc.page_content
            metadata = (
                addl_metadata.get(file_path, {})
                if metadata_source == "file"
                else {"source": file_path, "page": page_num}
            )
            file_metadata[page_num] = metadata

            if self.config["vectorstore"]["db_option"] not in ["RAGatouille"]:
                document_chunks = self.process_chunks(
                    doc.page_content,
                    file_type,
                    source=file_path,
                    page=page_num,
                    metadata=metadata,
                )
                self.document_chunks_full.extend(document_chunks)

        print(f"Processed {file_path}. File_data: {file_data}")
        self.document_data[file_path] = file_data
        self.document_metadata[file_path] = file_metadata

    def process_file(self, file_path, file_index, file_reader, addl_metadata):
        file_name = os.path.basename(file_path)

        if file_name in self.document_data:
            return

        file_type = file_name.split(".")[-1]

        read_methods = {
            "pdf": file_reader.read_pdf,
            "txt": file_reader.read_txt,
            "docx": file_reader.read_docx,
            "srt": file_reader.read_srt,
            "tex": file_reader.read_tex_from_url,
        }
        if file_type not in read_methods:
            self.logger.warning(f"Unsupported file type: {file_type}")
            return

        try:
            documents = read_methods[file_type](file_path)

            self.process_documents(
                documents, file_path, file_type, "file", addl_metadata
            )
        except Exception as e:
            self.logger.error(f"Error processing file {file_name}: {str(e)}")

    def process_weblink(self, link, link_index, file_reader, addl_metadata):
        if link in self.document_data:
            return

        self.logger.info(f"Reading link {link_index + 1} : {link}")

        try:
            if "youtube" in link:
                documents = file_reader.read_youtube_transcript(link)
            else:
                documents = file_reader.read_html(link)

            self.process_documents(documents, link, "txt", "link", addl_metadata)
        except Exception as e:
            self.logger.error(f"Error Reading link {link_index + 1} : {link}: {str(e)}")

    def save_document_data(self):
        if not os.path.exists(f"{self.config['log_chunk_dir']}/docs"):
            os.makedirs(f"{self.config['log_chunk_dir']}/docs")
            self.logger.info(
                f"Creating directory {self.config['log_chunk_dir']}/docs for document data"
            )
        self.logger.info(
            f"Saving document content to {self.config['log_chunk_dir']}/docs/doc_content.json"
        )
        if not os.path.exists(f"{self.config['log_chunk_dir']}/metadata"):
            os.makedirs(f"{self.config['log_chunk_dir']}/metadata")
            self.logger.info(
                f"Creating directory {self.config['log_chunk_dir']}/metadata for document metadata"
            )
        self.logger.info(
            f"Saving document metadata to {self.config['log_chunk_dir']}/metadata/doc_metadata.json"
        )
        with open(
            f"{self.config['log_chunk_dir']}/docs/doc_content.json", "w"
        ) as json_file:
            json.dump(self.document_data, json_file, indent=4)
        with open(
            f"{self.config['log_chunk_dir']}/metadata/doc_metadata.json", "w"
        ) as json_file:
            json.dump(self.document_metadata, json_file, indent=4)

    def load_document_data(self):
        with open(
            f"{self.config['log_chunk_dir']}/docs/doc_content.json", "r"
        ) as json_file:
            self.document_data = json.load(json_file)
        with open(
            f"{self.config['log_chunk_dir']}/metadata/doc_metadata.json", "r"
        ) as json_file:
            self.document_metadata = json.load(json_file)


class DataLoader:
    def __init__(self, config, logger=None):
        self.file_reader = FileReader(logger=logger, kind=config["llm_params"]["pdf_reader"])
        self.chunk_processor = ChunkProcessor(config, logger=logger)

    def get_chunks(self, uploaded_files, weblinks):
        return self.chunk_processor.chunk_docs(
            self.file_reader, uploaded_files, weblinks
        )


if __name__ == "__main__":
    import yaml

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    with open("../code/modules/config/config.yml", "r") as f:
        config = yaml.safe_load(f)

    STORAGE_DIR = os.path.join(BASE_DIR, config['vectorstore']["data_path"])
    uploaded_files = [
        os.path.join(STORAGE_DIR, file) for file in os.listdir(STORAGE_DIR) if file != "urls.txt"
    ]

    data_loader = DataLoader(config, logger=logger)
    document_chunks, document_names, documents, document_metadata = (
        data_loader.get_chunks(
            ["https://dl4ds.github.io/sp2024/static_files/lectures/05_loss_functions_v2.pdf"],
            [],
        )
    )

    print(document_names[:5])
    print(len(document_chunks))

