import os
import bs4
from urllib.parse import urljoin
import asyncio
import requests
import pysrt
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    YoutubeLoader,
    WebBaseLoader,
    TextLoader,
)
import html2text
import tempfile
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from llama_parse import LlamaParse
from langchain.schema import Document
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from ragatouille import RAGPretrainedModel
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate

try:
    from modules.helpers import get_lecture_metadata
    from modules.constants import OPENAI_API_KEY, LLAMA_CLOUD_API_KEY
except:
    from helpers import get_lecture_metadata
    from constants import OPENAI_API_KEY, LLAMA_CLOUD_API_KEY

logger = logging.getLogger(__name__)
BASE_DIR = os.getcwd()
STORAGE_DIR = os.path.join(BASE_DIR, "storage", "data")

class PDFReader:
    def __init__(self):
        pass

    def get_loader(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        return loader

    def get_documents(self, loader):
        return loader.load()


class LlamaParser:
    def __init__(self):
        self.GPT_API_KEY = OPENAI_API_KEY
        self.LLAMA_CLOUD_API_KEY = LLAMA_CLOUD_API_KEY
        self.parse_url = "https://api.cloud.llamaindex.ai/api/parsing/upload"
        self.headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer llx-vap5Bk2zbYLfqTq2aZDvNHwscvsBPQiSjvLOGkgUa9SS8CWB'
        }
        self.parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=True,
            language="en",
            gpt4o_mode=False,
            # gpt4o_api_key=OPENAI_API_KEY,
            parsing_instruction="The provided documents are PDFs of lecture slides of deep learning material. They contain LaTeX equations, images, and text. The goal is to extract the text, images and equations from the slides and convert them to markdown format. The markdown should be clean and easy to read, and any math equation should be converted to LaTeX, between $$. For images, give a description and if you can, a source."
        )

    def parse(self, pdf_path):
        pdf_name = os.path.basename(pdf_path)
        logger.info(f"Processing PDF: {pdf_name}. Path: {pdf_path}")

        path = os.path.join(STORAGE_DIR, pdf_name)
        if os.path.exists(path):
            pdf_path = os.path.join(STORAGE_DIR, path)
        else:
            pdf_path = FileReader.download_pdf_from_url(pdf_url=pdf_path)

        documents = self.parser.load_data(pdf_path)
        documents = [document.to_langchain_format() for document in documents]
        print(documents)

        os.remove(pdf_path)
        return documents

    def make_request(self, pdf_url):
        payload = {
            "gpt4o_mode": "false",
            "parsing_instruction": "The provided document is a PDF of lecture slides of deep learning material. They contain LaTeX equations, images, and text. The goal is to extract the text, images and equations from the slides and convert them to markdown format. The markdown should be clean and easy to read, and any math equation should be converted to LaTeX, between $$. For images, give a description and if you can, a source.",
        }

        files = [
            ('file', ('file', requests.get(pdf_url).content, 'application/octet-stream'))
        ]

        response = requests.request(
            "POST", self.parse_url, headers=self.headers, data=payload, files=files)

        return response.json()['id'], response.json()['status']

    async def get_result(self, job_id):
        url = f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/markdown"

        response = requests.request("GET", url, headers=self.headers, data={})

        return response.json()['markdown']

    async def _parse(self, pdf_path):
        job_id, status = self.make_request(pdf_path)
        print(f"Job ID: {job_id}", f"Status: {status}")

        while status != "SUCCESS":
            url = f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}"
            response = requests.request("GET", url, headers=self.headers, data={})
            status = response.json()["status"]

        print(status)

        result = await self.get_result(job_id)

        documents = [
            Document(
                page_content=result,
                metadata={"source": pdf_path}
            )
        ]

        return documents

    async def _parse(self, pdf_path):
        return await self._parse(pdf_path)

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
    def __init__(self, kind):
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

    @staticmethod
    def download_pdf_from_url(pdf_url):
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            return temp_file_path
        else:
            print("Failed to download PDF from URL:", pdf_url)
            return None

    def read_pdf(self, temp_file_path: str):
        if self.kind == "llama":
            #documents = asyncio.run(self.pdf_reader.parse(temp_file_path))
            documents = self.pdf_reader.parse(temp_file_path)
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
        return [Document(page_content=self.web_reader.read_html(url))]


class ChunkProcessor:
    def __init__(self, config):
        self.config = config

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
        logger.info("ChunkProcessor instance created")

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
        logger.info(f"\tNumber of pages after skipping: {len(document_chunks)}")
        return document_chunks

    def process_chunks(
            self, documents, file_type="txt", source="", page=0, metadata={}
    ):
        documents = [Document(page_content=documents, source=source, page=page)]
        if file_type == "txt":
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

    def get_chunks(self, file_reader, uploaded_files, weblinks):
        self.document_chunks_full = []
        self.parent_document_names = []
        self.child_document_names = []
        self.documents = []
        self.document_metadata = []

        lecture_metadata = get_lecture_metadata(
            "https://dl4ds.github.io/sp2024/lectures/",
            "https://dl4ds.github.io/sp2024/schedule/",
        )  # TODO: Use more efficiently

        for file_index, file_path in enumerate(uploaded_files):
            file_name = os.path.basename(file_path)
            file_type = file_name.split(".")[-1].lower()

            # try:
            if file_type == "pdf":
                documents = file_reader.read_pdf(file_path)
            elif file_type == "txt":
                documents = file_reader.read_txt(file_path)
            elif file_type == "docx":
                documents = file_reader.read_docx(file_path)
            elif file_type == "srt":
                documents = file_reader.read_srt(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                continue

            # full_text = ""
            # for doc in documents:
            #     full_text += doc.page_content
            #     break  # getting only first page for now

            # extracted_metadata = self.extract_metadata(full_text)

            for doc in documents:
                page_num = doc.metadata.get("page", 0)
                self.documents.append(doc.page_content)
                self.document_metadata.append({"source": file_path, "page": page_num})
                if "lecture" in file_path.lower():
                    metadata = lecture_metadata.get(file_path, {})
                    metadata["source_type"] = "lecture"
                    self.document_metadata[-1].update(metadata)
                else:
                    metadata = {"source_type": "other"}

                self.child_document_names.append(f"{file_name}_{page_num}")

                self.parent_document_names.append(file_name)
                if self.config["embedding_options"]["db_option"] not in ["RAGatouille"]:
                    document_chunks = self.process_chunks(
                        self.documents[-1],
                        file_type,
                        source=file_path,
                        page=page_num,
                        metadata=metadata,
                    )
                    self.document_chunks_full.extend(document_chunks)

            # except Exception as e:
            #     logger.error(f"Error processing file {file_name}: {str(e)}")

        self.process_weblinks(file_reader, weblinks)

        logger.info(
            f"Total document chunks extracted: {len(self.document_chunks_full)}"
        )
        return (
            self.document_chunks_full,
            self.child_document_names,
            self.documents,
            self.document_metadata,
        )

    def process_weblinks(self, file_reader, weblinks):
        if weblinks[0] != "":
            logger.info(f"Splitting weblinks: total of {len(weblinks)}")

            for link_index, link in enumerate(weblinks):
                try:
                    logger.info(f"\tSplitting link {link_index + 1} : {link}")
                    if "youtube" in link:
                        documents = file_reader.read_youtube_transcript(link)
                    else:
                        documents = file_reader.read_html(link)
                        print(f"Link: {link}")
                        print(documents)
                    for doc in documents:
                        page_num = doc.metadata.get("page", 0)
                        self.documents.append(doc.page_content)
                        self.document_metadata.append(
                            {"source": link, "page": page_num}
                        )
                        self.child_document_names.append(f"{link}")

                    self.parent_document_names.append(link)
                    if self.config["embedding_options"]["db_option"] not in [
                        "RAGatouille"
                    ]:
                        document_chunks = self.process_chunks(
                            self.documents[-1],
                            "txt",
                            source=link,
                            page=0,
                            metadata={"source_type": "webpage"},
                        )
                        self.document_chunks_full.extend(document_chunks)
                except Exception as e:
                    logger.error(
                        f"Error splitting link {link_index + 1} : {link}: {str(e)}"
                    )


class DataLoader:
    def __init__(self, config):
        if config["llm_params"]["pdf_reader"] == "llama":
            if LLAMA_CLOUD_API_KEY == None or OPENAI_API_KEY == None:
                raise ValueError(
                    "Please set the LLAMA_CLOUD_API_KEY and GPT4o_API_KEY environment variables"
                )

        self.file_reader = FileReader(kind=config["llm_params"]["pdf_reader"])
        self.chunk_processor = ChunkProcessor(config)

    def get_chunks(self, uploaded_files, weblinks):
        return self.chunk_processor.get_chunks(
            self.file_reader, uploaded_files, weblinks
        )


if __name__ == "__main__":
    # read config.yml file
    import yaml
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, "../", "config.yml"), "r") as f:
        config = yaml.safe_load(f)

    # create DataLoader instance
    chunk_processor = ChunkProcessor(config)
    file_reader = FileReader(kind=config["llm_params"]["pdf_reader"])

    weblinks = ["https://dl4ds.github.io/sp2024/"]

    uploaded_files = []

    # get document chunks
    document_chunks, child_document_names, documents, document_metadata = chunk_processor.get_chunks(
        file_reader, uploaded_files, weblinks
    )


    print(document_chunks)