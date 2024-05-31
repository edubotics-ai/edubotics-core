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
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from ragatouille import RAGPretrainedModel
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate

try:
    from modules.helpers import get_lecture_metadata
except:
    from helpers import get_lecture_metadata

logger = logging.getLogger(__name__)


class PDFReader:
    def __init__(self):
        pass

    def get_loader(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        return loader

    def get_documents(self, loader):
        return loader.load()


class FileReader:
    def __init__(self):
        self.pdf_reader = PDFReader()

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    def download_pdf_from_url(self, pdf_url):
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

    # def extract_metadata(self, document_content):

    #     llm = OpenAI()
    #     prompt_template = PromptTemplate(
    #         input_variables=["document_content"],
    #         template="Extract metadata for this document:\n\n{document_content}\n\nMetadata:",
    #     )
    #     chain = LLMChain(llm=llm, prompt=prompt_template)
    #     metadata = chain.run(document_content=document_content)
    #     return metadata

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
                    logger.info(f"\tSplitting link {link_index+1} : {link}")
                    if "youtube" in link:
                        documents = file_reader.read_youtube_transcript(link)
                    else:
                        documents = file_reader.read_html(link)

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
                        f"Error splitting link {link_index+1} : {link}: {str(e)}"
                    )


class DataLoader:
    def __init__(self, config):
        self.file_reader = FileReader()
        self.chunk_processor = ChunkProcessor(config)

    def get_chunks(self, uploaded_files, weblinks):
        return self.chunk_processor.get_chunks(
            self.file_reader, uploaded_files, weblinks
        )
