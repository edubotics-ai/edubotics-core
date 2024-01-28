import logging
import os
import yaml

from modules.embedding_model_loader import EmbeddingModelLoader
from langchain.vectorstores import FAISS
from modules.data_loader import DataLoader
from modules.constants import *
from modules.helpers import *


class VectorDB:
    def __init__(self, config, logger=None):
        self.config = config
        self.db_option = config["embedding_options"]["db_option"]
        self.document_names = None
        self.webpage_crawler = WebpageCrawler()

        # Set up logging to both console and a file
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File Handler
            log_file_path = "vector_db.log"  # Change this to your desired log file path
            file_handler = logging.FileHandler(log_file_path, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            self.logger = logger

        self.logger.info("VectorDB instance instantiated")

    def load_files(self):
        files = os.listdir(self.config["embedding_options"]["data_path"])
        files = [
            os.path.join(self.config["embedding_options"]["data_path"], file)
            for file in files
        ]
        urls = get_urls_from_file(self.config["embedding_options"]["url_file_path"])
        if self.config["embedding_options"]["expand_urls"]:
            all_urls = []
            for url in urls:
                base_url = get_base_url(url)
                all_urls.extend(self.webpage_crawler.get_all_pages(url, base_url))
            urls = all_urls
        return files, urls

    def create_embedding_model(self):
        self.logger.info("Creating embedding function")
        self.embedding_model_loader = EmbeddingModelLoader(self.config)
        self.embedding_model = self.embedding_model_loader.load_embedding_model()

    def initialize_database(self, document_chunks: list, document_names: list):
        # Track token usage
        self.logger.info("Initializing vector_db")
        self.logger.info("\tUsing {} as db_option".format(self.db_option))
        if self.db_option == "FAISS":
            self.vector_db = FAISS.from_documents(
                documents=document_chunks, embedding=self.embedding_model
            )
        self.logger.info("Completed initializing vector_db")

    def create_database(self):
        data_loader = DataLoader(self.config)
        self.logger.info("Loading data")
        files, urls = self.load_files()
        document_chunks, document_names = data_loader.get_chunks(files, urls)
        self.logger.info("Completed loading data")

        self.create_embedding_model()
        self.initialize_database(document_chunks, document_names)

    def save_database(self):
        self.vector_db.save_local(
            os.path.join(
                self.config["embedding_options"]["db_path"],
                "db_"
                + self.config["embedding_options"]["db_option"]
                + "_"
                + self.config["embedding_options"]["model"],
            )
        )
        self.logger.info("Saved database")

    def load_database(self):
        self.create_embedding_model()
        self.vector_db = FAISS.load_local(
            os.path.join(
                self.config["embedding_options"]["db_path"],
                "db_"
                + self.config["embedding_options"]["db_option"]
                + "_"
                + self.config["embedding_options"]["model"],
            ),
            self.embedding_model,
        )
        self.logger.info("Loaded database")
        return self.vector_db


if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    print(config)
    vector_db = VectorDB(config)
    vector_db.create_database()
    vector_db.save_database()
