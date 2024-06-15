from modules.vectorstore.faiss import FaissVectorStore
from modules.vectorstore.chroma import ChromaVectorStore
from modules.vectorstore.colbert import ColbertVectorStore
from modules.vectorstore.helpers import *
from modules.dataloader.webpage_crawler import WebpageCrawler
from modules.dataloader.data_loader import DataLoader
from modules.dataloader.helpers import *
from modules.vectorstore.embedding_model_loader import EmbeddingModelLoader
import logging
import os
import time
import asyncio


class VectorStoreManager:
    def __init__(self, config, logger=None):
        self.config = config
        self.db_option = config["vectorstore"]["db_option"]
        self.document_names = None

        # Set up logging to both console and a file
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Ensure log directory exists
            log_directory = self.config["log_dir"]
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)

            # File Handler
            log_file_path = f"{log_directory}/vector_db.log"  # Change this to your desired log file path
            file_handler = logging.FileHandler(log_file_path, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            self.logger = logger

        self.webpage_crawler = WebpageCrawler()

        self.logger.info("VectorDB instance instantiated")

    def load_files(self):
        files = os.listdir(self.config["vectorstore"]["data_path"])
        files = [
            os.path.join(self.config["vectorstore"]["data_path"], file)
            for file in files
        ]
        urls = get_urls_from_file(self.config["vectorstore"]["url_file_path"])
        if self.config["vectorstore"]["expand_urls"]:
            all_urls = []
            for url in urls:
                loop = asyncio.get_event_loop()
                all_urls.extend(
                    loop.run_until_complete(
                        self.webpage_crawler.get_all_pages(
                            url, url
                        )  # only get child urls, if you want to get all urls, replace the second argument with the base url
                    )
                )
            urls = all_urls
        return files, urls

    def create_embedding_model(self):
        self.logger.info("Creating embedding function")
        embedding_model_loader = EmbeddingModelLoader(self.config)
        embedding_model = embedding_model_loader.load_embedding_model()
        return embedding_model

    def initialize_database(
        self,
        document_chunks: list,
        document_names: list,
        documents: list,
        document_metadata: list,
    ):
        if self.db_option in ["FAISS", "Chroma"]:
            self.embedding_model = self.create_embedding_model()

        self.logger.info("Initializing vector_db")
        self.logger.info("\tUsing {} as db_option".format(self.db_option))
        if self.db_option == "FAISS":
            self.vector_db = FaissVectorStore(self.config)
            self.vector_db.create_database(document_chunks, self.embedding_model)
        elif self.db_option == "Chroma":
            self.vector_db = ChromaVectorStore(self.config)
            self.vector_db.create_database(document_chunks, self.embedding_model)
        elif self.db_option == "RAGatouille":
            self.vector_db = ColbertVectorStore(self.config)
            self.vector_db.create_database(documents, document_names, document_metadata)

    def create_database(self):
        start_time = time.time()  # Start time for creating database
        data_loader = DataLoader(self.config, self.logger)
        self.logger.info("Loading data")
        files, urls = self.load_files()
        files, webpages = self.webpage_crawler.clean_url_list(urls)
        self.logger.info(f"Number of files: {len(files)}")
        self.logger.info(f"Number of webpages: {len(webpages)}")
        if f"{self.config['vectorstore']['url_file_path']}" in files:
            files.remove(f"{self.config['vectorstores']['url_file_path']}")  # cleanup
        document_chunks, document_names, documents, document_metadata = (
            data_loader.get_chunks(files, webpages)
        )
        num_documents = len(document_chunks)
        self.logger.info(f"Number of documents in the DB: {num_documents}")
        metadata_keys = list(document_metadata[0].keys())
        self.logger.info(f"Metadata keys: {metadata_keys}")
        self.logger.info("Completed loading data")
        self.initialize_database(
            document_chunks, document_names, documents, document_metadata
        )
        end_time = time.time()  # End time for creating database
        self.logger.info("Created database")
        self.logger.info(
            f"Time taken to create database: {end_time - start_time} seconds"
        )

    def load_database(self):
        start_time = time.time()  # Start time for loading database
        if self.db_option in ["FAISS", "Chroma"]:
            self.embedding_model = self.create_embedding_model()
        if self.db_option == "FAISS":
            self.vector_db = FaissVectorStore(self.config)
            self.loaded_vector_db = self.vector_db.load_database(self.embedding_model)
        elif self.db_option == "Chroma":
            self.vector_db = ChromaVectorStore(self.config)
            self.loaded_vector_db = self.vector_db.load_database(self.embedding_model)
        elif self.db_option == "RAGatouille":
            self.vector_db = ColbertVectorStore(self.config)
            self.loaded_vector_db = self.vector_db.load_database()
        end_time = time.time()  # End time for loading database
        self.logger.info(
            f"Time taken to load database: {end_time - start_time} seconds"
        )
        self.logger.info("Loaded database")
        return self.loaded_vector_db


if __name__ == "__main__":
    import yaml

    with open("modules/config/config.yml", "r") as f:
        config = yaml.safe_load(f)
    print(config)
    print(f"Trying to create database with config: {config}")
    vector_db = VectorStoreManager(config)
    vector_db.create_database()
    print("Created database")

    print(f"Trying to load the database")
    vector_db = VectorStoreManager(config)
    vector_db.load_database()
    print("Loaded database")

    print(f"View the logs at {config['log_dir']}/vector_db.log")
