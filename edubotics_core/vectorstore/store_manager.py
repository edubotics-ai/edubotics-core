from edubotics_core.vectorstore.vectorstore import VectorStore
from edubotics_core.dataloader.helpers import get_urls_from_file
from edubotics_core.dataloader.webpage_crawler import WebpageCrawler
from edubotics_core.dataloader.data_loader import DataLoader
from edubotics_core.vectorstore.embedding_model_loader import EmbeddingModelLoader
import logging
import os
import time
import asyncio
import yaml
import argparse


class VectorStoreManager:
    def __init__(self, config, logger=None):
        self.config = config
        self.document_names = None

        # Set up logging to both console and a file
        self.logger = logger or self._setup_logging()
        self.webpage_crawler = WebpageCrawler()
        self.vector_db = VectorStore(self.config)

        self.logger.info("VectorDB instance instantiated")

    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Ensure log directory exists
            log_directory = self.config["log_dir"]
            os.makedirs(log_directory, exist_ok=True)

            # File Handler
            log_file_path = os.path.join(log_directory, "vector_db.log")
            file_handler = logging.FileHandler(log_file_path, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def load_files(self):
        files = os.listdir(self.config["vectorstore"]["data_path"])
        files = [
            os.path.join(self.config["vectorstore"]["data_path"], file)
            for file in files
            if file != "urls.txt"
        ]
        url_file_path = self.config["vectorstore"]["url_file_path"]
        urls = get_urls_from_file(url_file_path)
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
        if self.config["vectorstore"]["db_option"] in ["FAISS", "Chroma", "RAPTOR"]:
            self.embedding_model = self.create_embedding_model()
        else:
            self.embedding_model = None

        self.logger.info("Initializing vector_db")
        self.logger.info(
            "\tUsing {} as db_option".format(self.config["vectorstore"]["db_option"])
        )
        self.vector_db._create_database(
            document_chunks,
            document_names,
            documents,
            document_metadata,
            self.embedding_model,
        )

    def create_database(self):
        start_time = time.time()  # Start time for creating database
        data_loader = DataLoader(self.config, self.logger)
        self.logger.info("Loading data")
        local_files, urls = self.load_files()
        # print(f"Local files: {local_files}")
        # print(f"URLs: {urls}")
        files, webpages = self.webpage_crawler.clean_url_list(urls)
        files.extend(local_files)
        self.logger.info(f"Number of files: {len(files)}")
        self.logger.info(f"Number of webpages: {len(webpages)}")
        if f"{self.config['vectorstore']['url_file_path']}" in files:
            files.remove(f"{self.config['vectorstore']['url_file_path']}")  # cleanup
        (
            document_chunks,
            document_names,
            documents,
            document_metadata,
        ) = data_loader.get_chunks(files, webpages)
        num_documents = len(document_chunks)
        self.logger.info(f"Number of documents in the DB: {num_documents}")
        metadata_keys = list(document_metadata[0].keys()) if document_metadata else []
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
        if self.config["vectorstore"]["db_option"] in ["FAISS", "Chroma", "RAPTOR"]:
            self.embedding_model = self.create_embedding_model()
        else:
            self.embedding_model = None
        try:
            self.loaded_vector_db = self.vector_db._load_database(self.embedding_model)
        except Exception as e:
            raise ValueError(
                f"Error loading database, check if it exists. if not run python -m edubotics_core.vectorstore.store_manager / Resteart the HF Space: {e}"
            )
            # print(f"Creating database")
            # self.create_database()
            # self.loaded_vector_db = self.vector_db._load_database(self.embedding_model)
        end_time = time.time()  # End time for loading database
        self.logger.info(
            f"Time taken to load database {self.config['vectorstore']['db_option']}: {end_time - start_time} seconds"
        )
        self.logger.info("Loaded database")
        return self.loaded_vector_db

    def load_from_HF(self, HF_PATH):
        start_time = time.time()  # Start time for loading database
        self.vector_db._load_from_HF(HF_PATH)
        end_time = time.time()
        self.logger.info(
            f"Time taken to Download database {self.config['vectorstore']['db_option']} from Hugging Face: {end_time - start_time} seconds"
        )
        self.logger.info("Downloaded database")

    def __len__(self):
        return len(self.vector_db)


def main():
    # Add argument parsing for config files
    CWD = os.getcwd()
    parser = argparse.ArgumentParser(description="Load configuration files.")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the main config file",
        default=os.path.join(CWD, "config/config.yml"),
    )
    parser.add_argument(
        "--project_config_file",
        type=str,
        help="Path to the project config file",
        default=os.path.join(CWD, "config/project_config.yml"),
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    with open(args.project_config_file, "r") as f:
        project_config = yaml.safe_load(f)

    # combine the two configs
    config.update(project_config)
    print(config)
    print(f"Trying to create database with config: {config}")
    vector_db = VectorStoreManager(config)
    if config["vectorstore"]["load_from_HF"]:
        if (
            config["vectorstore"]["db_option"]
            in config["retriever"]["retriever_hf_paths"]
        ):
            vector_db.load_from_HF(
                HF_PATH=config["retriever"]["retriever_hf_paths"][
                    config["vectorstore"]["db_option"]
                ]
            )
        else:
            # print(f"HF_PATH not available for {config['vectorstore']['db_option']}")
            # print("Creating database")
            # vector_db.create_database()
            raise ValueError(
                f"HF_PATH not available for {config['vectorstore']['db_option']}"
            )
    else:
        vector_db.create_database()
    print("Created database")

    print("Trying to load the database")
    vector_db = VectorStoreManager(config)
    vector_db.load_database()
    print("Loaded database")

    print(f"View the logs at {config['log_dir']}/vector_db.log")


if __name__ == "__main__":
    main()
