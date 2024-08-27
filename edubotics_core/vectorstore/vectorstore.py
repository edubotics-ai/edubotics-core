from edubotics_core.vectorstore.faiss import FaissVectorStore
from edubotics_core.vectorstore.chroma import ChromaVectorStore
from edubotics_core.vectorstore.colbert import ColbertVectorStore
from edubotics_core.vectorstore.raptor import RAPTORVectoreStore
from huggingface_hub import snapshot_download
import os
import shutil


class VectorStore:
    def __init__(self, config):
        self.config = config
        self.vectorstore = None
        self.vectorstore_classes = {
            "FAISS": FaissVectorStore,
            "Chroma": ChromaVectorStore,
            "RAGatouille": ColbertVectorStore,
            "RAPTOR": RAPTORVectoreStore,
        }

    def _create_database(
        self,
        document_chunks,
        document_names,
        documents,
        document_metadata,
        embedding_model,
    ):
        db_option = self.config["vectorstore"]["db_option"]
        vectorstore_class = self.vectorstore_classes.get(db_option)
        if not vectorstore_class:
            raise ValueError(f"Invalid db_option: {db_option}")

        self.vectorstore = vectorstore_class(self.config)

        if db_option == "RAGatouille":
            self.vectorstore.create_database(
                documents, document_names, document_metadata
            )
        else:
            self.vectorstore.create_database(document_chunks, embedding_model)

    def _load_database(self, embedding_model):
        db_option = self.config["vectorstore"]["db_option"]
        vectorstore_class = self.vectorstore_classes.get(db_option)
        if not vectorstore_class:
            raise ValueError(f"Invalid db_option: {db_option}")

        self.vectorstore = vectorstore_class(self.config)

        if db_option == "RAGatouille":
            return self.vectorstore.load_database()
        else:
            return self.vectorstore.load_database(embedding_model)

    def _load_from_HF(self, HF_PATH):
        # Download the snapshot from Hugging Face Hub
        # Note: Download goes to the cache directory
        snapshot_path = snapshot_download(
            repo_id=HF_PATH,
            repo_type="dataset",
            force_download=True,
        )

        # Move the downloaded files to the desired directory
        target_path = os.path.join(
            self.config["vectorstore"]["db_path"],
            "db_" + self.config["vectorstore"]["db_option"],
        )

        # Create target path if it doesn't exist
        os.makedirs(target_path, exist_ok=True)

        # move all files and directories from snapshot_path to target_path
        # target path is used while loading the database
        for item in os.listdir(snapshot_path):
            s = os.path.join(snapshot_path, item)
            d = os.path.join(target_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    def _as_retriever(self):
        return self.vectorstore.as_retriever()

    def _get_vectorstore(self):
        return self.vectorstore

    def __len__(self):
        return self.vectorstore.__len__()
