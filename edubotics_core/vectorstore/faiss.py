from langchain_community.vectorstores import FAISS
from edubotics_core.vectorstore.base import VectorStoreBase
import os


class FAISS(FAISS):
    """To add length property to FAISS class"""

    def __len__(self):
        return self.index.ntotal


class FaissVectorStore(VectorStoreBase):
    def __init__(self, config):
        self.config = config
        self._init_vector_db()
        self.local_path = os.path.join(
            self.config["vectorstore"]["db_path"],
            "db_"
            + self.config["vectorstore"]["db_option"]
            + "_"
            + self.config["vectorstore"]["model"]
            + "_"
            + config["splitter_options"]["chunking_mode"],
        )

    def _init_vector_db(self):
        self.faiss = FAISS(
            embedding_function=None, index=0, index_to_docstore_id={}, docstore={}
        )

    def create_database(self, document_chunks, embedding_model):
        self.vectorstore = self.faiss.from_documents(
            documents=document_chunks, embedding=embedding_model
        )
        self.vectorstore.save_local(self.local_path)

    def load_database(self, embedding_model):
        self.vectorstore = self.faiss.load_local(
            self.local_path,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore

    def as_retriever(self):
        return self.vectorstore.as_retriever()

    def __len__(self):
        return len(self.vectorstore)
