from langchain_community.vectorstores import Chroma
from edubotics_core.vectorstore.base import VectorStoreBase
import os


class ChromaVectorStore(VectorStoreBase):
    def __init__(self, config):
        self.config = config
        self._init_vector_db()

    def _init_vector_db(self):
        self.chroma = Chroma()

    def create_database(self, document_chunks, embedding_model):
        self.vectorstore = self.chroma.from_documents(
            documents=document_chunks,
            embedding=embedding_model,
            persist_directory=os.path.join(
                self.config["vectorstore"]["db_path"],
                "db_"
                + self.config["vectorstore"]["db_option"]
                + "_"
                + self.config["vectorstore"]["model"],
            ),
        )

    def load_database(self, embedding_model):
        self.vectorstore = Chroma(
            persist_directory=os.path.join(
                self.config["vectorstore"]["db_path"],
                "db_"
                + self.config["vectorstore"]["db_option"]
                + "_"
                + self.config["vectorstore"]["model"],
            ),
            embedding_function=embedding_model,
        )
        return self.vectorstore

    def as_retriever(self):
        return self.vectorstore.as_retriever()

    def __len__(self):
        return len(self.vectorstore)
