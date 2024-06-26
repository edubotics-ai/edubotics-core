from ragatouille import RAGPretrainedModel
from modules.vectorstore.base import VectorStoreBase
import os


class ColbertVectorStore(VectorStoreBase):
    def __init__(self, config):
        self.config = config
        self._init_vector_db()

    def _init_vector_db(self):
        self.colbert = RAGPretrainedModel.from_pretrained(
            "colbert-ir/colbertv2.0",
            index_root=os.path.join(
                self.config["vectorstore"]["db_path"],
                "db_" + self.config["vectorstore"]["db_option"],
            ),
        )

    def create_database(self, documents, document_names, document_metadata):
        index_path = self.colbert.index(
            index_name="new_idx",
            collection=documents,
            document_ids=document_names,
            document_metadatas=document_metadata,
        )

    def load_database(self):
        path = os.path.join(
            self.config["vectorstore"]["db_path"],
            "db_" + self.config["vectorstore"]["db_option"],
        )
        self.vectorstore = RAGPretrainedModel.from_index(
            f"{path}/colbert/indexes/new_idx"
        )
        return self.vectorstore

    def as_retriever(self):
        return self.vectorstore.as_retriever()
