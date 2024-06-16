from modules.vectorstore.faiss import FaissVectorStore
from modules.vectorstore.chroma import ChromaVectorStore
from modules.vectorstore.colbert import ColbertVectorStore


class VectorStore:
    def __init__(self, config):
        self.config = config
        self.vectorstore = None

    def _create_database(
        self,
        document_chunks,
        document_names,
        documents,
        document_metadata,
        embedding_model,
    ):
        if self.config["vectorstore"]["db_option"] == "FAISS":
            self.vectorstore = FaissVectorStore(self.config)
            self.vectorstore.create_database(document_chunks, embedding_model)
        elif self.config["vectorstore"]["db_option"] == "Chroma":
            self.vectorstore = ChromaVectorStore(self.config)
            self.vectorstore.create_database(document_chunks, embedding_model)
        elif self.config["vectorstore"]["db_option"] == "RAGatouille":
            self.vectorstore = ColbertVectorStore(self.config)
            self.vectorstore.create_database(
                documents, document_names, document_metadata
            )
        else:
            raise ValueError(
                "Invalid db_option: {}".format(self.config["vectorstore"]["db_option"])
            )

    def _load_database(self, embedding_model):
        if self.config["vectorstore"]["db_option"] == "FAISS":
            self.vectorstore = FaissVectorStore(self.config)
            return self.vectorstore.load_database(embedding_model)
        elif self.config["vectorstore"]["db_option"] == "Chroma":
            self.vectorstore = ChromaVectorStore(self.config)
            return self.vectorstore.load_database(embedding_model)
        elif self.config["vectorstore"]["db_option"] == "RAGatouille":
            self.vectorstore = ColbertVectorStore(self.config)
            return self.vectorstore.load_database()

    def _as_retriever(self):
        return self.vectorstore.as_retriever()

    def _get_vectorstore(self):
        return self.vectorstore
