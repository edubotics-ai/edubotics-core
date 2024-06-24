from modules.vectorstore.faiss import FaissVectorStore
from modules.vectorstore.chroma import ChromaVectorStore
from modules.vectorstore.colbert import ColbertVectorStore
from modules.vectorstore.raptor import RAPTORVectoreStore


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

    def _as_retriever(self):
        return self.vectorstore.as_retriever()

    def _get_vectorstore(self):
        return self.vectorstore
