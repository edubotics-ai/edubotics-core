from modules.retriever.faiss_retriever import FaissRetriever
from modules.retriever.chroma_retriever import ChromaRetriever
from modules.retriever.colbert_retriever import ColbertRetriever
from modules.retriever.raptor_retriever import RaptorRetriever


class Retriever:
    def __init__(self, config):
        self.config = config
        self.retriever_classes = {
            "FAISS": FaissRetriever,
            "Chroma": ChromaRetriever,
            "RAGatouille": ColbertRetriever,
            "RAPTOR": RaptorRetriever,
        }
        self._create_retriever()

    def _create_retriever(self):
        db_option = self.config["vectorstore"]["db_option"]
        retriever_class = self.retriever_classes.get(db_option)
        if not retriever_class:
            raise ValueError(f"Invalid db_option: {db_option}")
        self.retriever = retriever_class()

    def _return_retriever(self, db):
        return self.retriever.return_retriever(db, self.config)
