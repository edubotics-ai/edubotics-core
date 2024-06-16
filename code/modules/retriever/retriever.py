from modules.retriever.faiss_retriever import FaissRetriever
from modules.retriever.chroma_retriever import ChromaRetriever
from modules.retriever.colbert_retriever import ColbertRetriever


class Retriever:
    def __init__(self, config):
        self.config = config
        self._create_retriever()

    def _create_retriever(self):
        if self.config["vectorstore"]["db_option"] == "FAISS":
            self.retriever = FaissRetriever()
        elif self.config["vectorstore"]["db_option"] == "Chroma":
            self.retriever = ChromaRetriever()
        elif self.config["vectorstore"]["db_option"] == "RAGatouille":
            self.retriever = ColbertRetriever()
        else:
            raise ValueError(
                "Invalid db_option: {}".format(self.config["vectorstore"]["db_option"])
            )

    def _return_retriever(self, db):
        return self.retriever.return_retriever(db, self.config)
