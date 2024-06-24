from .helpers import VectorStoreRetrieverScore
from .base import BaseRetriever


class RaptorRetriever(BaseRetriever):
    def __init__(self):
        pass

    def return_retriever(self, db, config):
        retriever = VectorStoreRetrieverScore(
            vectorstore=db,
            search_kwargs={
                "k": config["vectorstore"]["search_top_k"],
            },
        )
        return retriever
