from .helpers import VectorStoreRetrieverScore
from .base import BaseRetriever


class ChromaRetriever(BaseRetriever):
    def __init__(self):
        pass

    def return_retriever(self, db, config):
        retriever = VectorStoreRetrieverScore(
            vectorstore=db,
            # search_type="similarity_score_threshold",
            # search_kwargs={
            #     "score_threshold": self.config["vectorstore"][
            #         "score_threshold"
            #     ],
            #     "k": self.config["vectorstore"]["search_top_k"],
            # },
            search_kwargs={
                "k": config["vectorstore"]["search_top_k"],
            },
        )

        return retriever
