from .base import BaseRetriever


class ColbertRetriever(BaseRetriever):
    def __init__(self):
        pass

    def return_retriever(self, db, config):
        retriever = db.as_retriever()
        return retriever
