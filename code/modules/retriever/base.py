# template for retriever classes


class BaseRetriever:
    def __init__(self, config):
        self.config = config

    def return_retriever(self):
        """
        Returns the retriever object
        """
        raise NotImplementedError
