class BaseRetriever:
    def __init__(self, config):
        self.config = config

    def return_retriever(self):
        raise NotImplementedError
