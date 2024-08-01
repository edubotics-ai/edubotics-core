# template for vector store classes


class VectorStoreBase:
    def __init__(self, config):
        self.config = config

    def _init_vector_db(self):
        """
        Creates a vector store object
        """
        raise NotImplementedError

    def create_database(self):
        """
        Populates the vector store with documents
        """
        raise NotImplementedError

    def load_database(self):
        """
        Loads the vector store from disk
        """
        raise NotImplementedError

    def as_retriever(self):
        """
        Returns the vector store as a retriever
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
