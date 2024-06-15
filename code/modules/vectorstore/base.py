class VectorStoreBase:
    def __init__(self, config):
        self.config = config

    def _init_vector_db(self):
        raise NotImplementedError

    def create_database(self, database_name):
        raise NotImplementedError

    def load_database(self, database_name):
        raise NotImplementedError

    def as_retriever(self):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
