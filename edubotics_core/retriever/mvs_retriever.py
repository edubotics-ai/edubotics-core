from edubotics_core.retriever.faiss_retriever import FaissRetriever
import os


class MvsRetriever:
    def __init__(self, config):
        self.config = config
        self.return_top_k = config["vectorstore"]["search_top_k"]

    def load_retrievers(self):
        self.retrievers = {}
        for content_type in self.config["vectorstore"]["content_types"]:
            path = os.path.join(
                self.config["vectorstore"]["db_path"], "mvs", f"FAISS_{content_type}"
            )
            self.retrievers[content_type] = FaissRetriever().return_retriever(
                path, self.config
            )

    def return_retriever(self):
        return self.retrievers
