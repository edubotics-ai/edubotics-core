from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from modules.constants import *
import os


class EmbeddingModelLoader:
    def __init__(self, config):
        self.config = config

    def load_embedding_model(self):
        if self.config["embedding_options"]["model"] in ["text-embedding-ada-002"]:
            embedding_model = OpenAIEmbeddings(
                deployment="SL-document_embedder",
                model=self.config["embedding_options"]["model"],
                show_progress_bar=True,
                openai_api_key=OPENAI_API_KEY,
            )
        else:
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            # embedding_model = LlamaCppEmbeddings(
            #     model_path=os.path.abspath("storage/llama-7b.ggmlv3.q4_0.bin")
            # )

        return embedding_model
