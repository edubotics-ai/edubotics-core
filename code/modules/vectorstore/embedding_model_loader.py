from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings

from modules.config.constants import *
import os


class EmbeddingModelLoader:
    def __init__(self, config):
        self.config = config

    def load_embedding_model(self):
        if self.config["vectorstore"]["model"] in ["text-embedding-ada-002"]:
            embedding_model = OpenAIEmbeddings(
                deployment="SL-document_embedder",
                model=self.config["vectorestore"]["model"],
                show_progress_bar=True,
                openai_api_key=OPENAI_API_KEY,
                disallowed_special=(),
            )
        else:
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.config["vectorstore"]["model"],
                model_kwargs={
                    "device": f"{self.config['device']}",
                    "token": f"{HUGGINGFACE_TOKEN}",
                    "trust_remote_code": True,
                },
            )
            # embedding_model = LlamaCppEmbeddings(
            #     model_path=os.path.abspath("storage/llama-7b.ggmlv3.q4_0.bin")
            # )

        return embedding_model
