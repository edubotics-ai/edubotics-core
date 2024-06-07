from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings

try:
    from modules.constants import *
except:
    from constants import *
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
                disallowed_special=(),
            )
        else:
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.config["embedding_options"]["model"],
                model_kwargs={
                    "device": "cpu",
                    "token": f"{HUGGINGFACE_TOKEN}",
                    "trust_remote_code": True,
                },
            )
            # embedding_model = LlamaCppEmbeddings(
            #     model_path=os.path.abspath("storage/llama-7b.ggmlv3.q4_0.bin")
            # )

        return embedding_model
