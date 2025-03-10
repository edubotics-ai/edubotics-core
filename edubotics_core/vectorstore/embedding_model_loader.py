from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from edubotics_core.config.constants import OPENAI_API_KEY, HUGGINGFACE_TOKEN


class EmbeddingModelLoader:
    def __init__(self, config):
        self.config = config
        print("HUGGINGFACE_TOKEN", HUGGINGFACE_TOKEN)

    def load_embedding_model(self):
        if self.config["vectorstore"]["model"] in ["text-embedding-ada-002"]:
            embedding_model = OpenAIEmbeddings(
                deployment="SL-document_embedder",
                model=self.config["vectorstore"]["model"],
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

        return embedding_model
