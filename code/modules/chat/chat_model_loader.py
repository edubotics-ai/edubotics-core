from langchain_openai import ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer
from langchain_community.llms import LlamaCpp
import torch
import transformers
import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from modules.config.constants import LLAMA_PATH


class ChatModelLoader:
    def __init__(self, config):
        self.config = config
        self.huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    def load_chat_model(self):
        if self.config["llm_params"]["llm_loader"] in ["gpt-3.5-turbo-1106", "gpt-4"]:
            llm = ChatOpenAI(model_name=self.config["llm_params"]["llm_loader"])
        elif self.config["llm_params"]["llm_loader"] == "local_llm":
            n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            llm = LlamaCpp(
                model_path=LLAMA_PATH,
                n_batch=n_batch,
                n_ctx=2048,
                f16_kv=True,
                verbose=True,
                n_threads=2,
                temperature=self.config["llm_params"]["local_llm_params"][
                    "temperature"
                ],
            )
        else:
            raise ValueError(
                f"Invalid LLM Loader: {self.config['llm_params']['llm_loader']}"
            )
        return llm
