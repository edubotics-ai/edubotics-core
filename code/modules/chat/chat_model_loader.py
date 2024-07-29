from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer
from langchain_community.llms import LlamaCpp
import torch
import transformers
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class ChatModelLoader:
    def __init__(self, config):
        self.config = config
        self.huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    def _verify_model_cache(self, model_cache_path):
        hf_hub_download(
            repo_id=self.config["llm_params"]["local_llm_params"]["repo_id"],
            filename=self.config["llm_params"]["local_llm_params"]["filename"],
            cache_dir=model_cache_path
            )
        return str(list(Path(model_cache_path).glob("*/snapshots/*/*.gguf"))[0])

    def load_chat_model(self):
        if self.config["llm_params"]["llm_loader"] == "openai":
            llm = ChatOpenAI(
                model_name=self.config["llm_params"]["openai_params"]["model"]
            )
        elif self.config["llm_params"]["llm_loader"] == "local_llm":
            n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            model_path = self._verify_model_cache(self.config["llm_params"]["local_llm_params"]["model"])
            llm = LlamaCpp(
                model_path=model_path,
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
            raise ValueError("Invalid LLM Loader")
        return llm
