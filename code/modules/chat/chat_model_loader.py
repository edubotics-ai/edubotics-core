from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
import os
from pathlib import Path
from huggingface_hub import hf_hub_download


class ChatModelLoader:
    def __init__(self, config):
        self.config = config
        self.huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    def _verify_model_cache(self, model_cache_path):
        hf_hub_download(
            repo_id=self.config["llm_params"]["local_llm_params"]["repo_id"],
            filename=self.config["llm_params"]["local_llm_params"]["filename"],
            cache_dir=model_cache_path,
        )
        return str(list(Path(model_cache_path).glob("*/snapshots/*/*.gguf"))[0])

    def load_chat_model(self):
        if self.config["llm_params"]["llm_loader"] in [
            "gpt-3.5-turbo-1106",
            "gpt-4",
            "gpt-4o-mini",
        ]:
            llm = ChatOpenAI(model_name=self.config["llm_params"]["llm_loader"])
        elif self.config["llm_params"]["llm_loader"] == "local_llm":
            n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            model_path = self._verify_model_cache(
                self.config["llm_params"]["local_llm_params"]["model_path"]
            )
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
            raise ValueError(
                f"Invalid LLM Loader: {self.config['llm_params']['llm_loader']}"
            )
        return llm
