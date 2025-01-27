from pydantic import BaseModel, conint, confloat, HttpUrl
from typing import Optional, List
import yaml


class FaissParams(BaseModel):
    index_path: str = "vectorstores/faiss.index"
    index_type: str = "Flat"  # Options: [Flat, HNSW, IVF]
    index_dimension: conint(gt=0) = 384
    index_nlist: conint(gt=0) = 100
    index_nprobe: conint(gt=0) = 10


class ColbertParams(BaseModel):
    index_name: str = "new_idx"


class VectorStoreConfig(BaseModel):
    load_from_HF: bool = True
    reparse_files: bool = True
    data_path: str = "storage/data"
    url_file_path: str = "storage/data/urls.txt"
    expand_urls: bool = True
    db_option: str = "RAGatouille"  # Options: [FAISS, Chroma, RAGatouille, RAPTOR]
    db_path: str = "vectorstores"
    model: str = (
        # Options: [sentence-transformers/all-MiniLM-L6-v2, text-embedding-ada-002]
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    search_top_k: conint(gt=0) = 3
    score_threshold: confloat(ge=0.0, le=1.0) = 0.2

    faiss_params: Optional[FaissParams] = None
    colbert_params: Optional[ColbertParams] = None


class OpenAIParams(BaseModel):
    temperature: confloat(ge=0.0, le=1.0) = 0.7


class LocalLLMParams(BaseModel):
    temperature: confloat(ge=0.0, le=1.0) = 0.7
    repo_id: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"  # HuggingFace repo id
    filename: str = (
        "tinyllama-1.1b-chat-v1.0.Q5_0.gguf"  # Specific name of gguf file in the repo
    )
    model_path: str = (
        "storage/models/tinyllama-1.1b-chat-v1.0.Q5_0.gguf"  # Path to the model file
    )


class LLMParams(BaseModel):
    llm_arch: str = "langchain"  # Options: [langchain]
    use_history: bool = True
    generate_follow_up: bool = False
    memory_window: conint(ge=1) = 3
    llm_style: str = "Normal"  # Options: [Normal, ELI5]
    llm_loader: str = (
        "gpt-4o-mini"  # Options: [local_llm, gpt-3.5-turbo-1106, gpt-4, gpt-4o-mini]
    )
    openai_params: Optional[OpenAIParams] = None
    local_llm_params: Optional[LocalLLMParams] = None
    stream: bool = False
    pdf_reader: str = "gpt"  # Options: [llama, pymupdf, gpt]


class ChatLoggingConfig(BaseModel):
    log_chat: bool = True
    platform: str = "literalai"
    callbacks: bool = True


class SplitterOptions(BaseModel):
    use_splitter: bool = True
    split_by_token: bool = True
    remove_leftover_delimiters: bool = True
    remove_chunks: bool = False
    chunking_mode: str = "semantic"  # Options: [fixed, semantic]
    chunk_size: conint(gt=0) = 300
    chunk_overlap: conint(ge=0) = 30
    chunk_separators: List[str] = ["\n\n", "\n", " ", ""]
    front_chunks_to_remove: Optional[conint(ge=0)] = None
    last_chunks_to_remove: Optional[conint(ge=0)] = None
    delimiters_to_remove: List[str] = ["\t", "\n", "   ", "  "]


class RetrieverConfig(BaseModel):
    retriever_hf_paths: dict[str, str] = {"RAGatouille": "XThomasBU/Colbert_Index"}


class MetadataConfig(BaseModel):
    metadata_links: List[HttpUrl] = [
        "https://dl4ds.github.io/sp2024/lectures/",
        "https://dl4ds.github.io/sp2024/schedule/",
    ]
    slide_base_link: HttpUrl = "https://dl4ds.github.io"


class TokenConfig(BaseModel):
    cooldown_time: conint(gt=0) = 60
    regen_time: conint(gt=0) = 180
    tokens_left: conint(gt=0) = 2000
    all_time_tokens_allocated: conint(gt=0) = 1000000


class MiscConfig(BaseModel):
    github_repo: HttpUrl = "https://github.com/edubotics-ai/edubot-core"
    docs_website: HttpUrl = "https://dl4ds.github.io/dl4ds_tutor/"


class APIConfig(BaseModel):
    timeout: conint(gt=0) = 60


class Config(BaseModel):
    log_dir: str = "storage/logs"
    log_chunk_dir: str = "storage/logs/chunks"
    device: str = "cpu"  # Options: ['cuda', 'cpu']

    vectorstore: VectorStoreConfig
    llm_params: LLMParams
    chat_logging: ChatLoggingConfig
    splitter_options: SplitterOptions
    retriever: RetrieverConfig
    metadata: MetadataConfig
    token_config: TokenConfig
    misc: MiscConfig
    api_config: APIConfig


class ConfigManager:
    def __init__(self, config_path: str, project_config_path: str):
        self.config_path = config_path
        self.project_config_path = project_config_path
        self.config = self.load_config()
        self.validate_config()

    def load_config(self) -> Config:
        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)

        with open(self.project_config_path, "r") as f:
            project_config_data = yaml.safe_load(f)

        # Merge the two configurations
        merged_config = {**config_data, **project_config_data}

        return Config(**merged_config)

    def get_config(self) -> Config:
        return ConfigWrapper(self.config)

    def validate_config(self):
        # If any required fields are missing, raise an error
        # required_fields = [
        #     "vectorstore", "llm_params", "chat_logging", "splitter_options",
        #     "retriever", "metadata", "token_config", "misc", "api_config"
        # ]
        # for field in required_fields:
        #     if not hasattr(self.config, field):
        #         raise ValueError(f"Missing required configuration field: {field}")

        # # Validate types of specific fields
        # if not isinstance(self.config.vectorstore, VectorStoreConfig):
        #     raise TypeError("vectorstore must be an instance of VectorStoreConfig")
        # if not isinstance(self.config.llm_params, LLMParams):
        #     raise TypeError("llm_params must be an instance of LLMParams")
        pass


class ConfigWrapper:
    def __init__(self, config: Config):
        self._config = config

    def __getitem__(self, key):
        return getattr(self._config, key)

    def __getattr__(self, name):
        return getattr(self._config, name)

    def dict(self):
        return self._config.dict()


# Usage
config_manager = ConfigManager(
    config_path="config/config.yml", project_config_path="config/project_config.yml"
)
# config = config_manager.get_config().dict()
