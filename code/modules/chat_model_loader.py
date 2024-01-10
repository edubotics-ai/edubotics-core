from langchain_community.chat_models import ChatOpenAI
from langchain.llms import CTransformers


class ChatModelLoader:
    def __init__(self, config):
        self.config = config

    def load_chat_model(self):
        if self.config["llm_params"]["llm_loader"] == "openai":
            llm = ChatOpenAI(
                model_name=self.config["llm_params"]["openai_params"]["model"]
            )
        elif self.config["llm_params"]["llm_loader"] == "Ctransformers":
            llm = CTransformers(
                model=self.config["llm_params"]["ctransformers_params"]["model"],
                model_type=self.config["llm_params"]["ctransformers_params"][
                    "model_type"
                ],
                max_new_tokens=512,
                temperature=0.5,
            )
        else:
            raise ValueError("Invalid LLM Loader")
        return llm
