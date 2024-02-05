from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
import chainlit as cl
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import yaml
import logging
from dotenv import load_dotenv

from modules.llm_tutor import LLMTutor
from modules.constants import *
from modules.helpers import get_sources


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File Handler
log_file_path = "log_file.log"  # Change this to your desired log file path
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Adding option to select the chat profile
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Llama",
            markdown_description="Use the local LLM: **Tiny Llama**.",
        ),
        # cl.ChatProfile(
        #     name="Mistral",
        #     markdown_description="Use the local LLM: **Mistral**.",
        # ),
        cl.ChatProfile(
            name="gpt-3.5-turbo-1106",
            markdown_description="Use OpenAI API for **gpt-3.5-turbo-1106**.",
        ),
        cl.ChatProfile(
            name="gpt-4",
            markdown_description="Use OpenAI API for **gpt-4**.",
        ),
    ]


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "AI Tutor"}
    return rename_dict.get(orig_author, orig_author)


# chainlit code
@cl.on_chat_start
async def start():
    with open("code/config.yml", "r") as f:
        config = yaml.safe_load(f)
        print(config)
        logger.info("Config file loaded")
        logger.info(f"Config: {config}")
        logger.info("Creating llm_tutor instance")

    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile is not None:
        if chat_profile.lower() in ["gpt-3.5-turbo-1106", "gpt-4"]:
            config["llm_params"]["llm_loader"] = "openai"
            config["llm_params"]["openai_params"]["model"] = chat_profile.lower()
        elif chat_profile.lower() == "llama":
            config["llm_params"]["llm_loader"] = "local_llm"
            config["llm_params"]["local_llm_params"]["model"] = LLAMA_PATH
            config["llm_params"]["local_llm_params"]["model_type"] = "llama"
        elif chat_profile.lower() == "mistral":
            config["llm_params"]["llm_loader"] = "local_llm"
            config["llm_params"]["local_llm_params"]["model"] = MISTRAL_PATH
            config["llm_params"]["local_llm_params"]["model_type"] = "mistral"

        else:
            pass

    llm_tutor = LLMTutor(config, logger=logger)

    chain = llm_tutor.qa_bot()
    model = config["llm_params"]["local_llm_params"]["model"]
    msg = cl.Message(content=f"Starting the bot {model}...")
    await msg.send()
    msg.content = f"Hey, What Can I Help You With?\n\nYou can me ask me questions about the course logistics, course content, about the final project, or anything else!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    user = cl.user_session.get("user")
    chain = cl.user_session.get("chain")
    # cb = cl.AsyncLangchainCallbackHandler(
    #     stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    # )
    # cb.answer_reached = True
    # res=await chain.acall(message, callbacks=[cb])
    res = await chain.acall(message.content)
    print(f"response: {res}")
    try:
        answer = res["answer"]
    except:
        answer = res["result"]
    print(f"answer: {answer}")

    answer_with_sources, source_elements = get_sources(res, answer)

    await cl.Message(content=answer_with_sources, elements=source_elements).send()
