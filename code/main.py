from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import yaml
import logging
from dotenv import load_dotenv

from modules.chat.llm_tutor import LLMTutor
from modules.config.constants import *
from modules.chat.helpers import get_sources
from modules.chat_processor.chat_processor import ChatProcessor

global logger
# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="recording on CNNs?",
            message="Where can I find the recording for the lecture on Transfromers?",
            icon="/public/adv-screen-recorder-svgrepo-com.svg",
        ),
        cl.Starter(
            label="where's the slides?",
            message="When are the lectures? I can't find the schedule.",
            icon="/public/alarmy-svgrepo-com.svg",
        ),
        cl.Starter(
            label="Due Date?",
            message="When is the final project due?",
            icon="/public/calendar-samsung-17-svgrepo-com.svg",
        ),
        cl.Starter(
            label="Explain backprop.",
            message="I didnt understand the math behind backprop, could you explain it?",
            icon="/public/acastusphoton-svgrepo-com.svg",
        ),
    ]


# Adding option to select the chat profile
@cl.set_chat_profiles
async def chat_profile():
    return [
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
        cl.ChatProfile(
            name="Llama",
            markdown_description="Use the local LLM: **Tiny Llama**.",
        ),
    ]


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "AI Tutor"}
    return rename_dict.get(orig_author, orig_author)


# chainlit code
@cl.on_chat_start
async def start():
    with open("modules/config/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Ensure log directory exists
    log_directory = config["log_dir"]
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # File Handler
    log_file_path = (
        f"{log_directory}/tutor.log"  # Change this to your desired log file path
    )
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
    # msg = cl.Message(content=f"Starting the bot {chat_profile}...")
    # await msg.send()
    # msg.content = opening_message
    # await msg.update()

    tags = [chat_profile, config["vectorstore"]["db_option"]]
    chat_processor = ChatProcessor(config, tags=tags)
    cl.user_session.set("chain", chain)
    cl.user_session.set("counter", 0)
    cl.user_session.set("chat_processor", chat_processor)


@cl.on_chat_end
async def on_chat_end():
    await cl.Message(content="Sorry, I have to go now. Goodbye!").send()


@cl.on_message
async def main(message):
    global logger
    user = cl.user_session.get("user")
    chain = cl.user_session.get("chain")

    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)

    # if counter >= 3:  # Ensure the counter condition is checked
    #     await cl.Message(content="Your credits are up!").send()
    #     await on_chat_end()  # Call the on_chat_end function to handle the end of the chat
    #     return  # Exit the function to stop further processing
    # else:

    cb = cl.AsyncLangchainCallbackHandler()  # TODO: fix streaming here
    cb.answer_reached = True

    processor = cl.user_session.get("chat_processor")
    res = await processor.rag(message.content, chain, cb)
    try:
        answer = res["answer"]
    except:
        answer = res["result"]

    answer_with_sources, source_elements, sources_dict = get_sources(res, answer)
    processor._process(message.content, answer, sources_dict)

    answer_with_sources = answer_with_sources.replace("$$", "$")

    await cl.Message(content=answer_with_sources, elements=source_elements).send()
