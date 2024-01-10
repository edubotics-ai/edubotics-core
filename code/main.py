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

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
print(config)
logger.info("Config file loaded")
logger.info(f"Config: {config}")
logger.info("Creating llm_tutor instance")
llm_tutor = LLMTutor(config, logger=logger)


# chainlit code
@cl.on_chat_start
async def start():
    chain = llm_tutor.qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hey, What Can I Help You With?"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # res=await chain.acall(message, callbacks=[cb])
    res = await chain.acall(message.content, callbacks=[cb])
    # print(f"response: {res}")
    try:
        answer = res["answer"]
    except:
        answer = res["result"]
    print(f"answer: {answer}")
    source_elements_dict = {}
    source_elements = []
    found_sources = []

    for idx, source in enumerate(res["source_documents"]):
        title = source.metadata["source"]

        if title not in source_elements_dict:
            source_elements_dict[title] = {
                "page_number": [source.metadata["page"]],
                "url": source.metadata["source"],
                "content": source.page_content,
            }

        else:
            source_elements_dict[title]["page_number"].append(source.metadata["page"])
        source_elements_dict[title][
            "content_" + str(source.metadata["page"])
        ] = source.page_content
        # sort the page numbers
        # source_elements_dict[title]["page_number"].sort()

    for title, source in source_elements_dict.items():
        # create a string for the page numbers
        page_numbers = ", ".join([str(x) for x in source["page_number"]])
        text_for_source = f"Page Number(s): {page_numbers}\nURL: {source['url']}"
        source_elements.append(cl.Pdf(name="File", path=title))
        found_sources.append("File")
        # for pn in source["page_number"]:
        #     source_elements.append(
        #         cl.Text(name=str(pn), content=source["content_"+str(pn)])
        #     )
        #     found_sources.append(str(pn))

    if found_sources:
        answer += f"\nSource:{', '.join(found_sources)}"
    else:
        answer += f"\nNo source found."

    await cl.Message(content=answer, elements=source_elements).send()
