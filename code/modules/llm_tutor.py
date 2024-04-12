from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
import os

from modules.constants import *
from modules.helpers import get_prompt
from modules.chat_model_loader import ChatModelLoader
from modules.vector_db import VectorDB


class LLMTutor:
    def __init__(self, config, logger=None):
        self.config = config
        self.vector_db = VectorDB(config, logger=logger)
        if self.config["embedding_options"]["embedd_files"]:
            self.vector_db.create_database()
            self.vector_db.save_database()

    def set_custom_prompt(self):
        """
        Prompt template for QA retrieval for each vectorstore
        """
        prompt = get_prompt(self.config)
        # prompt = QA_PROMPT

        return prompt

    # Retrieval QA Chain
    def retrieval_qa_chain(self, llm, prompt, db):
        
        if self.config["llm_params"]["use_history"]:
            memory = ConversationBufferWindowMemory(
            k = self.config["llm_params"]["memory_window"], 
            memory_key="chat_history", return_messages=True, output_key="answer"
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(
                    search_kwargs={
                        "k": self.config["embedding_options"]["search_top_k"]
                    }
                ),
                return_source_documents=True,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
            )
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(
                    search_kwargs={
                        "k": self.config["embedding_options"]["search_top_k"]
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
            )
        return qa_chain

    # Loading the model
    def load_llm(self):
        chat_model_loader = ChatModelLoader(self.config)
        llm = chat_model_loader.load_chat_model()
        return llm

    # QA Model Function
    def qa_bot(self):
        db = self.vector_db.load_database()
        self.llm = self.load_llm()
        qa_prompt = self.set_custom_prompt()
        qa = self.retrieval_qa_chain(self.llm, qa_prompt, db)

        return qa

    # output function
    def final_result(query):
        qa_result = qa_bot()
        response = qa_result({"query": query})
        return response
