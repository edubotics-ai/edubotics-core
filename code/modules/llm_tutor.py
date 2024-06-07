from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
import os
from modules.constants import *
from modules.helpers import get_prompt
from modules.chat_model_loader import ChatModelLoader
from modules.vector_db import VectorDB, VectorDBScore
from typing import Dict, Any, Optional
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun
import inspect
from langchain.chains.conversational_retrieval.base import _get_chat_history


class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        print(f"chat_history_str: {chat_history_str}")
        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = await self.question_generator.arun(
                question=question, chat_history=chat_history_str, callbacks=callbacks
            )
        else:
            new_question = question
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = await self._aget_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(new_question, inputs)  # type: ignore[call-arg]

        output: Dict[str, Any] = {}
        if self.response_if_no_docs_found is not None and len(docs) == 0:
            output[self.output_key] = self.response_if_no_docs_found
        else:
            new_inputs = inputs.copy()
            if self.rephrase_question:
                new_inputs["question"] = new_question
            new_inputs["chat_history"] = chat_history_str

            # Prepare the final prompt with metadata
            context = "\n\n".join(
                [
                    f"Document content: {doc.page_content}\nMetadata: {doc.metadata}"
                    for doc in docs
                ]
            )
            final_prompt = f"""
                You are an AI Tutor for the course DS598, taught by Prof. Thomas Gardos. Use the following pieces of information to answer the user's question. 
                If you don't know the answer, just say that you don't know—don't try to make up an answer. 
                Use the chat history to answer the question only if it's relevant; otherwise, ignore it. The context for the answer will be under "Document context:". 
                Use the metadata from each document to guide the user to the correct sources. 
                The context is ordered by relevance to the question. Give more weight to the most relevant documents.
                Talk in a friendly and personalized manner, similar to how you would speak to a friend who needs help. Make the conversation engaging and avoid sounding repetitive or robotic.

                Chat History:
                {chat_history_str}

                Context:
                {context}

                Question: {new_question}
                AI Tutor:
                """

            new_inputs["input"] = final_prompt
            new_inputs["question"] = final_prompt
            output["final_prompt"] = final_prompt

            answer = await self.combine_docs_chain.arun(
                input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
            )
            output[self.output_key] = answer

        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_question
        return output


class LLMTutor:
    def __init__(self, config, logger=None):
        self.config = config
        self.llm = self.load_llm()
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
        if self.config["embedding_options"]["db_option"] in ["FAISS", "Chroma"]:
            retriever = VectorDBScore(
                vectorstore=db,
                # search_type="similarity_score_threshold",
                # search_kwargs={
                #     "score_threshold": self.config["embedding_options"][
                #         "score_threshold"
                #     ],
                #     "k": self.config["embedding_options"]["search_top_k"],
                # },
            )
        elif self.config["embedding_options"]["db_option"] == "RAGatouille":
            retriever = db.as_langchain_retriever(
                k=self.config["embedding_options"]["search_top_k"]
            )
        if self.config["llm_params"]["use_history"]:
            memory = ConversationSummaryBufferMemory(
                llm = llm,
                k=self.config["llm_params"]["memory_window"],
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                max_token_limit=128,
            )
            qa_chain = CustomConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
            )
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
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
        qa_prompt = self.set_custom_prompt()
        qa = self.retrieval_qa_chain(self.llm, qa_prompt, db)

        return qa

    # output function
    def final_result(query):
        qa_result = qa_bot()
        response = qa_result({"query": query})
        return response
