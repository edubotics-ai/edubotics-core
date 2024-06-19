from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
import os
from modules.config.constants import *
from modules.chat.helpers import get_prompt
from modules.chat.chat_model_loader import ChatModelLoader
from modules.vectorstore.store_manager import VectorStoreManager

from modules.retriever.retriever import Retriever

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun
import inspect
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain_core.messages import BaseMessage

CHAT_TURN_TYPE = Union[Tuple[str, str], BaseMessage]

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI


class CustomConversationalRetrievalChain(ConversationalRetrievalChain):

    def _get_chat_history(self, chat_history: List[CHAT_TURN_TYPE]) -> str:
        _ROLE_MAP = {"human": "Student: ", "ai": "AI Tutor: "}
        buffer = ""
        for dialogue_turn in chat_history:
            if isinstance(dialogue_turn, BaseMessage):
                role_prefix = _ROLE_MAP.get(
                    dialogue_turn.type, f"{dialogue_turn.type}: "
                )
                buffer += f"\n{role_prefix}{dialogue_turn.content}"
            elif isinstance(dialogue_turn, tuple):
                human = "Student: " + dialogue_turn[0]
                ai = "AI Tutor: " + dialogue_turn[1]
                buffer += "\n" + "\n".join([human, ai])
            else:
                raise ValueError(
                    f"Unsupported chat history format: {type(dialogue_turn)}."
                    f" Full chat history: {chat_history} "
                )
        return buffer

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self._get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        if chat_history_str:
            # callbacks = _run_manager.get_child()
            # new_question = await self.question_generator.arun(
            #     question=question, chat_history=chat_history_str, callbacks=callbacks
            # )
            system = (
                "You are someone that rephrases statements. Rephrase the student's question to add context from their chat history if relevant, ensuring it remains from the student's point of view. "
                "Incorporate relevant details from the chat history to make the question clearer and more specific."
                "Do not change the meaning of the original statement, and maintain the student's tone and perspective. "
                "If the question is conversational and doesn't require context, do not rephrase it. "
                "Example: If the student previously asked about backpropagation in the context of deep learning and now asks 'what is it', rephrase to 'What is backprogatation.'. "
                "Example: Do not rephrase if the user is asking something specific like 'cool, suggest a project with transformers to use as my final project'"
                "Chat history: \n{chat_history_str}\n"
                "Rephrase the following question only if necessary: '{question}'"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", "{question}, {chat_history_str}"),
                ]
            )
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
            step_back = prompt | llm | StrOutputParser()
            new_question = step_back.invoke(
                {"question": question, "chat_history_str": chat_history_str}
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
        output["original_question"] = question
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
                    f"Context {idx+1}: \n(Document content: {doc.page_content}\nMetadata: (source_file: {doc.metadata['source'] if 'source' in doc.metadata else 'unknown'}))"
                    for idx, doc in enumerate(docs)
                ]
            )
            final_prompt = (
                "You are an AI Tutor for the course DS598, taught by Prof. Thomas Gardos. Answer the user's question using the provided context. Only use the context if it is relevant. The context is ordered by relevance."
                "If you don't know the answer, do your best without making things up. Keep the conversation flowing naturally. "
                "Use chat history and context as guides but avoid repeating past responses. Provide links from the source_file metadata. Use the source context that is most relevent."
                "Speak in a friendly and engaging manner, like talking to a friend. Avoid sounding repetitive or robotic.\n\n"
                f"Chat History:\n{chat_history_str}\n\n"
                f"Context:\n{context}\n\n"
                "Answer the student's question below in a friendly, concise, and engaging manner. Use the context and history only if relevant, otherwise, engage in a free-flowing conversation.\n"
                f"Student: {question}\n"
                "AI Tutor:"
            )

            # new_inputs["input"] = final_prompt
            new_inputs["question"] = final_prompt
            # output["final_prompt"] = final_prompt

            answer = await self.combine_docs_chain.arun(
                input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
            )
            output[self.output_key] = answer

        if self.return_source_documents:
            output["source_documents"] = docs
        output["rephrased_question"] = new_question
        return output


class LLMTutor:
    def __init__(self, config, logger=None):
        self.config = config
        self.llm = self.load_llm()
        self.logger = logger
        self.vector_db = VectorStoreManager(config, logger=self.logger)
        if self.config["vectorstore"]["embedd_files"]:
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

        retriever = Retriever(self.config)._return_retriever(db)

        if self.config["llm_params"]["use_history"]:
            memory = ConversationBufferWindowMemory(
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
                response_if_no_docs_found="No context found",
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
        qa = self.retrieval_qa_chain(
            self.llm, qa_prompt, db
        )  # TODO:  PROMPT is overwritten in CustomConversationalRetrievalChain

        return qa

    # output function
    def final_result(query):
        qa_result = qa_bot()
        response = qa_result({"query": query})
        return response
