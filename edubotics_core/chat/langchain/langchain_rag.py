from langchain_core.prompts import ChatPromptTemplate

# from edubotics_core.chat.langchain.utils import
from langchain_community.chat_message_histories import ChatMessageHistory
from edubotics_core.chat.base import BaseRAG
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from .utils import (
    CustomConversationalRetrievalChain,
    create_history_aware_retriever,
    create_stuff_documents_chain,
    create_retrieval_chain,
    return_questions,
    CustomRunnableWithHistory,
    BaseChatMessageHistory,
    InMemoryHistory,
)


class Langchain_RAG_V1(BaseRAG):
    def __init__(
        self,
        llm,
        memory,
        retriever,
        qa_prompt: str,
        rephrase_prompt: str,
        config: dict,
        callbacks=None,
    ):
        """
        Initialize the Langchain_RAG class.

        Args:
            llm (LanguageModelLike): The language model instance.
            memory (BaseChatMessageHistory): The chat message history instance.
            retriever (BaseRetriever): The retriever instance.
            qa_prompt (str): The QA prompt string.
            rephrase_prompt (str): The rephrase prompt string.
        """
        self.llm = llm
        self.config = config
        # self.memory = self.add_history_from_list(memory)
        self.memory = ConversationBufferWindowMemory(
            k=self.config["llm_params"]["memory_window"],
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            max_token_limit=128,
        )
        self.retriever = retriever
        self.qa_prompt = qa_prompt
        self.rephrase_prompt = rephrase_prompt
        self.store = {}

        self.qa_prompt = PromptTemplate(
            template=self.qa_prompt,
            input_variables=["context", "chat_history", "input"],
        )

        self.rag_chain = CustomConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            response_if_no_docs_found="No context found",
        )

    def add_history_from_list(self, history_list):
        """
        TODO: Add messages from a list to the chat history.
        """
        history = []

        return history

    async def invoke(self, user_query, config):
        """
        Invoke the chain.

        Args:
            kwargs: The input variables.

        Returns:
            dict: The output variables.
        """
        res = await self.rag_chain.acall(user_query["input"])
        return res


class QuestionGenerator:
    """
    Generate a question from the LLMs response and users input and past conversations.
    """

    def __init__(self):
        pass

    def generate_questions(self, query, response, chat_history, context, config):
        questions = return_questions(query, response, chat_history, context, config)
        return questions


class Langchain_RAG_V2(BaseRAG):
    def __init__(
        self,
        llm,
        memory,
        retriever,
        qa_prompt: str,
        rephrase_prompt: str,
        config: dict,
        callbacks=None,
    ):
        """
        Initialize the Langchain_RAG class.

        Args:
            llm (LanguageModelLike): The language model instance.
            memory (BaseChatMessageHistory): The chat message history instance.
            retriever (BaseRetriever): The retriever instance.
            qa_prompt (str): The QA prompt string.
            rephrase_prompt (str): The rephrase prompt string.
        """
        self.llm = llm
        self.memory = self.add_history_from_list(memory)
        self.retriever = retriever
        self.qa_prompt = qa_prompt
        self.rephrase_prompt = rephrase_prompt
        self.store = {}

        # Contextualize question prompt
        contextualize_q_system_prompt = rephrase_prompt or (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_template(
            contextualize_q_system_prompt
        )

        # History-aware retriever
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )

        # Answer question prompt
        qa_system_prompt = qa_prompt or (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise."
            "\n\n"
            "{context}"
        )
        self.qa_prompt_template = ChatPromptTemplate.from_template(qa_system_prompt)

        # Question-answer chain
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm, self.qa_prompt_template
        )

        # Final retrieval chain
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )

        self.rag_chain = CustomRunnableWithHistory(
            self.rag_chain,
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="memory_window",
                    annotation=int,
                    name="Number of Conversations",
                    description="Number of conversations to consider for context.",
                    default=1,
                    is_shared=True,
                ),
            ],
        ).with_config(run_name="Langchain_RAG_V2")

        if callbacks is not None:
            self.rag_chain = self.rag_chain.with_config(callbacks=callbacks)

    def get_session_history(
        self, user_id: str, conversation_id: str, memory_window: int
    ) -> BaseChatMessageHistory:
        """
        Get the session history for a user and conversation.

        Args:
            user_id (str): The user identifier.
            conversation_id (str): The conversation identifier.
            memory_window (int): The number of conversations to consider for context.

        Returns:
            BaseChatMessageHistory: The chat message history.
        """
        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = InMemoryHistory()
            self.store[(user_id, conversation_id)].add_messages(
                self.memory.messages
            )  # add previous messages to the store. Note: the store is in-memory.
        return self.store[(user_id, conversation_id)]

    async def invoke(self, user_query, config, **kwargs):
        """
        Invoke the chain.

        Args:
            kwargs: The input variables.

        Returns:
            dict: The output variables.
        """
        res = await self.rag_chain.ainvoke(user_query, config, **kwargs)
        res["rephrase_prompt"] = self.rephrase_prompt
        res["qa_prompt"] = self.qa_prompt
        return res

    def stream(self, user_query, config):
        res = self.rag_chain.stream(user_query, config)
        return res

    def add_history_from_list(self, conversation_list):
        """
        Add messages from a list to the chat history.

        Args:
            messages (list): The list of messages to add.
        """
        history = ChatMessageHistory()

        for idx, message in enumerate(conversation_list):
            message_type = (
                message.get("type", None)
                if isinstance(message, dict)
                else getattr(message, "type", None)
            )

            message_content = (
                message.get("content", None)
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )

            if message_type in ["human", "user_message"]:
                history.add_user_message(message_content)
            elif message_type in ["ai", "ai_message"]:
                history.add_ai_message(message_content)

        return history
