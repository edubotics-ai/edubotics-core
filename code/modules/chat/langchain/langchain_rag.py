from langchain_core.prompts import ChatPromptTemplate

from modules.chat.langchain.utils import *


class CustomConversationalRetrievalChain:
    def __init__(self, llm, memory, retriever, qa_prompt: str, rephrase_prompt: str):
        """
        Initialize the CustomConversationalRetrievalChain class.

        Args:
            llm (LanguageModelLike): The language model instance.
            memory (BaseChatMessageHistory): The chat message history instance.
            retriever (BaseRetriever): The retriever instance.
            qa_prompt (str): The QA prompt string.
            rephrase_prompt (str): The rephrase prompt string.
        """
        self.llm = llm
        self.memory = memory
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
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
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
        self.qa_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

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
        )

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
        return self.store[(user_id, conversation_id)]

    def invoke(self, user_query, config):
        """
        Invoke the chain.

        Args:
            kwargs: The input variables.

        Returns:
            dict: The output variables.
        """
        print(user_query, config)
        return self.rag_chain.invoke(user_query, config)
