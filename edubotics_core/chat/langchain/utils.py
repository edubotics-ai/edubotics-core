from typing import Any, Dict, List, Union, Tuple, Optional
from langchain_core.prompts.base import BasePromptTemplate, format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.retrievers import BaseRetriever, RetrieverOutput
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    _validate_prompt,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun
import inspect
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

CHAT_TURN_TYPE = Union[Tuple[str, str], BaseMessage]


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
                "Rephrase the following question only if necessary: '{input}'"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", "{input}, {chat_history_str}"),
                ]
            )
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
            step_back = prompt | llm | StrOutputParser()
            new_question = step_back.invoke(
                {"input": question, "chat_history_str": chat_history_str}
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
                f"Student: {input}\n"
                "AI Tutor:"
            )

            new_inputs["input"] = final_prompt
            # new_inputs["question"] = final_prompt
            # output["final_prompt"] = final_prompt

            answer = await self.combine_docs_chain.arun(
                input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
            )
            output[self.output_key] = answer

        if self.return_source_documents:
            output["source_documents"] = docs
        output["rephrased_question"] = new_question
        output["context"] = output["source_documents"]
        return output


class CustomRunnableWithHistory(RunnableWithMessageHistory):
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

    async def _aenter_history(
        self, input: Any, config: RunnableConfig
    ) -> List[BaseMessage]:
        """
        Get the last k conversations from the message history.

        Args:
            input (Any): The input data.
            config (RunnableConfig): The runnable configuration.

        Returns:
            List[BaseMessage]: The last k conversations.
        """
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        messages = (await hist.aget_messages()).copy()
        if not self.history_messages_key:
            # return all messages
            input_val = (
                input if not self.input_messages_key else input[self.input_messages_key]
            )
            messages += self._get_input_messages(input_val)

        # return last k conversations
        if config["configurable"]["memory_window"] == 0:  # if k is 0, return empty list
            messages = []
        else:
            messages = messages[-2 * config["configurable"]["memory_window"] :]

        messages = self._get_chat_history(messages)

        return messages


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In-memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store."""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear the message history."""
        self.messages = []

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)


def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: BaseRetriever,
    prompt: BasePromptTemplate,
) -> Runnable[Dict[str, Any], RetrieverOutput]:
    """Create a chain that takes conversation history and returns documents."""
    if "input" not in prompt.input_variables:
        raise ValueError(
            "Expected `input` to be a prompt variable, "
            f"but got {prompt.input_variables}"
        )

    retrieve_documents = RunnableBranch(
        (
            lambda x: not x["chat_history"],
            (lambda x: x["input"]) | retriever,
        ),
        prompt | llm | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    return retrieve_documents


def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
) -> Runnable[Dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a model."""
    _validate_prompt(prompt, "context")
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    def format_docs(inputs: dict) -> str:
        return document_separator.join(
            format_document(doc, _document_prompt) for doc in inputs[DOCUMENTS_KEY]
        )

    return (
        RunnablePassthrough.assign(**{DOCUMENTS_KEY: format_docs}).with_config(
            run_name="format_inputs"
        )
        | prompt
        | llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")


def create_retrieval_chain(
    retriever: Union[BaseRetriever, Runnable[dict, RetrieverOutput]],
    combine_docs_chain: Runnable[Dict[str, Any], str],
) -> Runnable:
    """Create retrieval chain that retrieves documents and then passes them on."""
    if not isinstance(retriever, BaseRetriever):
        retrieval_docs = retriever
    else:
        retrieval_docs = (lambda x: x["input"]) | retriever

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents"),
        ).assign(answer=combine_docs_chain)
    ).with_config(run_name="retrieval_chain")

    return retrieval_chain


# TODO: Remove Hard-coded values
async def return_questions(query, response, chat_history_str, context, config):
    system = (
        "You are someone that suggests a question based on the student's input and chat history. "
        "Generate a question that is relevant to the student's input and chat history. "
        "Incorporate relevant details from the chat history to make the question clearer and more specific. "
        "Chat history: \n{chat_history_str}\n"
        "Use the context to generate a question that is relevant to the student's input and chat history: Context: {context}"
        "Generate 3 short and concise questions from the students voice based on the following input and response: "
        "The 3 short and concise questions should be sperated by dots. Example: 'What is the capital of France?...What is the population of France?...What is the currency of France?'"
        "User Query: {query}"
        "AI Response: {response}"
        "The 3 short and concise questions seperated by dots (...) are:"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            # ("human", "{chat_history_str}, {context}, {query}, {response}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    question_generator = prompt | llm | StrOutputParser()
    question_generator = question_generator.with_config(
        run_name="follow_up_question_generator"
    )
    new_questions = await question_generator.ainvoke(
        {
            "chat_history_str": chat_history_str,
            "context": context,
            "query": query,
            "response": response,
        },
        config=config,
    )

    list_of_questions = new_questions.split("...")
    return list_of_questions
