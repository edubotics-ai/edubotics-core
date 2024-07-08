from typing import Any, Dict, List, Union, Tuple, Optional
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    FunctionMessage,
    HumanMessage,
)

from langchain_core.prompts.base import BasePromptTemplate, format_document
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.retrievers import BaseRetriever, RetrieverOutput
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    BaseCombineDocumentsChain,
    _validate_prompt,
)
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document


CHAT_TURN_TYPE = Union[Tuple[str, str], BaseMessage]

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage


class CustomRunnableWithHistory(RunnableWithMessageHistory):
    def _enter_history(self, input: Any, config: RunnableConfig) -> List[BaseMessage]:
        """
        Get the last k conversations from the message history.

        Args:
            input (Any): The input data.
            config (RunnableConfig): The runnable configuration.

        Returns:
            List[BaseMessage]: The last k conversations.
        """
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        messages = hist.messages.copy()

        if not self.history_messages_key:
            # return all messages
            messages += self._get_input_messages(input)

        # return last k conversations
        if config["configurable"]["memory_window"] == 0:  # if k is 0, return empty list
            messages = []
        else:
            messages = messages[-2 * config["configurable"]["memory_window"] :]
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

    def get_last_n_conversations(self, n: int) -> "InMemoryHistory":
        """Return a new InMemoryHistory object with the last n conversations from the message history.

        Args:
            n (int): The number of last conversations to return. If 0, return an empty history.

        Returns:
            InMemoryHistory: A new InMemoryHistory object containing the last n conversations.
        """
        if n == 0:
            return InMemoryHistory()
        # Each conversation consists of a pair of messages (human + AI)
        num_messages = n * 2
        last_messages = self.messages[-num_messages:]
        return InMemoryHistory(messages=last_messages)


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
    _validate_prompt(prompt)
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
