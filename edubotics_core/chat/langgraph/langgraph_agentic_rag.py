from langchain_core.prompts import ChatPromptTemplate
from edubotics_core.chat.langchain.utils import (
    BaseChatMessageHistory,
    InMemoryHistory,
)
from edubotics_core.chat.base import BaseRAG
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pprint import pprint
# from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.tools import Tool
from typing import Annotated, Literal, Sequence, TypedDict
import ast
from langchain.schema import Document



import re

def parse_document_string(doc_string):
    # Use regex to find all instances of metadata and page_content
    pattern = r"Document\(metadata=(\{.*?\}), page_content='(.*?)'\)"
    matches = re.findall(pattern, doc_string, re.DOTALL)

    documents = []
    for metadata_str, page_content in matches:
        # Convert the metadata string to a dictionary using eval (safe here since it’s controlled)
        metadata = eval(metadata_str)
        
        # Create a Document object
        doc = Document(metadata=metadata, page_content=page_content)
        documents.append(doc)

    return documents

# Implemented from: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#nodes-and-edges
# and:https://github.com/aelaguiz/update_writer/blob/6910cfa6b7825548f4159d36250b25dec1055e66/src/lib/lib_tools.py#L86

from langchain_core.pydantic_v1 import BaseModel, Field
class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from functools import partial
from typing import Optional

from langchain_core.callbacks import Callbacks
from langchain_core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
    # aformat_document,
    # format_document,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.simple import Tool


class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")


def format_document(doc, document_prompt):
    return doc

async def aformat_document(doc, document_prompt):
    return doc

def _get_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks = None,
) -> str:
    docs = retriever.invoke(query, config={"callbacks": callbacks})
    # return document_separator.join(
    #     format_document(doc, document_prompt) for doc in docs
    # )
    docs_list = []
    for doc in docs:
        docs_list.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "score": doc.metadata["score"],
        })
    return docs_list


async def _aget_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks = None,
) -> str:
    docs = await retriever.ainvoke(query, config={"callbacks": callbacks})
    # return document_separator.join(
    #     [await aformat_document(doc, document_prompt) for doc in docs]
    # )
    docs_list = []
    for doc in docs:
        docs_list.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "score": doc.metadata["score"],
        })
    return docs_list
    


def create_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.
        document_prompt: The prompt to use for the document. Defaults to None.
        document_separator: The separator to use between documents. Defaults to "\n\n".

    Returns:
        Tool class to pass to an agent.
    """
    document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
    func = partial(
        _get_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
    )
    afunc = partial(
        _aget_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
    )
    return Tool(
        name=name,
        description=description,
        func=func,
        coroutine=afunc,
        args_schema=RetrieverInput,
    )


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Annotated[Sequence[BaseMessage], add_messages]


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class Langgraph_Agentic_RAG(BaseRAG):
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
        Initialize the Langgrah_Agentic_RAG class.

        Args:
            llm (LanguageModelLike): The language model instance.
            memory (BaseChatMessageHistory): The chat message history instance.
            retriever (BaseRetriever): The retriever instance.
            qa_prompt (str): The QA prompt string.
            rephrase_prompt (str): The rephrase prompt string.
            config (dict): Configuration dictionary.
            callbacks (Optional[list]): Optional list of callbacks.
        """
        self.llm = llm
        self.memory = self.add_history_from_list(memory)
        self.retriever = retriever
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_contextual_answers",
            "Search and return information about the course content.",
        )
        self.tools = [self.retriever_tool]

        self.qa_prompt = qa_prompt
        self.rephrase_prompt = rephrase_prompt
        self.config = config
        self.store = {}

        # Initialize the agentic graph workflow
        self.graph = self.initialize_graph()


    def grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)

        # LLM with tool and validation
        llm_with_tool = model.with_structured_output(grade)

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content
        

        print(question)
        # print(docs)
        # print(type(docs))
        docs = ast.literal_eval(docs)
        print(docs)
        # print(type(docs))
        # exit()
        # docs = parse_document_string(docs)  # TODO: This is a hack to convert the string representation of a list of documents to an actual list of documents. Fix later.
        # doc_contents = [doc.page_content for doc in docs]
        doc_contents_str = str(docs)
        # print(doc_contents_str)

        scored_result = chain.invoke({"question": question, "context": doc_contents_str})

        score = scored_result.binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(score)
            return "rewrite"


    ### Nodes


    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("---CALL AGENT---")
        messages = state["messages"]
        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
        model = model.bind_tools(self.tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response], "context": state["context"]}


    def rewrite(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        # Grader
        model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
        response = model.invoke(msg)
        return {"messages": [response]}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]

        print("\n\n\n\n\n")
        print("I am here")
        print(messages)
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content
        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

        # # Post-processing
        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response], "context": docs}


    def initialize_graph(self):
        """
        Initialize the agentic graph workflow.

        Returns:
            StateGraph: The initialized graph for the RAG process.
        """

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", self.agent)  # agent
        retrieve = ToolNode([self.retriever_tool])
        workflow.add_node("retrieve", retrieve)  # retrieval
        workflow.add_node("rewrite", self.rewrite)  # Re-writing the question
        workflow.add_node(
            "generate", self.generate
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not
        workflow.add_edge(START, "agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            self.grade_documents,
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        # Compile
        graph = workflow.compile()
        return graph


    def add_history_from_list(self, conversation_list):
        """
        Add messages from a list to the chat history.

        Args:
            conversation_list (list): The list of messages to add.
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
        inputs = {
            "messages": [
                ("user", "What is the course about??"),
            ]
        }
        for output in self.graph.stream(inputs, {"recursion_limit": 10}):
            for key, value in output.items():
                pprint(f"Output from node '{key}':")
                pprint("---")
                pprint(value, indent=2, width=80, depth=None)
            pprint("\n---\n")

        print("---END---")
        res = {}
        print(output)
        exit()
        # res["answer"] = output
        # res["context"] = self.docs
        return res


