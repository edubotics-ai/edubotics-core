from edubotics_core.chat.langchain.utils import (
    BaseChatMessageHistory,
    InMemoryHistory,
)
from edubotics_core.chat.base import BaseRAG
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated, Optional, TypedDict, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate, BasePromptTemplate
from langchain_core.callbacks import Callbacks
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.simple import Tool
from langchain.schema import Document
from functools import partial
from langchain_core.messages import SystemMessage


class RetrieverInput(BaseModel):
    """Input schema for the retriever tool."""

    input: str = Field(description="Query to look up in retriever")


def _get_relevant_documents(
    input: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks = None,
) -> list:
    """Retrieve relevant documents synchronously."""
    docs = retriever.get_relevant_documents(input)
    docs_list = []
    for doc in docs:
        docs_list.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "score": doc.metadata.get("score"),
            }
        )
    return docs_list


async def _aget_relevant_documents(
    input: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks = None,
) -> list:
    """Retrieve relevant documents asynchronously."""
    docs = await retriever.aget_relevant_documents(input)
    docs_list = []
    for doc in docs:
        docs_list.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "score": doc.metadata.get("score"),
            }
        )
    return docs_list


def create_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
) -> Tool:
    """Create a tool for document retrieval."""
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
    """State representation for the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: dict


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class LanggraphAgenticRAG(BaseRAG):
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
        Initialize the LanggraphAgenticRAG class.

        Args:
            llm: The language model instance.
            memory: The chat message history instance.
            retriever: The retriever instance.
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

        self.summary_prompt = """
            Summarize the main points from the retrieved documents in a concise manner, highlighting key information that directly answers the user’s query. Avoid additional questions or exploration.
            """

        self.action_prompt = """
            Extract actionable insights from the retrieved documents. Summarize any relevant information and suggest specific next steps or follow-up actions the user can take based on the provided content.
            """

        self.config = config
        self.store = {}

        # Initialize the agentic graph workflow
        self.graph = self.initialize_graph()

    def rewrite(self, state):
        print("---TRANSFORM QUERY---")
        question = state["messages"][0].content
        messages = [
            SystemMessage(
                content="Your task is to rephrase the user's question to make it more precise and clear, focusing on specificity to improve retrieval accuracy. Do not ask any questions back to the user. Only provide the rephrased question."
            ),
            HumanMessage(content=question),
        ]

        # LLM
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", streaming=True)
        response = model.invoke(messages)
        return {"messages": [response]}  # Return the rephrased question

    def generate(self, state):
        print("---GENERATE---")
        question = state["messages"][0].content
        docs = state["context"]

        messages = [
            SystemMessage(
                content="Provide a concise answer to the user's question using the provided context. Only include the final answer without any additional explanations or reasoning."
            ),
            HumanMessage(content=f"Context:\n{docs}\n\nQuestion:\n{question}"),
        ]

        # LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

        # Run
        response = llm.invoke(messages)
        return {"messages": [response], "context": docs}

    def extract_actionable_insights(self, state):
        print("---EXTRACT ACTIONABLE INSIGHTS---")
        # retrieved_docs = state["context"]
        question = state["messages"][0].content

        messages = [
            SystemMessage(
                content="Based on the provided context, provide actionable insights that address the user's question. Summarize relevant information and suggest specific next steps or follow-up actions. Do not include your reasoning or the context in your final response."
            ),
            HumanMessage(content=f"Question: {question}"),
        ]

        # LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

        # Run
        response = llm.invoke(messages)
        return {"messages": [response], "context": state["context"]}

    def grade_documents(self, state) -> Literal["extract_insights", "rewrite"]:
        print("---CHECK RELEVANCE---")
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", streaming=True)
        llm_with_tool = model.with_structured_output(GradeDocuments)

        prompt = PromptTemplate(
            template="""Evaluate whether the retrieved documents contain information directly relevant to the user's question. Respond with 'yes' if relevant, or 'no' if not, without any additional explanation.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool
        question = state["messages"][0].content
        docs = state["context"]
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score.lower()
        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "extract_insights"
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            return "rewrite"

    def agent(self, state):
        print("---CALL AGENT---")
        messages = state["messages"]
        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
        model = model.bind_tools(self.tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response], "context": state["context"]}

    def initialize_graph(self):
        """
        Initialize the agentic graph workflow.

        Returns:
            StateGraph: The initialized graph for the RAG process.
        """
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("agent", self.agent)
        retrieve = ToolNode([self.retriever_tool])
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rewrite", self.rewrite)
        workflow.add_node("generate", self.generate)
        workflow.add_node("extract_insights", self.extract_actionable_insights)

        # Start with the agent
        workflow.add_edge(START, "agent")

        # Agent decides whether to retrieve or not
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        # After retrieval, go to generate
        workflow.add_edge("retrieve", "generate")

        # After generating, grade the documents
        workflow.add_conditional_edges(
            "generate",
            self.grade_documents,
            {
                "extract_insights": "extract_insights",
                "rewrite": "rewrite",
            },
        )

        # After extracting insights, end the process
        workflow.add_edge("extract_insights", END)

        # If rewriting is necessary, loop back to the agent
        workflow.add_edge("rewrite", "agent")

        # Compile the workflow
        graph = workflow.compile()
        return graph

    def add_history_from_list(self, conversation_list):
        """
        Add messages from a list to the chat history.

        Args:
            conversation_list (list): The list of messages to add.
        """
        history = ChatMessageHistory()
        for message in conversation_list:
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
            self.store[(user_id, conversation_id)].add_messages(self.memory.messages)
        return self.store[(user_id, conversation_id)]

    async def invoke(self, user_query, config, **kwargs):
        """
        Invoke the agentic RAG process with the given user query.

        Args:
            user_query (dict): The user query containing 'input'.
            config (dict): Configuration for the process.

        Returns:
            dict: The response containing 'answer' and 'context'.
        """
        inputs = {"messages": [HumanMessage(content=user_query["input"])]}
        output = {}

        # Stream outputs from the graph
        for node_output in self.graph.stream(inputs, {"recursion_limit": 10}):
            output.update(node_output)

        print("---END---")

        # Extract the final answer and context
        key = next(
            (
                k
                for k in ["extract_insights", "rewrite", "generate", "agent"]
                if k in output
            ),
            None,
        )

        if key:
            context_data = output[key].get("context", [])
            # Convert context_data to a list of Document objects
            res_context = (
                [
                    Document(page_content=doc["page_content"], metadata=doc["metadata"])
                    for doc in context_data
                ]
                if isinstance(context_data, list)
                else []
            )
            answer_message = output[key]["messages"][0]
            if isinstance(answer_message, AIMessage):
                answer = answer_message.content
            else:
                answer = answer_message
            res = {"answer": answer, "context": res_context}
        else:
            res = {"answer": None, "context": []}

        return res
