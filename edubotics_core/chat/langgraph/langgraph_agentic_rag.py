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
from langchain_core.messages import BaseMessage, HumanMessage
import pprint
from langchain.tools.retriever import create_retriever_tool
from langchain import hub

# Implemented from: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#nodes-and-edges


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


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
        self.retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_contextual_answers",
            "Search and return information about the course content.",
        )

        self.qa_prompt = qa_prompt
        self.rephrase_prompt = rephrase_prompt
        self.config = config
        self.store = {}

        # Initialize the agentic graph workflow
        self.graph = self.initialize_graph()

    def initialize_graph(self):
        """
        Initialize the agentic RAG graph workflow.
        """
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("agent", self.agent)
        retrieve = ToolNode([self.retriever_tool])
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rewrite", self.rewrite)
        workflow.add_node("generate", self.generate)

        # Define the edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        return workflow.compile()

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
        Invoke the agentic RAG workflow.

        Args:
            kwargs: The input variables.

        Returns:
            dict: The output variables.
        """
        initial_state = {"messages": [user_query]}
        inputs = {
            "messages": [
                ("user", f"{initial_state}"),
            ]
        }
        output = self.graph.invoke(inputs)
        output["context"] = self.context
        output["answer"] = output["messages"][-1].content
        return output

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

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state.
        """
        print("---CALL AGENT---")
        messages = state["messages"]
        model = ChatOpenAI(
            temperature=0, streaming=True, model=self.config["llm_params"]["llm_loader"]
        )
        model = model.bind_tools([self.retriever_tool])
        response = model.invoke(messages)
        return {"messages": [response]}

    def rewrite(self, state):
        """
        Transform the query to produce a better question.
        """
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f"{self.rephrase_prompt}\n{question}\nFormulate an improved question:"
            )
        ]

        model = ChatOpenAI(
            temperature=0, model=self.config["llm_params"]["llm_loader"], streaming=True
        )
        response = model.invoke(msg)
        return {"messages": [response]}

    def generate(self, state):
        """
        Generate an answer based on retrieved documents and the original question.
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # LLM
        llm = ChatOpenAI(
            model_name=self.config["llm_params"]["llm_loader"],
            temperature=0,
            streaming=True,
        )

        prompt = PromptTemplate(
            template=f"{self.qa_prompt}\nAnswer the question based on the provided context.",
            input_variables=["context", "question", "chat_history", "input"],
        )

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        response = rag_chain.invoke(
            {
                "context": docs,
                "question": question,
                "chat_history": self.memory.messages,
                "input": question,
            }
        )
        return {"messages": [response]}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.
        """
        print("---CHECK RELEVANCE---")
        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content
        self.context = docs

        class grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        model = ChatOpenAI(
            temperature=0, model=self.config["llm_params"]["llm_loader"], streaming=True
        )
        llm_with_tool = model.with_structured_output(grade)

        prompt = PromptTemplate(
            template=f"{self.qa_prompt}\nDetermine relevance of the retrieved document given the question and context.",
            input_variables=["context", "question", "chat_history", "input"],
        )

        print("question: ", question)
        print("context: ", docs)
        print("chat_history: ", self.memory.messages)
        print("input: ", question)

        chain = prompt | llm_with_tool
        scored_result = chain.invoke(
            {
                "question": question,
                "context": docs,
                "chat_history": self.memory.messages,
                "input": question,
            }
        )

        score = scored_result.binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            return "rewrite"
