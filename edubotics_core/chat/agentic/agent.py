import asyncio
import sys
import os
from typing import Literal, TypedDict, List
from PIL import Image

from pprint import pprint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate
from chainlit.types import ThreadDict
from langgraph.checkpoint.memory import MemorySaver

from .utils import RouteQuery, GraphState, rag_template, router_template
from edubotics_core.vectorstore.mvs import MultiVectorStore


load_dotenv()


class Agent:
    def __init__(
        self, thread_id: str = None, config: dict = None, prompts: dict = None
    ):
        self.config = config
        self.retrievers = {}
        self.content_types = self.config["metadata"]["content_types"]

        mvs = MultiVectorStore(self.config)
        self.retrievers = mvs.as_retriever()

        self.workflow = self.create_workflow()
        self.graph = self.workflow.compile(checkpointer=MemorySaver())

        self.updates = []
        self.thread_id = thread_id

        self.conversation_history = []
        self.memory_window = self.config["llm_params"]["memory_window"]

        self.prompts = prompts

        api_key = os.environ["OPENAI_API_KEY"]

        self.rag_prompt = ChatPromptTemplate.from_template(
            prompts["openai"]["prompt_with_history"]["normal"]
        )
        self.route_prompt = ChatPromptTemplate.from_template(router_template)
        self.model = ChatOpenAI(
            temperature=0.5,
            model_name="gpt-4o-mini",
            openai_api_key=api_key,
            # stream_options={"include_usage": True},
            # stream=True,
        )

        self.rag_chain = self.rag_prompt | self.model | StrOutputParser()

        structured_model_router = self.model.with_structured_output(RouteQuery)
        self.question_router = self.route_prompt | structured_model_router

    def set_thread_id(self, thread_id: str):
        self.thread_id = thread_id

    def create_workflow(self):

        workflow = StateGraph(GraphState)

        workflow.add_node("assignments_retrieve", self.assignments_retrieve)
        workflow.add_node("lectures_retrieve", self.lectures_retrieve)
        workflow.add_node("discussions_retrieve", self.discussions_retrieve)
        workflow.add_node("other_retrieve", self.other_retrieve)
        workflow.add_node("not_needed", self.no_retrieve)
        workflow.add_node("generate", self.generate)

        # Build graph
        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "assignments_retrieve": "assignments_retrieve",
                "lectures_retrieve": "lectures_retrieve",
                "discussions_retrieve": "discussions_retrieve",
                "other_retrieve": "other_retrieve",
                "not_needed": "not_needed",
            },
        )

        workflow.add_edge("assignments_retrieve", "generate")
        workflow.add_edge("lectures_retrieve", "generate")
        workflow.add_edge("discussions_retrieve", "generate")
        workflow.add_edge("other_retrieve", "generate")
        workflow.add_edge("not_needed", "generate")
        workflow.add_edge("generate", END)

        return workflow

    async def stream(self, question: str):
        config = {"configurable": {"thread_id": self.thread_id}}
        messages_with_history = self.conversation_history + [
            HumanMessage(content=question)
        ]
        token_count = 0
        for event in self.graph.stream(
            {"messages": messages_with_history},
            config,
            stream_mode=["messages", "updates"],
        ):
            if event[0] == "messages" and event[1][1]["langgraph_node"] == "generate":
                ai_message = event[1][0]

                if isinstance(ai_message, AIMessageChunk):
                    if ai_message.usage_metadata:
                        tokens = ai_message.usage_metadata.get("output_tokens", 0)
                        token_count += tokens

                    yield {"content": ai_message.content, "total_tokens": token_count}
            else:
                update = event[1]
                self.updates.append(update)

    def run(self, question: str) -> dict:
        config = {"configurable": {"thread_id": self.thread_id}}
        last_state = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]}, config
        )
        response = last_state["messages"][-1].content
        documents = last_state["documents"]
        self.updates.append(last_state)
        return {"response": response, "documents": documents}

    def get_sources(self):
        """
        Get the sources of the documents retrieved from the vector stores for the last question.
        """
        if len(self.updates) == 0:
            return []
        else:
            last_update = self.updates[-1]
            if "generate" in last_update:
                return last_update["generate"]["documents_sources"]
            elif "documents" in last_update:
                return last_update["documents"]
            else:
                return []

    def assignments_retrieve(self, state):
        """
        Retrieve documents from the assignments vector store

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE ASSIGNMENTS---")
        messages = state["messages"]
        question = messages[-1].content

        # Retrieval
        documents = self.retrievers["assignments"].invoke(question)
        return {"documents": documents, "type": "retrieve"}

    def lectures_retrieve(self, state):
        """
        Retrieve documents from the lectures vector store

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE LECTURES---")
        messages = state["messages"]
        question = messages[-1].content

        # Retrieval
        documents = self.retrievers["lecture"].invoke(question)
        return {"documents": documents, "type": "retrieve"}

    def discussions_retrieve(self, state):
        """
        Retrieve documents from the discussions vector store
        """
        print("---RETRIEVE DISCUSSIONS---")
        messages = state["messages"]
        question = messages[-1].content

        documents = self.retrievers["discussion"].invoke(question)
        return {"documents": documents, "type": "retrieve"}

    def other_retrieve(self, state):
        """
        Retrieve documents from the other vector store
        """
        print("---RETRIEVE OTHER---")
        messages = state["messages"]
        question = messages[-1].content

        documents = self.retrievers["other"].invoke(question)
        return {"documents": documents, "type": "retrieve"}

    def no_retrieve(self, state):
        """
        Return empty documents
        """
        print("---RETRIEVE NOT NEEDED---")
        return {"documents": [], "type": "retrieve"}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        # print("---GENERATE---")
        messages = state["messages"]
        question = messages[-1].content

        conversation_history = messages[-6:-1]

        documents = state["documents"]
        documents_sources = [doc.metadata["source"] for doc in documents]

        # RAG generation
        response = self.rag_chain.invoke(
            {
                "context": documents,
                "input": question,
                "chat_history": conversation_history,
            }
        )
        ai_message = AIMessage(content=response)

        return {
            "documents_sources": documents_sources,
            "messages": [ai_message],
            "type": "generate",
        }

    def route_question(self, state):
        """
        Route question to corresponding RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        # print("---ROUTE QUESTION---")
        messages = state["messages"]
        question = messages[-1].content

        source = self.question_router.invoke({"input": question})
        if source.datasource == "assignment":
            # print("---ROUTE QUESTION TO ASSIGNMENTS---")
            return "assignments_retrieve"
        elif source.datasource == "lecture":
            # print("---ROUTE QUESTION TO LECTURES---")
            return "lectures_retrieve"
        elif source.datasource == "discussion":
            # print("---ROUTE QUESTION TO DISCUSSIONS---")
            return "discussions_retrieve"
        elif source.datasource == "other":
            # print("---ROUTE QUESTION TO OTHER---")
            return "other_retrieve"
        else:
            # print("---ROUTE QUESTION TO RETRIEVAL NOT NEEDED---")
            return "not_needed"

    def reset_history(self):
        """
        Reset conversation history - before resuming a chat or starting a new one.
        """
        self.conversation_history = []

    def get_history(self) -> List[HumanMessage | AIMessage]:
        return self.conversation_history

    def get_state(self, config):
        return self.graph.get_state(config)

    def populate_conversation_history(self, thread: ThreadDict):
        """
        Populate conversation history from a thread
        """
        steps = thread["steps"]
        thread_id = thread["id"]

        self.set_thread_id(thread_id)

        self.conversation_history = []

        for step in steps:
            message_type = step["type"]
            if message_type == "user_message":
                content = step["output"]
                self.conversation_history.append(HumanMessage(content=content))
            elif message_type in [
                "assistant_message",
                "assistant_message_chunk",
                "ai_message",
                "ai_message_chunk",
            ]:
                content = step["output"]
                self.conversation_history.append(AIMessage(content=content))

    def update_config(self, config):
        self.config.update(config)


if __name__ == "__main__":
    import yaml

    from .prompts import prompts

    with open(
        "/Users/faridkarimli/Desktop/Programming/AI/edubot-core/edubotics_core/chat/agentic/config.yml",
        "r",
    ) as f:
        config = yaml.safe_load(f)

    agent = Agent(thread_id="123", config=config, prompts=prompts)

    async def stream_responses():
        async for response in agent.stream("What do we do in discussion 4?"):
            print(response["content"])

    asyncio.run(stream_responses())
