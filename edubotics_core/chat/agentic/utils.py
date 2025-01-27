import sys
import os
from typing import Literal, TypedDict, List
from PIL import Image
from typing import Annotated
from langgraph.graph.message import add_messages

from pprint import pprint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pydantic import BaseModel, Field

# Load the variables from .env
load_dotenv()

content_types = ['assignment', "lecture", "discussion", "other"]
NUM_VECTORSTORES = len(content_types)
VS_PATH = "vectorstores"

# Data model


class RouteQuery(BaseModel):
    """Route a user query to the most relevant vector store."""

    datasource: Literal['assignment', "lecture",
                        "discussion", "other", "not_needed"] = Field(
        ...,
        description="Given a user question choose to route it to the relevant vector store or none.",
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: list of messages
        generation: LLM generation
        documents: list of documents
    """

    messages: Annotated[list, add_messages]
    documents: List[str]
    documents_sources: List[str]
    type: Literal['retrieve', 'generate']
    next: str


system_prompt = """
You are an AI Assistant for a university course.
"""

rag_template = """
Answer the question with the help of the following context and conversation history:
Context: {context}
Conversation history: {conversation_history}

You may not need the provided context to answer the question. If that is the case, just answer the question based on your knowledge.
You could also not be provided with any context at all, in that case, just answer the question based on your knowledge and/or conversation history.

Input: {input}
"""

router_template = """You are an expert at routing a user question to different vector stores.
There are 4 vector stores:
- assignment: chunks from assignment notebooks containing code exercises and maybe free-form responses. Also contains the midterm challenge.
- lecture: lecture content on machine learning, classification, regression and clustering.
- discussion: discussion content that mirrors content from lecture on a smaller scale, containing shorter exercises meant for classroom discussion
- other: anything else about the class - office hours, syllabus, project and professor info, or answer the question using the conversation history.
Return the corresponding vector store depending of the topics of the question or just not_needed because it does't match with the vector stores.
 
Input: {input}
"""
