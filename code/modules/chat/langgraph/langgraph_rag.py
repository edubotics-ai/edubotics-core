# Adapted from https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb?ref=blog.langchain.dev

from typing import List

from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from modules.chat.base import BaseRAG
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


class Langgraph_RAG(BaseRAG):
    def __init__(self, llm, memory, retriever, qa_prompt: str, rephrase_prompt: str):
        """
        Initialize the Langgraph_RAG class.

        Args:
            llm (LanguageModelLike): The language model instance.
            memory (BaseChatMessageHistory): The chat message history instance.
            retriever (BaseRetriever): The retriever instance.
            qa_prompt (str): The QA prompt string.
            rephrase_prompt (str): The rephrase prompt string.
        """
        self.llm = llm
        self.structured_llm_grader = llm.with_structured_output(GradeDocuments)
        self.memory = self.add_history_from_list(memory)
        self.retriever = retriever
        self.qa_prompt = (
            "You are an AI Tutor for the course DS598, taught by Prof. Thomas Gardos. Answer the user's question using the provided context. Only use the context if it is relevant. The context is ordered by relevance. "
            "If you don't know the answer, do your best without making things up. Keep the conversation flowing naturally. "
            "Speak in a friendly and engaging manner, like talking to a friend. Avoid sounding repetitive or robotic.\n\n"
            "Context:\n{context}\n\n"
            "Answer the student's question below in a friendly, concise, and engaging manner. Use the context and history only if relevant, otherwise, engage in a free-flowing conversation.\n"
            "Student: {question}\n"
            "AI Tutor:"
        )
        self.rephrase_prompt = rephrase_prompt
        self.store = {}

        ## Fix below ##

        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )

        self.retrieval_grader = grade_prompt | self.structured_llm_grader

        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        # Generate
        self.qa_prompt_template = ChatPromptTemplate.from_template(self.qa_prompt)
        self.rag_chain = self.qa_prompt_template | self.llm | StrOutputParser()

        ###

        # build the agentic graph
        self.app = self.create_agentic_graph()

    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def create_agentic_graph(self):
        """
        Create an agentic graph to answer questions.

        Returns:
            dict: Agentic graph
        """
        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node(
            "grade_documents", self.grade_documents
        )  # grade documents
        self.workflow.add_node("generate", self.generate)  # generatae
        self.workflow.add_node(
            "transform_query", self.transform_query
        )  # transform_query

        # build the graph
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        self.workflow.add_edge("transform_query", "generate")
        self.workflow.add_edge("generate", END)

        # Compile
        app = self.workflow.compile()
        return app

    def invoke(self, user_query, config):
        """
        Invoke the chain.

        Args:
            kwargs: The input variables.

        Returns:
            dict: The output variables.
        """

        inputs = {
            "question": user_query["input"],
        }

        for output in self.app.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"Node {key} returned: {value}")
            print("\n\n")

        print(value["generation"])

        # rename generation to answer
        value["answer"] = value.pop("generation")
        value["context"] = value.pop("documents")

        return value

    def add_history_from_list(self, history_list):
        """
        Add messages from a list to the chat history.

        Args:
            messages (list): The list of messages to add.
        """
        history = ChatMessageHistory()

        for idx, message_pairs in enumerate(history_list):
            history.add_user_message(message_pairs[0])
            history.add_ai_message(message_pairs[1])

        return history
