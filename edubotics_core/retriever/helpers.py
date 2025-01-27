from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from typing import List
from edubotics_core.config.constants import COHERE_API_KEY
import cohere


class VectorStoreRetrieverScore(VectorStoreRetriever):
    # See https://github.com/langchain-ai/langchain/blob/61dd92f8215daef3d9cf1734b0d1f8c70c1571c3/libs/langchain/langchain/vectorstores/base.py#L500
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )
        # Make the score part of the document metadata
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity

        docs = [doc for doc, _ in docs_and_similarities]
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )
        docs = [doc for doc, _ in docs_and_similarities]

        cohere_client = cohere.Client(COHERE_API_KEY)

        docs_content = [doc.page_content for doc in docs if doc.page_content != ""]
        response = cohere_client.rerank(
            query=query, documents=docs_content, top_n=5, model="rerank-english-v3.0"
        )

        final_docs = []
        for result in response.results:
            doc = docs[result.index]
            doc.metadata["score"] = result.relevance_score
            final_docs.append(doc)
        return final_docs
