from literalai import LiteralClient
import os
from .base import ChatProcessorBase


class LiteralaiChatProcessor(ChatProcessorBase):
    def __init__(self, tags=None):
        self.literal_client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))
        self.literal_client.reset_context()
        with self.literal_client.thread(name="TEST") as thread:
            self.thread_id = thread.id
            self.thread = thread
            if tags is not None and type(tags) == list:
                self.thread.tags = tags
        print(f"Thread ID: {self.thread}")

    def process(self, user_message, assistant_message, source_dict):
        with self.literal_client.thread(thread_id=self.thread_id) as thread:
            self.literal_client.message(
                content=user_message,
                type="user_message",
                name="User",
            )
            self.literal_client.message(
                content=assistant_message,
                type="assistant_message",
                name="AI_Tutor",
            )

    async def rag(self, user_query: str, chain, cb):
        with self.literal_client.step(
            type="retrieval", name="RAG", thread_id=self.thread_id
        ) as step:
            step.input = {"question": user_query}
            res = await chain.acall(user_query, callbacks=[cb])
            step.output = res
        return res
