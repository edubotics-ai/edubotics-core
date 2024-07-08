from literalai import LiteralClient
from literalai.api import LiteralAPI
from literalai.filter import Filter as ThreadFilter

import os
from .base import ChatProcessorBase


class LiteralaiChatProcessor(ChatProcessorBase):
    def __init__(self, user=None, tags=None):
        super().__init__()
        self.user = user
        self.tags = tags
        self.literal_client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))
        self.literal_api = LiteralAPI(
            api_key=os.getenv("LITERAL_API_KEY"), url=os.getenv("LITERAL_API_URL")
        )
        self.literal_client.reset_context()
        self.user_info = self._fetch_userinfo()
        self.user_thread = self._fetch_user_threads()
        if len(self.user_thread["data"]) == 0:
            self.thread = self._create_user_thread()
        else:
            self.thread = self._get_user_thread()
        self.thread_id = self.thread["id"]

        self.prev_conv = self._get_prev_k_conversations()

    def _get_user_thread(self):
        thread = self.literal_api.get_thread(id=self.user_thread["data"][0]["id"])
        return thread.to_dict()

    def _create_user_thread(self):
        thread = self.literal_api.create_thread(
            name=f"{self.user_info['identifier']}",
            participant_id=self.user_info["metadata"]["id"],
            environment="dev",
        )

        return thread.to_dict()

    def _get_prev_k_conversations(self, k=3):

        steps = self.thread["steps"]
        conversation_pairs = []
        count = 0
        for i in range(len(steps) - 1, 0, -1):
            if (
                steps[i - 1]["type"] == "user_message"
                and steps[i]["type"] == "assistant_message"
            ):
                user_message = steps[i - 1]["output"]["content"]
                assistant_message = steps[i]["output"]["content"]
                conversation_pairs.append((user_message, assistant_message))

                count += 1
                if count >= k:
                    break

        # Return the last k conversation pairs, reversed to maintain chronological order
        return conversation_pairs[::-1]

    def _fetch_user_threads(self):
        filters = filters = [
            {
                "operator": "eq",
                "field": "participantId",
                "value": self.user_info["metadata"]["id"],
            }
        ]
        user_threads = self.literal_api.get_threads(filters=filters)
        return user_threads.to_dict()

    def _fetch_userinfo(self):
        user_info = self.literal_api.get_or_create_user(
            identifier=self.user["user_id"]
        ).to_dict()
        # TODO: Have to do this more elegantly
        # update metadata with unique id for now
        # (literalai seems to not return the unique id as of now,
        # so have to explicitly update it in the metadata)
        user_info = self.literal_api.update_user(
            id=user_info["id"],
            metadata={
                "id": user_info["id"],
            },
        ).to_dict()
        return user_info

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

    async def rag(self, user_query: dict, config: dict, chain):
        with self.literal_client.step(
            type="retrieval", name="RAG", thread_id=self.thread_id, tags=self.tags
        ) as step:
            step.input = {"question": user_query["input"]}
            res = chain.invoke(user_query, config)
            step.output = res
        return res
