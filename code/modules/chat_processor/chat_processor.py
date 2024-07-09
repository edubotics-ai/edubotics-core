from modules.chat_processor.literal_ai import LiteralaiChatProcessor


class ChatProcessor:
    def __init__(self, config, user, tags=None):
        self.config = config
        self.chat_processor_type = self.config["chat_logging"]["platform"]
        self.logging = self.config["chat_logging"]["log_chat"]
        self.user = user
        if tags is None:
            self.tags = self._create_tags()
        else:
            self.tags = tags
        if self.logging:
            self._init_processor()

    def _create_tags(self):
        tags = []
        tags.append(self.config["vectorstore"]["db_option"])
        return tags

    def _init_processor(self):
        if self.chat_processor_type == "literalai":
            self.processor = LiteralaiChatProcessor(self.user, self.tags)
        else:
            raise ValueError(
                f"Chat processor type {self.chat_processor_type} not supported"
            )

    def _process(self, user_message, assistant_message, source_dict):
        if self.logging:
            return self.processor.process(user_message, assistant_message, source_dict)
        else:
            pass

    async def rag(self, user_query: str, chain, stream):
        user_query_dict = {"input": user_query}
        # Define the base configuration
        config = {
            "configurable": {
                "user_id": self.user["user_id"],
                "conversation_id": self.user["session_id"],
                "memory_window": self.config["llm_params"]["memory_window"],
            }
        }

        # Process the user query using the appropriate method
        if self.logging:
            return await self.processor.rag(
                user_query=user_query_dict, config=config, chain=chain
            )
        else:
            if stream:
                return chain.stream(user_query=user_query_dict, config=config)
            return chain.invoke(user_query=user_query_dict, config=config)
