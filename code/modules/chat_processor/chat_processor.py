from modules.chat_processor.literal_ai import LiteralaiChatProcessor


class ChatProcessor:
    def __init__(self, config, tags=None):
        self.chat_processor_type = config["chat_logging"]["platform"]
        self.logging = config["chat_logging"]["log_chat"]
        self.tags = tags
        if self.logging:
            self._init_processor()

    def _init_processor(self):
        if self.chat_processor_type == "literalai":
            self.processor = LiteralaiChatProcessor(self.tags)
        else:
            raise ValueError(
                f"Chat processor type {self.chat_processor_type} not supported"
            )

    def _process(self, user_message, assistant_message, source_dict):
        if self.logging:
            return self.processor.process(user_message, assistant_message, source_dict)
        else:
            pass

    async def rag(self, user_query: str, chain, cb):
        if self.logging:
            return await self.processor.rag(user_query, chain, cb)
        else:
            return await chain.acall(user_query, callbacks=[cb])
