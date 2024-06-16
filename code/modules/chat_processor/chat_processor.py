from modules.chat_processor.literal_ai import LiteralaiChatProcessor


class ChatProcessor:
    def __init__(self, chat_processor_type, tags=None):
        self.chat_processor_type = chat_processor_type
        self.tags = tags
        self._init_processor()

    def _init_processor(self):
        if self.chat_processor_type == "literalai":
            self.processor = LiteralaiChatProcessor(self.tags)
        else:
            raise ValueError(
                f"Chat processor type {self.chat_processor_type} not supported"
            )

    def _process(self, user_message, assistant_message, source_dict):
        self.processor.process(user_message, assistant_message, source_dict)

    async def rag(self, user_query: str, chain, cb):
        try:
            return await self.processor.rag(user_query, chain, cb)
        except:
            return await chain.acall(user_query, callbacks=[cb])
