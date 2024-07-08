# Template for chat processor classes


class ChatProcessorBase:
    def __init__(self):
        pass

    def process(self, message):
        """
        Processes and Logs the message
        """
        raise NotImplementedError("process method not implemented")

    async def rag(self, user_query: dict, config: dict, chain):
        """
        Retrieves the response from the chain
        """
        raise NotImplementedError("rag method not implemented")
