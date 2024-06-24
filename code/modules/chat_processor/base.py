# Template for chat processor classes


class ChatProcessorBase:
    def __init__(self, config):
        self.config = config

    def process(self, message):
        """
        Processes and Logs the message
        """
        raise NotImplementedError("process method not implemented")
