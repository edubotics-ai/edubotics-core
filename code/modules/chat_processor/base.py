class ChatProcessorBase:
    def __init__(self, config):
        self.config = config

    def process(self, message):
        raise NotImplementedError("process method not implemented")
