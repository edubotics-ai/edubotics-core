# from .env setup all constants here

import os
from dotenv import load_dotenv

load_dotenv()

# Required Constants # TODO: MOVE THIS TO APP SPECIFIC DIRECTORY
TIMEOUT = os.getenv("TIMEOUT", 60)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
