# from .env setup all constants here

import os
from dotenv import load_dotenv

load_dotenv(".env")

# Centralized definition of required constants for easy management and access
TIMEOUT = os.getenv("TIMEOUT", 60)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
