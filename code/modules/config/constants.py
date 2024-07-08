from dotenv import load_dotenv
import os

load_dotenv()

# API Keys - Loaded from the .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
LITERAL_API_KEY = os.getenv("LITERAL_API_KEY")
LITERAL_API_URL = os.getenv("LITERAL_API_URL")

OAUTH_GOOGLE_CLIENT_ID = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
OAUTH_GOOGLE_CLIENT_SECRET = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")

opening_message = f"Hey, What Can I Help You With?\n\nYou can me ask me questions about the course logistics, course content, about the final project, or anything else!"

# Model Paths

LLAMA_PATH = "../storage/models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
