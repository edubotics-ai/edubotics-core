from dotenv import load_dotenv
import os

load_dotenv()

# API Keys - Loaded from the .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
LITERAL_API_KEY_LOGGING = os.getenv("LITERAL_API_KEY_LOGGING")
LITERAL_API_URL = os.getenv("LITERAL_API_URL")
CHAINLIT_URL = os.getenv("CHAINLIT_URL")
EMAIL_ENCRYPTION_KEY = os.getenv("EMAIL_ENCRYPTION_KEY")

OAUTH_GOOGLE_CLIENT_ID = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
OAUTH_GOOGLE_CLIENT_SECRET = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")

opening_message = "Hey, What Can I Help You With?\n\nYou can me ask me questions about the course logistics, course content, about the final project, or anything else!"
chat_end_message = (
    "I hope I was able to help you. If you have any more questions, feel free to ask!"
)

# Model Paths

LLAMA_PATH = "../storage/models/tinyllama"
