from dotenv import load_dotenv
import os

load_dotenv()

# API Keys - Loaded from the .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Prompt Templates

# prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

prompt_template_with_history = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use the history to answer the question if you can.
Chat History:
{chat_history}
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt_template = """
<|im_start|>system
Assistant is an intelligent chatbot designed to help students with questions regarding the course. Only answer questions using the context below and if you're not sure of an answer, you can say "I don't know". Always give a breif and concise answer to the question.

Context:
{context}
<|im_end|>
<|im_start|>user
Question: Who is the instructor for this course?
<|im_end|>
<|im_start|>assistant
The instructor for this course is Prof. Thomas Gardos.
<|im_end|>
<|im_start|>user
Question: {question}
<|im_end|>
<|im_start|>assistant
"""

# Model Paths

LLAMA_PATH = "storage/models/tinyllama.gguf"
MISTRAL_PATH = "storage/models/mistral-7b-v0.1.Q4_K_M.gguf"
