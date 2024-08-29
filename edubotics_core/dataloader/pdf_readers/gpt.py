import base64
import requests

from io import BytesIO
from openai import OpenAI
from pdf2image import convert_from_path
from langchain.schema import Document
from edubotics_core.config.constants import TIMEOUT, OPENAI_API_KEY


class GPTParser:
    """
    This class uses OpenAI's GPT-4o mini model to parse PDFs and extract text, images and equations.
    It is the most advanced parser in the system and is able to handle complex formats and layouts
    """

    def __init__(self):
        self.client = OpenAI()
        self.api_key = OPENAI_API_KEY
        self.prompt = """
         The provided documents are images of PDFs of lecture slides of deep learning material.
         They contain LaTeX equations, images, and text.
         The goal is to extract the text, images and equations from the slides and convert everything to markdown format. Some of the equations may be complicated.
         The markdown should be clean and easy to read, and any math equation should be converted to LaTeX, between $$.
         For images, give a description and if you can, a source. Separate each page with '---'.
         Just respond with the markdown. Do not include page numbers or any other metadata. Do not try to provide titles. Strictly the content.
         """

    def parse(self, pdf_path):
        images = convert_from_path(pdf_path)

        encoded_images = [self.encode_image(image) for image in images]

        chunks = [encoded_images[i : i + 5] for i in range(0, len(encoded_images), 5)]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        output = ""
        for chunk_num, chunk in enumerate(chunks):
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
                for image in chunk
            ]

            content.insert(0, {"type": "text", "text": self.prompt})

            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": content}],
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )

            resp = response.json()

            chunk_output = (
                resp["choices"][0]["message"]["content"]
                .replace("```", "")
                .replace("markdown", "")
                .replace("````", "")
            )

            output += chunk_output + "\n---\n"

        output = output.split("\n---\n")
        output = [doc for doc in output if doc.strip() != ""]

        documents = [
            Document(page_content=page, metadata={"source": pdf_path, "page": i})
            for i, page in enumerate(output)
        ]
        return documents

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
