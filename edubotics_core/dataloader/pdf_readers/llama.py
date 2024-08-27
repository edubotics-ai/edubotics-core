import os
import requests
from llama_parse import LlamaParse
from langchain.schema import Document
from edubotics_core.config.constants import OPENAI_API_KEY, LLAMA_CLOUD_API_KEY, TIMEOUT
from edubotics_core.dataloader.helpers import download_pdf_from_url


class LlamaParser:
    def __init__(self):
        self.GPT_API_KEY = OPENAI_API_KEY
        self.LLAMA_CLOUD_API_KEY = LLAMA_CLOUD_API_KEY
        self.parse_url = "https://api.cloud.llamaindex.ai/api/parsing/upload"
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {LLAMA_CLOUD_API_KEY}",
        }
        self.parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=True,
            language="en",
            gpt4o_mode=False,
            # gpt4o_api_key=OPENAI_API_KEY,
            parsing_instruction="The provided documents are PDFs of lecture slides of deep learning material. They contain LaTeX equations, images, and text. The goal is to extract the text, images and equations from the slides. The markdown should be clean and easy to read, and any math equation should be converted to LaTeX format, between $ signs. For images, if you can, give a description and a source.",
        )

    def parse(self, pdf_path):
        if not os.path.exists(pdf_path):
            pdf_path = download_pdf_from_url(pdf_path)

        documents = self.parser.load_data(pdf_path)
        document = [document.to_langchain_format() for document in documents][0]

        content = document.page_content
        pages = content.split("\n---\n")
        pages = [page.strip() for page in pages]

        documents = [
            Document(page_content=page, metadata={"source": pdf_path, "page": i})
            for i, page in enumerate(pages)
        ]

        return documents

    def make_request(self, pdf_url):
        payload = {
            "gpt4o_mode": "false",
            "parsing_instruction": "The provided document is a PDF of lecture slides of deep learning material. They contain LaTeX equations, images, and text. The goal is to extract the text, images and equations from the slides and convert them to markdown format. The markdown should be clean and easy to read, and any math equation should be converted to LaTeX, between $$. For images, give a description and if you can, a source.",
        }

        files = [
            (
                "file",
                (
                    "file",
                    requests.get(pdf_url, timeout=TIMEOUT).content,
                    "application/octet-stream",
                ),
            )
        ]

        response = requests.request(
            "POST", self.parse_url, headers=self.headers, data=payload, files=files
        )

        return response.json()["id"], response.json()["status"]

    async def get_result(self, job_id):
        url = (
            f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/markdown"
        )

        response = requests.request("GET", url, headers=self.headers, data={})

        return response.json()["markdown"]

    async def _parse(self, pdf_path):
        job_id, status = self.make_request(pdf_path)

        while status != "SUCCESS":
            url = f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}"
            response = requests.request("GET", url, headers=self.headers, data={})
            status = response.json()["status"]

        result = await self.get_result(job_id)

        documents = [Document(page_content=result, metadata={"source": pdf_path})]

        return documents

    # async def _parse(self, pdf_path):
    #     return await self._parse(pdf_path)
