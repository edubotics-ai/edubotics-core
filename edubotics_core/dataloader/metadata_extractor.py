import requests
import json
import os
from typing import List
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMMetadataExtractor:
    """
    Extracts metadata from a given webpage using an LLM.
    """

    def __init__(self, fields: List[str]):
        self.client = OpenAI()
        self.fields = fields

    def extract_metadata(self, url: str) -> dict:
        # Fetch and parse the webpage
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the main content (you might need to adjust this based on the page structure)
        content = soup.find("main") or soup.find("body")
        text = content.get_text(separator="\n", strip=True)

        fields_str = ", ".join(self.fields)

        prompt = f"""
        Extract the following metadata from the given webpage about a course assignment:
        {fields_str}

        Please format the output as a JSON object with keys: {fields_str}.
        If any information is not found, set the value to null.

        Text:
        {text[:4000]}

        JSON Output:
        """

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts metadata from course assignment texts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        try:
            metadata = (
                response.choices[0]
                .message.content.replace("```json\n", "")
                .replace("\n```", "")
            )
            metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            print("Error: Could not parse JSON from LLM response")
            print(e)
            metadata = {}

        return metadata


if __name__ == "__main__":
    extractor = LLMMetadataExtractor(fields=["title", "due_date", "release_date"])
    metadata = extractor.extract_metadata(
        "https://tools4ds.github.io/fa2024/assignments/02_assignment.html"
    )
    print(json.dumps(metadata, indent=2))
