import requests
import json
from typing import List
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def gather_metadata(files, urls, config):
    pass


def filter_assignment_urls(files, config):
    assignment_pattern = config["metadata"]["assignment_base_link"]
    assignment_urls = []
    for file in files:
        if assignment_pattern in file:
            assignment_urls.append(file)

    return assignment_urls


def filter_lecture_urls(files, urls, config):
    lecture_pattern = config["metadata"]["lectures_pattern"]
    lecture_urls = []
    for file in files:
        if lecture_pattern in file:
            lecture_urls.append(file)

    return lecture_urls


class LLMMetadataExtractor:
    """
    Extracts metadata from a given webpage using an LLM.
    """

    def __init__(self, fields: List[str]):
        self.client = OpenAI()
        self.fields = fields

    def extract_metadata(self, url: str) -> dict:
        # Fetch and parse the webpage
        response = requests.get(url, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the main content (you might need to adjust this based on the page structure)
        content = soup.find("main") or soup.find("body")
        text = content.get_text(separator="\n", strip=True)

        fields_str = ", ".join(self.fields)

        prompt = f"""
        Extract the following metadata from the given webpage about a course assignment:
        {fields_str}

        Please format the output as a JSON object with keys: {fields_str}.
        If applicable, the source_file is the link that points to an assignment file (e.g. .ipynb, .pdf, etc).
        Usually, it's under an <a> tag with the texts "Download", "View" or "notebook".
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

            # TODO: This is a hack to get the source_file. We need to improve the LLM output.
            try:
                source_file = soup.find("a", string=metadata["source_file"])
                metadata["source_file"] = source_file["href"]
            except Exception as e:
                print("Error: Could not find source_file in the webpage")
                print(e)

        except json.JSONDecodeError as e:
            print("Error: Could not parse JSON from LLM response")
            print(e)
            metadata = {}

        return metadata


if __name__ == "__main__":
    extractor = LLMMetadataExtractor(
        fields=["title", "due_date", "release_date", "source_file"]
    )
    metadata = extractor.extract_metadata(
        "https://tools4ds.github.io/fa2024/assignments/01_assignment.html"
    )
    print(json.dumps(metadata, indent=2))
