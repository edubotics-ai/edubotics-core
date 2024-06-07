import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import chainlit as cl
from langchain import PromptTemplate
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
import asyncio
import aiohttp
from aiohttp import ClientSession
from typing import Dict, Any, List

try:
    from modules.constants import *
except:
    from constants import *

"""
Ref: https://python.plainenglish.io/scraping-the-subpages-on-a-website-ea2d4e3db113
"""


class WebpageCrawler:
    def __init__(self):
        self.dict_href_links = {}

    async def fetch(self, session: ClientSession, url: str) -> str:
        async with session.get(url) as response:
            try:
                return await response.text()
            except UnicodeDecodeError:
                return await response.text(encoding="latin1")

    def url_exists(self, url: str) -> bool:
        try:
            response = requests.head(url)
            return response.status_code == 200
        except requests.ConnectionError:
            return False

    async def get_links(self, session: ClientSession, website_link: str, base_url: str):
        html_data = await self.fetch(session, website_link)
        soup = BeautifulSoup(html_data, "html.parser")
        list_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            full_url = urljoin(base_url, href)
            normalized_url = self.normalize_url(full_url)  # sections removed
            if (
                normalized_url not in self.dict_href_links
                and self.is_child_url(normalized_url, base_url)
                and self.url_exists(normalized_url)
            ):
                self.dict_href_links[normalized_url] = None
                list_links.append(normalized_url)

        return list_links

    async def get_subpage_links(
        self, session: ClientSession, urls: list, base_url: str
    ):
        tasks = [self.get_links(session, url, base_url) for url in urls]
        results = await asyncio.gather(*tasks)
        all_links = [link for sublist in results for link in sublist]
        return all_links

    async def get_all_pages(self, url: str, base_url: str):
        async with aiohttp.ClientSession() as session:
            dict_links = {url: "Not-checked"}
            counter = None
            while counter != 0:
                unchecked_links = [
                    link
                    for link, status in dict_links.items()
                    if status == "Not-checked"
                ]
                if not unchecked_links:
                    break
                new_links = await self.get_subpage_links(
                    session, unchecked_links, base_url
                )
                for link in unchecked_links:
                    dict_links[link] = "Checked"
                    print(f"Checked: {link}")
                dict_links.update(
                    {
                        link: "Not-checked"
                        for link in new_links
                        if link not in dict_links
                    }
                )
                counter = len(
                    [
                        status
                        for status in dict_links.values()
                        if status == "Not-checked"
                    ]
                )

            checked_urls = [
                url for url, status in dict_links.items() if status == "Checked"
            ]
            return checked_urls

    def is_webpage(self, url: str) -> bool:
        try:
            response = requests.head(url, allow_redirects=True)
            content_type = response.headers.get("Content-Type", "").lower()
            return "text/html" in content_type
        except requests.RequestException:
            return False

    def clean_url_list(self, urls):
        files, webpages = [], []

        for url in urls:
            if self.is_webpage(url):
                webpages.append(url)
            else:
                files.append(url)

        return files, webpages

    def is_child_url(self, url, base_url):
        return url.startswith(base_url)

    def normalize_url(self, url: str):
        # Strip the fragment identifier
        defragged_url, _ = urldefrag(url)
        return defragged_url


def get_urls_from_file(file_path: str):
    """
    Function to get urls from a file
    """
    with open(file_path, "r") as f:
        urls = f.readlines()
    urls = [url.strip() for url in urls]
    return urls


def get_base_url(url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    return base_url


def get_prompt(config):
    if config["llm_params"]["use_history"]:
        if config["llm_params"]["llm_loader"] == "local_llm":
            custom_prompt_template = tinyllama_prompt_template_with_history
        elif config["llm_params"]["llm_loader"] == "openai":
            custom_prompt_template = openai_prompt_template_with_history
        # else:
        #     custom_prompt_template = tinyllama_prompt_template_with_history # default
        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "chat_history", "question"],
        )
    else:
        if config["llm_params"]["llm_loader"] == "local_llm":
            custom_prompt_template = tinyllama_prompt_template
        elif config["llm_params"]["llm_loader"] == "openai":
            custom_prompt_template = openai_prompt_template
        # else:
        #     custom_prompt_template = tinyllama_prompt_template
        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"],
        )
    return prompt


def get_sources(res, answer):
    source_elements = []
    source_dict = {}  # Dictionary to store URL elements

    for idx, source in enumerate(res["source_documents"]):
        source_metadata = source.metadata
        url = source_metadata["source"]
        score = source_metadata.get("score", "N/A")
        page = source_metadata.get("page", 1)

        lecture_tldr = source_metadata.get("tldr", "N/A")
        lecture_recording = source_metadata.get("lecture_recording", "N/A")
        suggested_readings = source_metadata.get("suggested_readings", "N/A")
        date = source_metadata.get("date", "N/A")

        source_type = source_metadata.get("source_type", "N/A")

        url_name = f"{url}_{page}"
        if url_name not in source_dict:
            source_dict[url_name] = {
                "text": source.page_content,
                "url": url,
                "score": score,
                "page": page,
                "lecture_tldr": lecture_tldr,
                "lecture_recording": lecture_recording,
                "suggested_readings": suggested_readings,
                "date": date,
                "source_type": source_type,
            }
        else:
            source_dict[url_name]["text"] += f"\n\n{source.page_content}"

    # First, display the answer
    full_answer = "**Answer:**\n"
    full_answer += answer

    # Then, display the sources
    full_answer += "\n\n**Sources:**\n"
    for idx, (url_name, source_data) in enumerate(source_dict.items()):
        full_answer += f"\nSource {idx + 1} (Score: {source_data['score']}): {source_data['url']}\n"

        name = f"Source {idx + 1} Text\n"
        full_answer += name
        source_elements.append(
            cl.Text(name=name, content=source_data["text"], display="side")
        )

        # Add a PDF element if the source is a PDF file
        if source_data["url"].lower().endswith(".pdf"):
            name = f"Source {idx + 1} PDF\n"
            full_answer += name
            pdf_url = f"{source_data['url']}#page={source_data['page']+1}"
            source_elements.append(cl.Pdf(name=name, url=pdf_url, display="side"))

    full_answer += "\n**Metadata:**\n"
    for idx, (url_name, source_data) in enumerate(source_dict.items()):
        full_answer += f"\nSource {idx + 1} Metadata:\n"
        source_elements.append(
            cl.Text(
                name=f"Source {idx + 1} Metadata",
                content=f"Source: {source_data['url']}\n"
                f"Page: {source_data['page']}\n"
                f"Type: {source_data['source_type']}\n"
                f"Date: {source_data['date']}\n"
                f"TL;DR: {source_data['lecture_tldr']}\n"
                f"Lecture Recording: {source_data['lecture_recording']}\n"
                f"Suggested Readings: {source_data['suggested_readings']}\n",
                display="side",
            )
        )

    return full_answer, source_elements


def get_metadata(lectures_url, schedule_url):
    """
    Function to get the lecture metadata from the lectures and schedule URLs.
    """
    lecture_metadata = {}

    # Get the main lectures page content
    r_lectures = requests.get(lectures_url)
    soup_lectures = BeautifulSoup(r_lectures.text, "html.parser")

    # Get the main schedule page content
    r_schedule = requests.get(schedule_url)
    soup_schedule = BeautifulSoup(r_schedule.text, "html.parser")

    # Find all lecture blocks
    lecture_blocks = soup_lectures.find_all("div", class_="lecture-container")

    # Create a mapping from slides link to date
    date_mapping = {}
    schedule_rows = soup_schedule.find_all("li", class_="table-row-lecture")
    for row in schedule_rows:
        try:
            date = (
                row.find("div", {"data-label": "Date"}).get_text(separator=" ").strip()
            )
            description_div = row.find("div", {"data-label": "Description"})
            slides_link_tag = description_div.find("a", title="Download slides")
            slides_link = slides_link_tag["href"].strip() if slides_link_tag else None
            slides_link = (
                f"https://dl4ds.github.io{slides_link}" if slides_link else None
            )
            if slides_link:
                date_mapping[slides_link] = date
        except Exception as e:
            print(f"Error processing schedule row: {e}")
            continue

    for block in lecture_blocks:
        try:
            # Extract the lecture title
            title = block.find("span", style="font-weight: bold;").text.strip()

            # Extract the TL;DR
            tldr = block.find("strong", text="tl;dr:").next_sibling.strip()

            # Extract the link to the slides
            slides_link_tag = block.find("a", title="Download slides")
            slides_link = slides_link_tag["href"].strip() if slides_link_tag else None
            slides_link = (
                f"https://dl4ds.github.io{slides_link}" if slides_link else None
            )

            # Extract the link to the lecture recording
            recording_link_tag = block.find("a", title="Download lecture recording")
            recording_link = (
                recording_link_tag["href"].strip() if recording_link_tag else None
            )

            # Extract suggested readings or summary if available
            suggested_readings_tag = block.find("p", text="Suggested Readings:")
            if suggested_readings_tag:
                suggested_readings = suggested_readings_tag.find_next_sibling("ul")
                if suggested_readings:
                    suggested_readings = suggested_readings.get_text(
                        separator="\n"
                    ).strip()
                else:
                    suggested_readings = "No specific readings provided."
            else:
                suggested_readings = "No specific readings provided."

            # Get the date from the schedule
            date = date_mapping.get(slides_link, "No date available")

            # Add to the dictionary
            lecture_metadata[slides_link] = {
                "date": date,
                "tldr": tldr,
                "title": title,
                "lecture_recording": recording_link,
                "suggested_readings": suggested_readings,
            }
        except Exception as e:
            print(f"Error processing block: {e}")
            continue

    return lecture_metadata
