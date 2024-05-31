import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse
import chainlit as cl
from langchain import PromptTemplate
import requests
from bs4 import BeautifulSoup

try:
    from modules.constants import *
except:
    from constants import *

"""
Ref: https://python.plainenglish.io/scraping-the-subpages-on-a-website-ea2d4e3db113
"""


class WebpageCrawler:
    def __init__(self):
        pass

    def getdata(self, url):
        r = requests.get(url)
        return r.text

    def url_exists(self, url):
        try:
            response = requests.head(url)
            return response.status_code == 200
        except requests.ConnectionError:
            return False

    def get_links(self, website_link, base_url=None):
        if base_url is None:
            base_url = website_link
        html_data = self.getdata(website_link)
        soup = BeautifulSoup(html_data, "html.parser")
        list_links = []
        for link in soup.find_all("a", href=True):

            # clean the link
            # remove empty spaces
            link["href"] = link["href"].strip()
            # Append to list if new link contains original link
            if str(link["href"]).startswith((str(website_link))):
                list_links.append(link["href"])

            # Include all href that do not start with website link but with "/"
            if str(link["href"]).startswith("/"):
                if link["href"] not in self.dict_href_links:
                    print(link["href"])
                    self.dict_href_links[link["href"]] = None
                    link_with_www = base_url + link["href"][1:]
                    if self.url_exists(link_with_www):
                        print("adjusted link =", link_with_www)
                        list_links.append(link_with_www)

        # Convert list of links to dictionary and define keys as the links and the values as "Not-checked"
        dict_links = dict.fromkeys(list_links, "Not-checked")
        return dict_links

    def get_subpage_links(self, l, base_url):
        for link in tqdm(l):
            print("checking link:", link)
            if not link.endswith("/"):
                l[link] = "Checked"
                dict_links_subpages = {}
            else:
                # If not crawled through this page start crawling and get links
                if l[link] == "Not-checked":
                    dict_links_subpages = self.get_links(link, base_url)
                    # Change the dictionary value of the link to "Checked"
                    l[link] = "Checked"
                else:
                    # Create an empty dictionary in case every link is checked
                    dict_links_subpages = {}
            # Add new dictionary to old dictionary
            l = {**dict_links_subpages, **l}
        return l

    def get_all_pages(self, url, base_url):
        dict_links = {url: "Not-checked"}
        self.dict_href_links = {}
        counter, counter2 = None, 0
        while counter != 0:
            counter2 += 1
            dict_links2 = self.get_subpage_links(dict_links, base_url)
            # Count number of non-values and set counter to 0 if there are no values within the dictionary equal to the string "Not-checked"
            # https://stackoverflow.com/questions/48371856/count-the-number-of-occurrences-of-a-certain-value-in-a-dictionary-in-python
            counter = sum(value == "Not-checked" for value in dict_links2.values())
            dict_links = dict_links2
        checked_urls = [
            url for url, status in dict_links.items() if status == "Checked"
        ]
        return checked_urls


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
        source_elements.append(cl.Text(name=name, content=source_data["text"]))

        # Add a PDF element if the source is a PDF file
        if source_data["url"].lower().endswith(".pdf"):
            name = f"Source {idx + 1} PDF\n"
            full_answer += name
            pdf_url = f"{source_data['url']}#page={source_data['page']+1}"
            source_elements.append(cl.Pdf(name=name, url=pdf_url))

    # Finally, include lecture metadata for each unique source
    # displayed_urls = set()
    # full_answer += "\n**Metadata:**\n"
    # for url_name, source_data in source_dict.items():
    #     if source_data["url"] not in displayed_urls:
    #         full_answer += f"\nSource: {source_data['url']}\n"
    #         full_answer += f"Type: {source_data['source_type']}\n"
    #         full_answer += f"TL;DR: {source_data['lecture_tldr']}\n"
    #         full_answer += f"Lecture Recording: {source_data['lecture_recording']}\n"
    #         full_answer += f"Suggested Readings: {source_data['suggested_readings']}\n"
    #         displayed_urls.add(source_data["url"])
    full_answer += "\n**Metadata:**\n"
    for url_name, source_data in source_dict.items():
        full_answer += f"\nSource: {source_data['url']}\n"
        full_answer += f"Page: {source_data['page']}\n"
        full_answer += f"Type: {source_data['source_type']}\n"
        full_answer += f"TL;DR: {source_data['lecture_tldr']}\n"
        full_answer += f"Lecture Recording: {source_data['lecture_recording']}\n"
        full_answer += f"Suggested Readings: {source_data['suggested_readings']}\n"

    return full_answer, source_elements


def get_lecture_metadata(schedule_url):
    """
    Function to get the lecture metadata from the schedule URL.
    """
    lecture_metadata = {}

    # Get the main schedule page content
    r = requests.get(schedule_url)
    soup = BeautifulSoup(r.text, "html.parser")

    # Find all lecture blocks
    lecture_blocks = soup.find_all("div", class_="lecture-container")

    for block in lecture_blocks:
        try:
            # Extract the lecture title
            title = block.find("span", style="font-weight: bold;").text.strip()

            # Extract the TL;DR
            tldr = block.find("strong", text="tl;dr:").next_sibling.strip()

            # Extract the link to the slides
            slides_link_tag = block.find("a", title="Download slides")
            slides_link = slides_link_tag["href"].strip() if slides_link_tag else None

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

            # Add to the dictionary
            slides_link = f"https://dl4ds.github.io{slides_link}"
            lecture_metadata[slides_link] = {
                "tldr": tldr,
                "title": title,
                "lecture_recording": recording_link,
                "suggested_readings": suggested_readings,
            }
        except Exception as e:
            print(f"Error processing block: {e}")
            continue

    return lecture_metadata
