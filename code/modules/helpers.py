import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse
import chainlit as cl
from langchain import PromptTemplate
from modules.constants import *

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
    source_elements_dict = {}
    source_elements = []
    found_sources = []

    source_dict = {}  # Dictionary to store URL elements

    for idx, source in enumerate(res["source_documents"]):
        source_metadata = source.metadata
        url = source_metadata["source"]

        if url not in source_dict:
            source_dict[url] = [source.page_content]
        else:
            source_dict[url].append(source.page_content)

    for source_idx, (url, text_list) in enumerate(source_dict.items()):
        full_text = ""
        for url_idx, text in enumerate(text_list):
            full_text += f"Source {url_idx+1}:\n {text}\n\n\n"
        source_elements.append(cl.Text(name=url, content=full_text))
        found_sources.append(url)

    if found_sources:
        answer += f"\n\nSources: {', '.join(found_sources)} "
    else:
        answer += f"\n\nNo source found."

    # for idx, source in enumerate(res["source_documents"]):
    #     title = source.metadata["source"]

    #     if title not in source_elements_dict:
    #         source_elements_dict[title] = {
    #             "page_number": [source.metadata["page"]],
    #             "url": source.metadata["source"],
    #             "content": source.page_content,
    #         }

    #     else:
    #         source_elements_dict[title]["page_number"].append(source.metadata["page"])
    #     source_elements_dict[title][
    #         "content_" + str(source.metadata["page"])
    #     ] = source.page_content
    #     # sort the page numbers
    #     # source_elements_dict[title]["page_number"].sort()

    # for title, source in source_elements_dict.items():
    #     # create a string for the page numbers
    #     page_numbers = ", ".join([str(x) for x in source["page_number"]])
    #     text_for_source = f"Page Number(s): {page_numbers}\nURL: {source['url']}"
    #     source_elements.append(cl.Pdf(name="File", path=title))
    #     found_sources.append("File")
    #     # for pn in source["page_number"]:
    #     #     source_elements.append(
    #     #         cl.Text(name=str(pn), content=source["content_"+str(pn)])
    #     #     )
    #     #     found_sources.append(str(pn))

    # if found_sources:
    #     answer += f"\nSource:{', '.join(found_sources)}"
    # else:
    #     answer += f"\nNo source found."

    return answer, source_elements
