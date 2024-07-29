import aiohttp
from aiohttp import ClientSession
import asyncio
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag

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
