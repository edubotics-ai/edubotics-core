import aiohttp
from aiohttp import ClientSession
import asyncio
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag, urlparse
from edubotics_core.config.constants import TIMEOUT


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
        if url.startswith("mailto:"):
            return False
        try:
            response = requests.head(url, timeout=TIMEOUT)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False

    async def get_links(self, session: ClientSession, website_link: str, base_url: str):
        if not website_link.startswith(base_url):
            return []
        elif website_link.startswith("mailto:"):
            return []

        html_data = await self.fetch(session, website_link)
        soup = BeautifulSoup(html_data, "html.parser")
        list_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            full_url = urljoin(base_url, href)
            normalized_url = self.normalize_url(full_url)  # sections removed
            if (
                normalized_url not in self.dict_href_links
                # and self.is_child_url(normalized_url, base_url)
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

        if url.endswith(".ipynb") or url.endswith(".pdf"):
            return False
        else:
            try:
                response = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
                content_type = response.headers.get("Content-Type", "").lower()
                return "text/html" in content_type
            except (requests.RequestException, ValueError, requests.exceptions.Timeout):
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
        if url.startswith("url: "):
            url = url[5:]
        defragged_url, _ = urldefrag(url)
        return defragged_url

    async def find_target_url(self, base_url: str, target_url: str, depth: int) -> str:
        async with aiohttp.ClientSession() as session:
            visited = set()  # To keep track of visited URLs
            return await self._search_links(
                session, base_url, target_url, visited, depth
            )

    async def _search_links(
        self,
        session: ClientSession,
        current_url: str,
        target_url: str,
        visited: set,
        depth: int,
    ) -> str:
        if current_url.startswith("mailto:"):
            return None
        if current_url in visited or depth < 0:
            return None
        visited.add(current_url)

        base_url = urlparse(current_url).netloc
        print(f"base_url: {base_url}")
        links = await self.get_links(session, current_url, base_url)
        for link in links:
            if link == target_url:
                return link
            found_url = await self._search_links(
                session, link, target_url, visited, depth - 1
            )
            if found_url:
                return found_url

        return None
