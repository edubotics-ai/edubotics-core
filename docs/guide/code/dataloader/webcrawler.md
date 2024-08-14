# WebpageCrawler Class Documentation

## Overview

The `WebpageCrawler` class provides a tool for crawling a given webpage and recursively exploring links to fetch all child pages under the same base URL. It validates the accessibility of links and categorizes them as either webpages or non-webpage resources based on their MIME type.

## Class Methods

### `__init__(self)`

Constructor that initializes the `WebpageCrawler` instance.

- **Attributes**:
  - `self.dict_href_links` *(dict)*: A dictionary to store and track the discovered links.

### `async def fetch(self, session: ClientSession, url: str) -> str`

Asynchronously fetches the HTML content of a specified URL.

- **Parameters**:
  - `session` *(ClientSession)*: The session used to make HTTP requests.
  - `url` *(str)*: The URL to fetch.

- **Returns**: The HTML content of the page as a string.

### `def url_exists(self, url: str) -> bool`

Checks if a given URL exists by performing a `HEAD` request.

- **Parameters**:
  - `url` *(str)*: The URL to check.

- **Returns**: `True` if the URL is accessible (status code 200), otherwise `False`.

### `async def get_links(self, session: ClientSession, website_link: str, base_url: str) -> list`

Extracts and normalizes valid links from the HTML content of a webpage.

- **Parameters**:
  - `session` *(ClientSession)*: The session used for making HTTP requests.
  - `website_link` *(str)*: The URL of the webpage to extract links from.
  - `base_url` *(str)*: The base URL to filter out external links.

- **Returns**: A list of normalized URLs that are valid and fall under the base URL.

### `async def get_subpage_links(self, session: ClientSession, urls: list, base_url: str) -> list`

Asynchronously gathers links from multiple webpages.

- **Parameters**:
  - `session` *(ClientSession)*: The session used for making HTTP requests.
  - `urls` *(list)*: A list of URLs to fetch links from.
  - `base_url` *(str)*: The base URL to filter out external links.

- **Returns**: A combined list of all child URLs discovered from the provided list of URLs.

### `async def get_all_pages(self, url: str, base_url: str) -> list`

Recursively crawls a website to gather all valid URLs under the same base URL.

- **Parameters**:
  - `url` *(str)*: The starting URL for the crawl.
  - `base_url` *(str)*: The base URL to restrict the scope of the crawl.

- **Returns**: A list of all discovered URLs categorized as webpages.

### `def is_webpage(self, url: str) -> bool`

Determines if a given URL points to a webpage by checking its `Content-Type`.

- **Parameters**:
  - `url` *(str)*: The URL to check.

- **Returns**: `True` if the URL is an HTML webpage (`Content-Type: text/html`), otherwise `False`.

### `def clean_url_list(self, urls: list) -> tuple`

Categorizes a list of URLs into two lists: one for files and one for webpages.

- **Parameters**:
  - `urls` *(list)*: A list of URLs to categorize.

- **Returns**: A tuple containing two lists:
  - `files` *(list)*: URLs categorized as non-webpage resources.
  - `webpages` *(list)*: URLs categorized as webpages.

### `def is_child_url(self, url: str, base_url: str) -> bool`

Checks if a URL is a child of the base URL.

- **Parameters**:
  - `url` *(str)*: The URL to check.
  - `base_url` *(str)*: The base URL to compare against.

- **Returns**: `True` if the URL starts with the base URL, otherwise `False`.

### `def normalize_url(self, url: str) -> str`

Normalizes a URL by removing any fragment identifiers (e.g., `#section`).

- **Parameters**:
  - `url` *(str)*: The URL to normalize.

- **Returns**: The normalized URL without any fragment identifiers.
