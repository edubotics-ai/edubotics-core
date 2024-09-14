# WebpageCrawler Class Documentation

## Overview

The `WebpageCrawler` class provides tools for crawling a given webpage and recursively exploring links to fetch all child pages under the same base URL. It validates the accessibility of links and categorizes them as either webpages or non-webpage resources based on their MIME type.

## Class Reference

### WebpageCrawler

#### Attributes

- `dict_href_links` (dict): Stores and tracks discovered links during the crawl.

#### Methods

---

##### `__init__()`

Initializes a new instance of the `WebpageCrawler` class.

**Usage**:

```python
crawler = WebpageCrawler()
```

---

##### `async fetch(session: ClientSession, url: str) -> str`

Asynchronously fetches the HTML content of the specified URL.

**Parameters**:

- `session` (ClientSession): The session used to make HTTP requests.
- `url` (str): The URL to fetch.

**Returns**:

- (str): The HTML content of the page.

**Raises**:

- `aiohttp.ClientError`: If the HTTP request fails.

**Usage**:

```python
try:
    html_content = await crawler.fetch(session, url)
except aiohttp.ClientError as e:
    print(f"Failed to fetch {url}: {e}")
```

---

##### `url_exists(url: str) -> bool`

Checks if a given URL is accessible by performing a `HEAD` request.

**Parameters**:

- `url` (str): The URL to check.

**Returns**:

- (bool): `True` if the URL is accessible (status code 200), otherwise `False`.

**Usage**:

```python
if crawler.url_exists(url):
    print("URL exists")
else:
    print("URL does not exist")
```

---

##### `async get_links(session: ClientSession, website_link: str, base_url: str) -> List[str]`

Extracts and normalizes valid links from the HTML content of a webpage.

**Parameters**:

- `session` (ClientSession): The session used for making HTTP requests.
- `website_link` (str): The URL of the webpage to extract links from.
- `base_url` (str): The base URL to filter out external links.

**Returns**:

- (List[str]): A list of normalized URLs that are valid and fall under the base URL.

**Usage**:

```python
links = await crawler.get_links(session, website_link, base_url)
```

---

##### `async get_subpage_links(session: ClientSession, urls: List[str], base_url: str) -> List[str]`

Asynchronously gathers links from multiple webpages.

**Parameters**:

- `session` (ClientSession): The session used for making HTTP requests.
- `urls` (List[str]): A list of URLs to fetch links from.
- `base_url` (str): The base URL to filter out external links.

**Returns**:

- (List[str]): A combined list of all child URLs discovered from the provided list of URLs.

**Usage**:

```python
all_links = await crawler.get_subpage_links(session, urls, base_url)
```

---

##### `async get_all_pages(url: str, base_url: str) -> List[str]`

Recursively crawls a website to gather all valid URLs under the same base URL.

**Parameters**:

- `url` (str): The initial URL to start crawling from.
- `base_url` (str): The base URL to filter out external links.

**Returns**:

- (List[str]): A complete list of all URLs discovered under the base URL.

**Usage**:

```python
all_pages = await crawler.get_all_pages(url, base_url)
```
