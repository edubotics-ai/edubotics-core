import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import tempfile
from edubotics_core.config.constants import (
    TIMEOUT,
)  # TODO: MOVE THIS TO APP SPECIFIC DIRECTORY


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


### THIS FUNCTION IS NOT GENERALIZABLE.. IT IS SPECIFIC TO THE COURSE WEBSITE ###
def get_metadata(lectures_url, schedule_url, config):
    """
    Function to get the lecture metadata from the lectures and schedule URLs.
    """
    lecture_metadata = {}

    # Get the main lectures page content
    r_lectures = requests.get(lectures_url, timeout=TIMEOUT)
    soup_lectures = BeautifulSoup(r_lectures.text, "html.parser")

    # Get the main schedule page content
    r_schedule = requests.get(schedule_url, timeout=TIMEOUT)
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
                f"{config['metadata']['slide_base_link']}{slides_link}"
                if slides_link
                else None
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
                f"{config['metadata']['slide_base_link']}{slides_link}"
                if slides_link
                else None
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


def download_pdf_from_url(pdf_url):
    """
    Function to temporarily download a PDF file from a URL and return the local file path.

    Args:
        pdf_url (str): The URL of the PDF file to download.

    Returns:
        str: The local file path of the downloaded PDF file.
    """
    response = requests.get(pdf_url, timeout=TIMEOUT)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        return temp_file_path
    else:
        return None
