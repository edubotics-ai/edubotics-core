from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document


def determine_content_type(document: Document) -> str:
    """
    Determine the content type of a document based on its source.
    TODO: Need to un-hardcode the if statements, use the content type patterns listed in the config instead.
    """
    source = document.metadata["source"]
    if "/assignments/" in source or "midterm" in source:
        return "assignment"
    elif (
        "/lectures/" in source
        or "Course-Notes" in source
        or "slides" in source
        or "lecture" in source
    ):
        return "lecture"
    elif (
        "/discussions/" in source
        or "discussion_slides" in source
        or "discussion" in source
    ):
        return "discussion"
    elif "/project/" in source:
        return "project"
    else:
        return "other"
