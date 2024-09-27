import os
import nbformat
import requests
import argparse
from langchain_text_splitters import MarkdownHeaderTextSplitter


def read_notebook_from_url(notebook_url):
    """
    Read the contents of a Jupyter notebook from a URL.

    Args:
        notebook_url (str): The URL of the Jupyter notebook file.

    Returns:
        str: The contents of the Jupyter notebook.
    """
    response = requests.get(notebook_url, timeout=60)
    if response.status_code == 200:
        notebook_content = response.text
        return notebook_content
    else:
        print(f"Failed to fetch notebook from URL: {response.status_code}")
        return None


def read_notebook_from_file(notebook_path, headers_to_split_on):
    """
    Read the contents of a Jupyter notebook from a file.

    Args:
        notebook_path (str): The path to the Jupyter notebook file.

    Returns:
        str: The contents of the Jupyter notebook.
    """
    if not os.path.exists(notebook_path):
        print(f"File {notebook_path} does not exist. Using filepath as URL instead.")
        notebook_content = read_notebook_from_url(notebook_path)
    else:
        with open(notebook_path, "r") as file:
            notebook_content = file.read()
    return extract_notebook_content(notebook_content, headers_to_split_on)


def extract_notebook_content(
    notebook_content,
    headers_to_split_on=[("###", "Section"), ("##", "Subsection"), ("#", "Title")],
):
    """
    Extract the content from a Jupyter notebook, preserving the order of the cells.

    Args:
        notebook_content (str): The contents of the Jupyter notebook.
        headers_to_split_on (list): A list of headers to split the notebook content by. Default is [("###", "Section"), ("##", "Subsection"), ("#", "Title")].

    Returns:
        List[Document]: The contents of the notebook, split by the headers_to_split_on.
    """
    notebook = nbformat.reads(notebook_content, as_version=4)
    content = ""
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            content += cell.source + "\n"
        elif cell.cell_type == "code":
            content += "```python\n" + cell.source + "\n```\n"
        elif cell.cell_type == "raw":
            content += cell.source + "\n"

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    chunks = markdown_splitter.split_text(content)
    return chunks


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Read and print notebook content from a file."
    )

    # Add notebook_path as an argument
    parser.add_argument(
        "--notebook_path",
        type=str,
        help="The path to the Jupyter notebook file (.ipynb) to read.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Read the notebook path from args
    notebook_content = read_notebook_from_file(args.notebook_path)
    for doc in notebook_content:
        print(doc)
        print("---")
