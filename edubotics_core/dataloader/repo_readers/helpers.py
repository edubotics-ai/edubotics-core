import nbformat
import requests
import os


def read_notebook_from_url(notebook_url):
    """
    Read the contents of a Jupyter notebook from a URL.

    Args:
        notebook_url (str): The URL of the Jupyter notebook file.

    Returns:
        str: The contents of the Jupyter notebook.
    """
    response = requests.get(notebook_url)
    if response.status_code == 200:
        notebook_content = response.text
        return notebook_content
    else:
        print(f"Failed to fetch notebook from URL: {response.status_code}")
        return None


def read_notebook_from_file(notebook_path):
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
    return extract_notebook_content(notebook_content)


def extract_notebook_content(notebook_content):
    """
    Extract the content from a Jupyter notebook, preserving the order of the cells.

    Args:
        notebook_content (str): The contents of the Jupyter notebook.

    Returns:
        str: The contents of the notebook, with the cells in their original order.
    """
    notebook = nbformat.reads(notebook_content, as_version=4)
    notebook_content = ""
    for cell in notebook.cells:
        if cell.cell_type == 'markdown' or cell.cell_type == 'raw':
            notebook_content += cell.source + "\n"
        elif cell.cell_type == 'code':
            notebook_content += "```python\n" + cell.source + "\n```\n"
    return notebook_content


if __name__ == "__main__":
    notebook_path = '/Users/faridkarimli/Desktop/Programming/AI/edubot-core/edubotics_core/storage/data/assignment1(1).ipynb'
    notebook_content = read_notebook_from_file(notebook_path)
    if notebook_content:
        print(notebook_content)
