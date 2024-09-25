import requests
import base64
from edubotics_core.dataloader.repo_readers.helpers import extract_notebook_content
from edubotics_core.config.constants import (
    GITHUB_USERNAME,
    GITHUB_PERSONAL_ACCESS_TOKEN,
)
import argparse


class GithubReader:
    def __init__(self, username=None, personal_access_token=None):
        """
        Initialize the GithubReader with the username and personal access token.

        Args:
            username (str): The GitHub username for authentication.
            personal_access_token (str): The GitHub personal access token for authentication.
        """
        if username is None:
            self.username = GITHUB_USERNAME
        else:
            self.username = username
        if personal_access_token is None:
            self.personal_access_token = GITHUB_PERSONAL_ACCESS_TOKEN
        else:
            self.personal_access_token = personal_access_token

        self.ignore_files = [
            "README.md",
            ".DS_Store",
            "requirements.txt",
            "LICENSE",
            "COPYING",
            "COPYRIGHT",
            "NOTICE",
            "AUTHORS",
            "CONTRIBUTORS",
            ".gitignore",
        ]

        self.ignore_ext = [
            "csv",
            "pyc",
            "jpg",
            "png",
            "gif",
            "jpeg",
        ]

        self.repo_allow_list = ["release/", "contents/"]

        if not self.personal_access_token:
            raise Warning(
                "Personal access token is not set. You may need to use a personal access token with the correct scopes for private repositories."
            )

    def get_repo_contents(self, url):
        """
        Fetch the contents of a private GitHub repository.

        Args:
            repo_owner (str): The owner of the repository.
            repo_name (str): The name of the repository.
            branch (str, optional): The branch to fetch the contents from. Defaults to 'main'.
            path (str, optional): The path to the repository. Defaults to ''.
        """
        repo_owner, repo_name, branch = self.parse_github_url(url)

        # top level path is ''
        return self.read_github_repo_contents(repo_owner, repo_name, branch)

    def read_github_repo_contents(self, repo_owner, repo_name, branch="main", path=""):
        """
        Fetch the contents of a private GitHub repository.

        Args:
            repo_owner (str): The owner of the repository.
            repo_name (str): The name of the repository.
            branch (str, optional): The branch to fetch the contents from. Defaults to 'main'.

        Returns:
            dict: The contents of the repository, with file paths as keys and file contents as values.
        """
        repo_contents = {}

        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}?ref={branch}"
        auth_string = f"{self.username}:{self.personal_access_token}"
        auth_bytes = auth_string.encode("ascii")
        auth_b64 = base64.b64encode(auth_bytes).decode("ascii")

        headers = {"Authorization": f"Basic {auth_b64}"}

        response = requests.get(url, headers=headers, timeout=60)

        if response.status_code == 200:
            for item in response.json():
                if item["type"] == "file":

                    file_path = item["path"]
                    extension = file_path.split(".")[-1]

                    if self.repo_allow_list:
                        if not any(
                            pattern in file_path for pattern in self.repo_allow_list
                        ):
                            continue
                    if file_path in self.ignore_files or extension in self.ignore_ext:
                        continue

                    file_content = self.get_github_file_content(
                        repo_owner, repo_name, file_path, branch
                    )

                    full_path = f"https://github.com/{repo_owner}/{repo_name}/blob/{branch}/{file_path}"
                    repo_contents[full_path] = file_content

                elif item["type"] == "dir":

                    sub_dir_path = item["path"]
                    sub_dir_contents = self.read_github_repo_contents(
                        repo_owner, repo_name, branch, sub_dir_path
                    )

                    repo_contents.update(sub_dir_contents)
        else:
            print(
                f"Failed to fetch repository contents: {response.status_code}. You may need to use a personal access token with the correct scopes."
            )
        return repo_contents

    def get_github_file_content(
        self, repo_owner: str, repo_name: str, file_path: str, branch: str = "main"
    ):
        """
        Fetch the content of a file from a private GitHub repository.

        Args:
            repo_owner (str): The owner of the repository.
            repo_name (str): The name of the repository.
        file_path (str): The path to the file within the repository.
        branch (str, optional): The branch to fetch the file from. Defaults to 'main'.

        Returns:
        str: The content of the file, or None if the request fails.
        """
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}?ref={branch}"
        auth_string = f"{self.username}:{self.personal_access_token}"
        auth_bytes = auth_string.encode("ascii")
        auth_b64 = base64.b64encode(auth_bytes).decode("ascii")

        headers = {"Authorization": f"Basic {auth_b64}"}
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code == 200:
            content = response.json()["content"]
            decoded_content = base64.b64decode(content).decode("utf-8")

            if not decoded_content.strip():
                print(f"File {file_path} is empty.")
                return None

            if file_path.endswith(".ipynb"):
                decoded_content = extract_notebook_content(decoded_content)

            return decoded_content
        else:
            print(f"Failed to fetch file: {response.status_code}")
            return None

    @staticmethod
    def parse_github_url(url):
        """
        Parse a GitHub URL to extract the repository owner, name, and branch.

        Args:
            url (str): The GitHub repository URL.

        Returns:
            tuple: A tuple containing (repo_owner, repo_name, branch).
                   If branch is not specified in the URL, it defaults to 'main'.
        """
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL")

        repo_owner = path_parts[0]
        repo_name = path_parts[1]
        branch = "main"  # Default branch

        if len(path_parts) > 3 and path_parts[2] == "tree":
            branch = path_parts[3]

        return repo_owner, repo_name, branch


# Usage example
if __name__ == "__main__":
    # Set up argparse to get username and github_url as arguments
    parser = argparse.ArgumentParser(description="Read GitHub repository contents.")
    parser.add_argument(
        "--github_url", type=str, help="GitHub repository URL", required=True
    )

    args = parser.parse_args()
    github_url = args.github_url

    reader = GithubReader()  # Initialize the GithubReader
    owner, name, branch = GithubReader.parse_github_url(github_url)
    print(f"Owner: {owner}, Repo: {name}, Branch: {branch}")
    repo_contents = reader.get_repo_contents(github_url)

    # The repo_contents dictionary now contains the contents of all files in the repository
    for file_path, file_content in repo_contents.items():
        if file_content is None:
            continue
        print(f"File: {file_path}")
        print(file_content[:20])
        print("---")
