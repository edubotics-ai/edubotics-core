from setuptools import setup, find_packages
import os

# Read the contents of requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as f:
    readme = f.read()

# Tag is in the format v0.1.0, so we need to remove the v
git_tag = os.environ.get("GITHUB_REF_NAME", "")

if git_tag.startswith("v"):
    version = git_tag[1:]
else:
    version = "0.0.0"  # Fall back to 0.0.0 if we can't find the tag

if not version:
    print("No version found, defaulting to 0.0.0")
    version = "0.0.0"

setup(
    name="edubotics-core",
    version=version,
    packages=find_packages(),
    package_dir={"edubotics-core": "edubotics_core"},
    python_requires=">=3.7",
    install_requires=requirements,
    description="Core modules for edubotics-based LLM AI chatbots",
    author="Xavier Thomas, Farid Karimli, Tom Gardos",
    url="https://github.com/edubotics-ai/edubot-core",
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "vectorstore_creator=edubotics_core.vectorstore.store_manager:main",
        ],
    },
)
