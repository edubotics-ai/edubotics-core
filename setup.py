from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as f:
    readme = f.read()

setup(
    name="edubotics-core",
    version="0.1.0",
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
)
