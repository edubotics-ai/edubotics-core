# Initial Setup

## Python Environment

Python Version: 3.11

Create a virtual environment and install the required packages:

```bash
conda create -n ai_tutor python=3.11
conda activate ai_tutor
pip install -r requirements.txt
```

## Code Formatting

The codebase is formatted using [black](https://github.com/psf/black), and if making changes to the codebase, ensure that the code is formatted before submitting a pull request. More instructions can be found in `docs/contribute.md`.

## Google OAuth 2.0 Client ID and Secret

To set up the Google OAuth 2.0 Client ID and Secret, follow these steps:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/apis/credentials).
2. Create a new project or select an existing one.
3. Navigate to the "Credentials" page.
4. Click on "Create Credentials" and select "OAuth 2.0 Client ID".
5. Configure the OAuth consent screen if you haven't already.
6. Choose "Web application" as the application type.
7. Enter a name for your client ID and configure the redirect URIs as needed.
8. Click "Create" and copy the generated `Client ID` and `Client Secret`.

Set the following in the .env file (if running locally) or in secrets (if running on Hugging Face Spaces):

```bash
OAUTH_GOOGLE_CLIENT_ID=<your_client_id>
OAUTH_GOOGLE_CLIENT_SECRET=<your_client_secret>
```

## Literal AI API Key

To obtain the Literal AI API key:

1. Sign up or log in to [Literal AI](https://cloud.getliteral.ai/).
2. Navigate to the API Keys section under your account settings.
3. Create a new API key if necessary and copy it.

Set the following in the .env file (if running locally) or in secrets (if running on Hugging Face Spaces):

```bash
LITERAL_API_KEY=<your_api_key>
```

## LlamaCloud API Key

To obtain the LlamaCloud API Key:

1. Go to [LlamaCloud](https://cloud.llamaindex.ai/).
2. Sign up or log in to your account.
3. Navigate to the API section and generate a new API key if necessary.

Set the following in the .env file (if running locally) or in secrets (if running on Hugging Face Spaces):

```bash
LLAMA_CLOUD_API_KEY=<your_api_key>
```

## Hugging Face Access Token

To obtain your Hugging Face access token:

1. Go to [Hugging Face settings](https://huggingface.co/settings/tokens).
2. Log in or create an account.
3. Generate a new token or use an existing one.

Set the following in the .env file (if running locally) or in secrets (if running on Hugging Face Spaces):

```bash
HUGGINGFACE_TOKEN=<your-huggingface-token>
```

# Configuration

The configuration file `code/modules/config.yaml` contains the parameters that control the behaviour of your app.
The configuration file `code/modules/user_config.yaml` contains user-defined parameters.