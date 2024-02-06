# Documentation

## File Structure:
- `docs/` - Documentation files
- `code/` - Code files
- `storage/` - Storage files
- `vectorstores/` - Vector Databases
- `.env` - Environment Variables
- `Dockerfile` - Dockerfile for Hugging Face
- `.chainlit` - Chainlit Configuration
- `chainlit.md` - Chainlit README
- `README.md` - Repository README
- `.gitignore` - Gitignore file
- `requirements.txt` - Python Requirements
- `.gitattributes` - Gitattributes file

## Code Structure

- `code/main.py` - Main Chainlit App
- `code/config.yaml` - Configuration File to set Embedding related, Vector Database related, and Chat Model related parameters.
- `code/modules/vector_db.py` - Vector Database Creation
- `code/modules/chat_model_loader.py` - Chat Model Loader (Creates the Chat Model)
- `code/modules/constants.py` - Constants (Loads the Environment Variables, Prompts, Model Paths, etc.)
- `code/modules/data_loader.py` - Loads and Chunks the Data
- `code/modules/embedding_model.py` - Creates the Embedding Model to Embed the Data
- `code/modules/llm_tutor.py` - Creates the RAG LLM Tutor
    - The Function `qa_bot()` loads the vector database and the chat model, and sets the prompt to pass to the chat model.
- `code/modules/helpers.py` - Helper Functions    

## Storage and Vectorstores

- `storage/data/` - Data Storage (Put your pdf files under this directory, and urls in the urls.txt file)
- `storage/models/` - Model Storage (Put your local LLMs under this directory)

- `vectorstores/` - Vector Databases (Stores the Vector Databases generated from `code/modules/vector_db.py`)


## Useful Configurations
set these in `code/config.yaml`:
* ``["embedding_options"]["expand_urls"]`` - If set to True, gets and reads the data from all the links under the url provided. If set to False, only reads the data in the url provided.
* ``["embedding_options"]["search_top_k"]`` - Number of sources that the retriever returns
* ``["llm_params]["use_history"]`` - Whether to use history in the prompt or not
* ``["llm_params]["memory_window"]`` - Number of interactions to keep a track of in the history