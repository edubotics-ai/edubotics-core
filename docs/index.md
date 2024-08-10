
## Running Locally

Please view `docs/setup.md` for more information on setting up the project.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/DL4DS/dl4ds_tutor
   ```

2. **Put your data under the `storage/data` directory**
   - Add URLs in the `urls.txt` file.
   - Add other PDF files in the `storage/data` directory.

3. **To test Data Loading (Optional)**
   ```bash
   cd code
   python -m modules.dataloader.data_loader
   ```

4. **Create the Vector Database**
   ```bash
   cd code
   python -m modules.vectorstore.store_manager
   ```
   - Note: You need to run the above command when you add new data to the `storage/data` directory, or if the `storage/data/urls.txt` file is updated.

5. **Run the Chainlit App**
   ```bash
   chainlit run main.py
   ```

See the [docs](https://github.com/DL4DS/dl4ds_tutor/tree/main/docs) for more information.

## File Structure

```plaintext
code/
 ├── modules
 │   ├── chat                # Contains the chatbot implementation
 │   ├── chat_processor      # Contains the implementation to process and log the conversations
 │   ├── config              # Contains the configuration files
 │   ├── dataloader          # Contains the implementation to load the data from the storage directory
 │   ├── retriever           # Contains the implementation to create the retriever
 │   └── vectorstore         # Contains the implementation to create the vector database
 ├── public
 │   ├── logo_dark.png       # Dark theme logo
 │   ├── logo_light.png      # Light theme logo
 │   └── test.css            # Custom CSS file
 └── main.py

 
docs/                        # Contains the documentation to the codebase and methods used

storage/
 ├── data                    # Store files and URLs here
 ├── logs                    # Logs directory, includes logs on vector DB creation, tutor logs, and chunks logged in JSON files
 └── models                  # Local LLMs are loaded from here

vectorstores/                # Stores the created vector databases

.env                         # This needs to be created, store the API keys here
```
- `code/modules/vectorstore/vectorstore.py`: Instantiates the `VectorStore` class to create the vector database.
- `code/modules/vectorstore/store_manager.py`: Instantiates the `VectorStoreManager:` class to manage the vector database, and all associated methods.
- `code/modules/retriever/retriever.py`: Instantiates the `Retriever` class to create the retriever.


## Docker 

The HuggingFace Space is built using the `Dockerfile` in the repository. To run it locally, use the `Dockerfile.dev` file.

```bash
docker build --tag dev  -f Dockerfile.dev .
docker run -it --rm -p 8000:8000 dev
```

## Contributing

Please create an issue if you have any suggestions or improvements, and start working on it by creating a branch and by making a pull request to the main branch.

Please view `docs/contribute.md` for more information on contributing.