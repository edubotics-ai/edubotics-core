## Terrier Tutor

Welcome to the Terrier Tutor documentation! This guide is designed to help you get started with using and developing your own LLM-based Retrieval-Augmented Generation (RAG) applications.

## What is Terrier Tutor?

Terrier Tutor is a framework designed to simplify the development of LLM-based RAG applications. Whether you're an educator, developer, or researcher, this documentation will walk you through the process of setting up, customizing, and extending Terrier Tutor to suit your specific needs.

## Getting Started

To get started, follow the setup instructions provided at [Setup](https://dl4ds.github.io/dl4ds_tutor/guide/setup/). Additionally, for instructions on running the project on your local machine, visit [Run Locally](https://dl4ds.github.io/dl4ds_tutor/guide/run_locally/).

## File Structure and Coding Guidelines

The project is organized into several key modules and files. Below is an overview of the file structure:
   
```bash
code/
├── app.py                
├── main.py                
└── modules/
    ├── dataloader/        
    ├── vectorstore/       
    ├── retriever/          
    ├── config/            
    ├── chat/            
    └── chat_processor/    
```
Each module is designed to facilitate easy customization and extension. For instance, the `modules/vectorstore/vectorstore.py` file contains the core logic for creating the vector database. If you want to implement your own custom vector store while maintaining consistency and without impacting any downstream logic, you can do so by following these steps:

1. Use the template provided in `modules/vectorstore/base.py`, which outlines the required methods and structure for a vector store.
2. Create a new file, such as `modules/vectorstore/new_vectorstore.py`, and implement your custom vector store based on the template.
3. Finally, integrate your custom vector store by calling it in the `modules/vectorstore/vectorstore.py` file.

This approach ensures that your customizations remain consistent with the existing pipeline and do not affect any other parts of the application.


### Key Files and Directories

- **`app.py`**: This is the FastAPI application that serves as the main entry point. It handles user roles, authentication, and mounts the Chainlit UI as a sub-application. All endpoints, except for the Chainlit-specific ones, are defined here.

- **`main.py`**: This file is the entry point for the Chainlit application, which is mounted as a sub-app in `app.py`. It manages the chat UI, conversation logic, and logs user interactions to LiteralAI.

- **`modules/dataloader/`**: Contains the code for loading data from various sources like URLs, text files, and PDFs.

- **`modules/vectorstore/`**: Handles the creation of the vector database from the loaded data.

- **`modules/retriever/`**: Contains the logic for creating the retriever, which is responsible for fetching relevant data from the vector database.

- **`modules/config/`**: Houses configuration settings for the application, allowing customization and adjustments.

- **`modules/chat/`**: Manages the chat logic, including how conversations are handled within the application.

- **`modules/chat_processor/`**: Responsible for processing chat interactions and logging them to LiteralAI.


With this structure, Terrier Tutor offers a modular and organized approach to building fully-customizable LLM-based RAG applications. By following the guidelines provided in each section of this documentation, you can effectively set up, and deploy your own Tutor application.

For further details on each module and component, refer to the corresponding documentation sections.
