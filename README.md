---
title: Dl4ds Tutor
emoji: 🏃
colorFrom: green
colorTo: red
sdk: docker
pinned: false
hf_oauth: true
---

# DL4DS Tutor

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

You can find an implementation of the Tutor at https://dl4ds-dl4ds-tutor.hf.space/, which is hosted on Hugging Face [here](https://huggingface.co/spaces/dl4ds/dl4ds_tutor)

## To Run Locally 

Clone the repository from: https://github.com/DL4DS/dl4ds_tutor    

Put your data under the `storage/data` directory. Note: You can add urls in the urls.txt file, and other pdf files in the `storage/data` directory.    

To create the Vector Database, run the following command:   
```python code/modules/vector_db.py```    
(Note: You would need to run the above when you add new data to the `storage/data` directory, or if the ``storage/data/urls.txt`` file is updated. Or you can set ``["embedding_options"]["embedd_files"]`` to True in the `code/config.yaml` file, which would embed files from the storage directory everytime you run the below chainlit command.)

### Setting Up OAuth

The following steps are from [the official Chainlit Authentication guide](https://docs.chainlit.io/authentication/overview).

On Google Cloud console, on the project drop down, choose "New Project"
Give it a project name, choose organization and select location.

Go to [console.cloud.google.com/apis](console.cloud.google.com/apis)

Under APIs & Services > Enabled APIs & Services

Select "Credentials", then on that screen "+ Create Credentials"
1. Create the OAuth Consent Screen
2. Select Internal
3. Fill out minial app name, e-emails.

Select scope

Create OAuth client ID
Application Type: Web application

https://docs.chainlit.io/authentication/oauth
Set redirect

Run `chainlit create-secret` to generate secret, save it as CHAINLIT_AUTH_SECRET env variable. 

### Evaluation and Observability using Literal AI 

Set up your LiteralAI API key as the LITERAL_API_KEY env variable. 
  
### Start Chainlit

To run the chainlit app, run the following command:   
```chainlit run code/main.py```

See the [docs](https://github.com/DL4DS/dl4ds_tutor/tree/main/docs) for more information.

## Contributing

Please create an issue if you have any suggestions or improvements, and start working on it by creating a branch and by making a pull request to the main branch.
