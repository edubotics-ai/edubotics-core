---
title: Dl4ds Tutor
emoji: 🏃
colorFrom: green
colorTo: red
sdk: docker
pinned: false
hf_oauth: true
---

DL4DS Tutor
===========

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

You can find an implementation of the Tutor at https://dl4ds-dl4ds-tutor.hf.space/, which is hosted on Hugging Face [here](https://huggingface.co/spaces/dl4ds/dl4ds_tutor)

To run locally, 

Clone the repository from: https://github.com/DL4DS/dl4ds_tutor    

Put your data under the `storage/data` directory. Note: You can add urls in the urls.txt file, and other pdf files in the `storage/data` directory.    

To create the Vector Database, run the following command:   
```cd code```    
```python -m modules.vectorstore.store_manager```    
(Note: You would need to run the above when you add new data to the `storage/data` directory, or if the ``storage/data/urls.txt`` file is updated. Or you can set ``["vectorstore"]["embedd_files"]`` to True in the `code/modules/config/config.yaml` file, which would embed files from the storage directory everytime you run the below chainlit command.)

To run the chainlit app, run the following command:   
```chainlit run main.py```

See the [docs](https://github.com/DL4DS/dl4ds_tutor/tree/main/docs) for more information.

## Contributing

Please create an issue if you have any suggestions or improvements, and start working on it by creating a branch and by making a pull request to the main branch.
