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

To run locally, 

Clone the repository from: https://github.com/DL4DS/dl4ds_tutor    

Put your data under the `storage/data` directory. Note: You can add urls in the urls.txt file, and other pdf files in the `storage/data` directory.    

To create the Vector Database, run the following command:   
```python code/modules/vector_db.py```

To run the chainlit app, run the following command:   
```chainlit run code/main.py```

See the [docs](https://github.com/DL4DS/dl4ds_tutor/tree/main/docs) for more information.
