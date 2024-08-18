---
title: AI Class Tutor -- Dev
description: An LLM based AI class tutor with RAG on DL4DS course
emoji: 🐶
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
---
# DL4DS Tutor 🏃

Check out the configuration reference at [Hugging Face Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference).

You can find a "production" implementation of the Tutor running live at [DL4DS Tutor](https://dl4ds-dl4ds-tutor.hf.space/)  from the
Hugging Face [Space](https://huggingface.co/spaces/dl4ds/dl4ds_tutor). It is pushed automatically from the `main` branch of this repo by this
[Actions Workflow](https://github.com/DL4DS/dl4ds_tutor/blob/main/.github/workflows/push_to_hf_space.yml) upon a push to `main`.

A "development" version of the Tutor is running live at [DL4DS Tutor -- Dev](https://dl4ds-tutor-dev.hf.space/) from this Hugging Face
[Space](https://huggingface.co/spaces/dl4ds/tutor_dev). It is pushed automatically from the `dev_branch` branch of this repo by this
[Actions Workflow](https://github.com/DL4DS/dl4ds_tutor/blob/dev_branch/.github/workflows/push_to_hf_space_prototype.yml) upon a push to `dev_branch`.

## Setup

Please visit [setup](https://dl4ds.github.io/dl4ds_tutor/guide/setup/) for more information on setting up the project.

## Running Locally

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

## Documentation

Please visit the [docs](https://dl4ds.github.io/dl4ds_tutor/) for more information.


## Docker 

The HuggingFace Space is built using the `Dockerfile` in the repository. To run it locally, use the `Dockerfile.dev` file.

```bash
docker build --tag dev  -f Dockerfile.dev .
docker run -it --rm -p 8000:8000 dev
```

## Contributing

Please create an issue if you have any suggestions or improvements, and start working on it by creating a branch and by making a pull request to the `dev_branch`.

Please visit [contribute](https://dl4ds.github.io/dl4ds_tutor/guide/contribute/) for more information on contributing.

## Future Work

For more information on future work, please visit [roadmap](https://dl4ds.github.io/dl4ds_tutor/guide/readmap/).
