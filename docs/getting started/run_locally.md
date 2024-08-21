## Running Locally

**Clone the Repository**
```bash
git clone https://github.com/DL4DS/dl4ds_tutor
```

**Put your data under the `storage/data` directory**    
- Add URLs in the `urls.txt` file.    
- Add other PDF files in the `storage/data` directory.    

**To test Data Loading (Optional)**
```bash
cd code
python -m modules.dataloader.data_loader
```

**Create the Vector Database**
```bash
cd code
python -m modules.vectorstore.store_manager
```
- Note: You need to run the above command when you add new data to the `storage/data` directory, or if the `storage/data/urls.txt` file is updated.    

**Run the App**
```bash
cd code
uvicorn app:app --port 7860  
```