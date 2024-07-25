from langchain_community.document_loaders import PyMuPDFLoader


class PDFReader:
    def __init__(self):
        pass

    def get_loader(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        return loader

    def parse(self, pdf_path):
        loader = self.get_loader(pdf_path)
        return loader.load()
