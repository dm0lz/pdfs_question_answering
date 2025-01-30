import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfs_reader import PDFSReader


class VectorStore:
    def __init__(self, embedding_model: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.index_path = "faiss_index"

    def load_or_create_index(self, urls: Optional[List[str]] = None) -> FAISS:
        if os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            return FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )

        print("Creating new FAISS index...")
        if not urls:
            raise ValueError("URLs required for creating new index")

        pdfs_texts = PDFSReader().read_pdfs_from_urls(urls)
        texts = self._prepare_texts(pdfs_texts)
        vectorstore = FAISS.from_texts(texts, self.embeddings)
        vectorstore.save_local(self.index_path)
        return vectorstore

    def _prepare_texts(self, pdfs_texts: List[str]) -> List[str]:
        combined_text = " ".join(pdfs_texts)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        return text_splitter.split_text(combined_text)
