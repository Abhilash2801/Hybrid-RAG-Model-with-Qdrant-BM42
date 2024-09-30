import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models
from PyPDF2 import PdfReader
import logging
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List
import uuid
from fastembed import SparseTextEmbedding
class QdrantIndexing:
    def __init__(self, pdf_path):
        """
        Initialize the QdrantIndexing object.
        """
        self.pdf_path = pdf_path
        # Load model directly
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "collection_bm42"
        self.document_text = ""
        logging.info("QdrantIndexing object initialized.")

    def read_pdf(self):
        """
        Read text from the PDF file.
        """
        try:
            reader = PdfReader(self.pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()  # Extract text from each page
            self.document_text = text
            logging.info(f"Extracted text from PDF: {self.pdf_path}")
        except Exception as e:
            logging.error(f"Error reading PDF: {e}")

    def client_collection(self):
        """
        Create a collection in Qdrant vector database.
        """
        if self.qdrant_client.collection_exists("collection_bm42"):
            self.qdrant_client.delete_collection("collection_bm42")
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                     'dense': models.VectorParams(
                         size=384,
                         distance = models.Distance.COSINE,
                     )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                              index=models.SparseIndexParams(
                                on_disk=False,              
                            ),
                        )
                    }
            )
            logging.info(f"Created collection '{self.collection_name}' in Qdrant vector database.")
    def chunk_text(self, text: str) -> List[str]:
        """
        Split the text into overlapping chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        chunks = splitter.split_text(text)
        return chunks
    def create_sparse_vector(self, text):
        """
        Create a sparse vector from the text using the SPLADE model.
        """
        embeddings = list(self.sparse_embedding_model.embed([text]))[0]
        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            sparse_vector = models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
            return sparse_vector
        else:
            raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")

    def get_dense_embedding(self, text):
        """
        Get dense embedding for the given text using BERT-based model.
        """
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode(text).tolist()
        return embedding  # Convert to numpy array

    def document_insertion(self):
        """
        Insert the document text along with its dense and sparse vectors into Qdrant.
        """
        chunks = self.chunk_text(self.document_text)
        for chunk_index, chunk in enumerate(chunks):
            dense_embedding = self.get_dense_embedding(chunk)
            sparse_vector = self.create_sparse_vector(chunk)
            chunk_id = str(uuid.uuid4())
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id" : chunk_id,
                        "vector" : {
                            'dense': dense_embedding,
                            'sparse': sparse_vector,
                        },
                        "payload" : {
                            'chunk_index': chunk_index,
                            'text': chunk,
                        }
                    }]
            )
            logging.info(f"Inserted chunk {chunk_index + 1}/{len(chunks)} into Qdrant.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pdf_file_path = "data/Module-1.pdf"
    indexing = QdrantIndexing(pdf_path=pdf_file_path)
    indexing.read_pdf()
    indexing.client_collection()
    indexing.document_insertion()
