# Hybrid-RAG-Model-with-Qdrant-BM42-and-mixtral-8x7b

This project implements a **Hybrid Retrieval Augmented Generation (RAG) model** that leverages **Qdrant**, a vector database, and BM42 for efficient document retrieval and question answering. The solution combines **sparse** (BM42) and **dense** embeddings to perform hybrid searches, along with language model generation to provide concise answers based on retrieved context.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Indexer](#indexer)
  - [Retriever](#retriever)
  - [LLM Generator](#llm-generator)
  - [Streamlit Application](#streamlit-application)
- [License](#license)

## Overview
The project utilizes:
- **Qdrant** as the vector store for hybrid search combining sparse and dense embeddings.
- **BM42** (a SPLADE-based sparse embedding model) for sparse vector representations.
- **Sentence-Transformers** for dense embeddings.
- A custom **LLM** powered by Groq API for generating responses based on retrieved context.

### Key Concepts
- **Hybrid Retrieval**: This involves combining sparse and dense vector searches for more effective retrieval, particularly with the use of **Reciprocal Rank Fusion (RRF)**.
- **RAG Model**: Retrieval Augmented Generation leverages retrieved documents to provide informed responses, improving traditional question-answering models.

## Features
- **PDF Ingestion**: Upload and process PDF documents for indexing.
- **Hybrid Search**: Efficient search using both sparse (BM42) and dense (Sentence-Transformer) embeddings.
- **Contextual Answer Generation**: Using the Groq API to provide detailed answers based on the retrieved content.
- **Streamlit UI**: Simple interface to upload PDFs, ask questions, and get answers.

## Dependencies
- Python 3.9 or later
- PyPDF2
- QdrantClient
- Sentence-Transformers
- FastEmbed (BM42)
- Streamlit
- LangChain
- Groq API

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Hybrid-RAG-Model-with-Qdrant-BM42.git
   cd Hybrid-RAG-Model-with-Qdrant-BM42
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Start the Qdrant server locally:
   ```bash
   docker run -p 6333:6333 -d qdrant/qdrant
# PDF Query Assistant

## Usage

### Indexer

The `indexer.py` file processes a PDF, splits the text into chunks, and inserts dense and sparse embeddings into Qdrant. To use it:

```bash
python src/indexer.py
```

Make sure you update the `pdf_file_path` in the script to point to the PDF file you want to index.

### Retriever

The `retriever.py` script performs hybrid search using a query. To use it:

```bash
python src/retriever.py
```

You can modify the query string in the script to test different questions.

### LLM Generator

The `generate.py` script interacts with the Groq API to generate answers based on the query and retrieved context. It is used internally by the Streamlit app.

### Streamlit Application

To run the main Streamlit interface, which allows uploading PDFs and asking questions, execute the following:

```bash
streamlit run src/app.py
```

1. Upload a PDF in the sidebar.
2. Wait for the document to be indexed.
3. Ask questions related to the PDF content in the main interface.
4. The answers will be generated and displayed alongside the PDF content.

## License

This project is licensed under the MIT License.
