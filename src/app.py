import streamlit as st
import os
import base64
from PyPDF2 import PdfReader
from indexer import QdrantIndexing
from retriver import retriver
from generate import generate
from qdrant_client import QdrantClient

# Streamlit layout
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Hybrid RAG using BM42</h1>", unsafe_allow_html=True)
# Initialize session state for storing question, answer, PDF, and refresh flag
if 'question' not in st.session_state:
    st.session_state['question'] = ''
if 'answer' not in st.session_state:
    st.session_state['answer'] = ''
if 'pdf_uploaded' not in st.session_state:
    st.session_state['pdf_uploaded'] = False
if 'pdf_file_name' not in st.session_state:
    st.session_state['pdf_file_name'] = None
if 'refresh_page' not in st.session_state:
    st.session_state['refresh_page'] = False
if 'indexing_complete' not in st.session_state:
    st.session_state['indexing_complete'] = False

# Function to delete the uploaded PDF and remove the Qdrant collection
def clear_data():
    qdrant_client = QdrantClient(url="http://localhost:6333")
    
    # Check and delete the collection from Qdrant
    if qdrant_client.collection_exists("collection_bm42"):
        qdrant_client.delete_collection("collection_bm42")
    
    # Remove the uploaded PDF file from the temp directory
    if st.session_state['pdf_file_name'] and os.path.exists(f"temp/{st.session_state['pdf_file_name']}"):
        os.remove(f"temp/{st.session_state['pdf_file_name']}")
    
    # Reset session states
    st.session_state['question'] = ''
    st.session_state['answer'] = ''
    st.session_state['pdf_uploaded'] = False
    st.session_state['pdf_file_name'] = None
    st.session_state['indexing_complete'] = False

    # Pop file uploader session to reset the file uploader
    st.session_state.pop('file_uploader', None)

    # Set the refresh flag to True to trigger page reload
    st.session_state['refresh_page'] = True

# Function to display PDF
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Sidebar for PDF Upload
st.sidebar.header("Upload your PDF")

# Add a file uploader in the sidebar (PDF Upload)
pdf_file = st.sidebar.file_uploader("Drag and drop your PDF here", type="pdf", key="file_uploader")

# Create two columns for layout
col1, col2 = st.columns([0.4, 0.6])

# PDF processing and display
if pdf_file is not None and not st.session_state['pdf_uploaded']:
    st.session_state['pdf_uploaded'] = True
    st.session_state['pdf_file_name'] = pdf_file.name
    
    # Save the PDF temporarily for display
    with open(f"temp/{pdf_file.name}", "wb") as f:
        f.write(pdf_file.getbuffer())

    # Initialize Qdrant indexing (for document processing)
    with st.spinner("Indexing the document... This may take a moment."):
        collection_name = "collection_bm42"
        indexing = QdrantIndexing(pdf_path=f"temp/{pdf_file.name}")
        indexing.read_pdf()  # Read the PDF text
        indexing.client_collection()  # Create collection in Qdrant
        indexing.document_insertion()  # Insert chunks into Qdrant
        st.session_state['indexing_complete'] = True
    st.sidebar.success("Document indexed successfully!")

# Always display the PDF if it's uploaded
if st.session_state['pdf_uploaded']:
    with col1:
        st.write(f"Displaying PDF: {st.session_state['pdf_file_name']}")
        display_pdf(f"temp/{st.session_state['pdf_file_name']}")

# Text input and Answer Display (right column)
with col2:
    st.header("Got a question? Fire away!")
    st.session_state['question'] = st.text_input("Enter your question here:", st.session_state['question'])
    answer_placeholder = st.empty()  # Placeholder for displaying the answer

    # When a new question is asked and indexing is complete
    if st.session_state['question'] and st.session_state['indexing_complete']:
        with st.spinner("Searching for relevant information..."):
            # Retrieve relevant context from indexed PDF
            search = retriver()
            retrieved_docs = search.hybrid_search(query=st.session_state['question'])
            context = " ".join(retrieved_docs)  # Extract the new context based on the query

        with st.spinner("Generating answer..."):
            # Generate an answer based on the new context
            llm = generate()
            st.session_state['answer'] = llm.llm_query(question=st.session_state['question'], context=context)

        # Display the new answer
        answer_placeholder.write(f"Answer:\n{st.session_state['answer']}")
    elif st.session_state['question'] and not st.session_state['indexing_complete']:
        st.warning("Please wait for the document to be indexed before asking questions.")

    # Clear Button
    if st.button("Clear"):
        clear_data()  # Clear collection, PDF, and reset the state

# Check if page refresh is needed
if st.session_state['refresh_page']:
    st.session_state['refresh_page'] = False  # Reset refresh flag
    st.rerun()  # This forces the entire page to reload