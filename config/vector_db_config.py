from dotenv import load_dotenv
load_dotenv()
import os
from uuid import uuid4

import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters.character import CharacterTextSplitter

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize the vector store
# check if the vector store exists
if os.path.exists("faiss_vector_store"):
    vector_store = FAISS.load_local(
        "faiss_vector_store", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
else:
    vector_store = FAISS(
        embedding_function=embeddings, 
        index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

# Split the text into smaller chunks
def split_text(text, chunk_size=1000):
    """Splits the text into smaller chunks of specified size."""
    char_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=chunk_size
    )
    chunks = char_splitter.split_text(text)
    return chunks

# Insert data to the vector store
def add_data_to_vector_store(filename, data):
    # create document chunks
    chunks = split_text(data)

    # create id for the document
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "filename": filename,
                "chunk_order": i,
            }
        )
        documents.append(doc)
    
    doc_ids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=documents, ids=doc_ids)
    vector_store.save_local("faiss_vector_store")

# Query the vector store
def query_vector_store(query, k=5, filename=None):

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter={"filename": filename} if filename else None
    )

    # reverse the results
    results = results[::-1]

    return results

# # Delete the vector store
def reset_vector_store():
    if os.path.exists("faiss_vector_store"):
        os.remove("faiss_vector_store")
