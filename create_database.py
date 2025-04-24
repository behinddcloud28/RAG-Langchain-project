import os
import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

def load_documents():
    document_loader = PyPDFDirectoryLoader("./Data")
    return document_loader.load()

def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Debug: Inspect first 5 chunks
    print("Inspecting first 5 chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i}: {chunk.page_content[:200]}... (Length: {len(chunk.page_content)})")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: List[Document]):
    print(f"Attempting to save {len(chunks)} chunks to Chroma...")
    
    # Clear any existing database (optional, for debugging)
    try:
        vectorstore = Chroma(persist_directory="./chroma", embedding_function=embeddings)
        vectorstore.delete_collection()
        print("Cleared existing Chroma database.")
    except Exception as e:
        print(f"No existing collection to delete: {e}")

    # Create a new Chroma vector store and add documents
    vectorstore = Chroma(persist_directory="./chroma", embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    print(f"Saved {len(chunks)} chunks to Chroma database.")
 
    # Debug: Verify the number of vectors in the database
    client = chromadb.PersistentClient(path="./chroma")
    try:
        collection = client.get_collection(name="langchain")  # Default collection name
        print(f"Total vectors in Chroma database after saving: {collection.count()}")
    except Exception as e:
        print(f"Error accessing Chroma collection: {e}")

def main():
    # Load documents
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    print("Splitting documents into chunks...")
    chunks = split_text(documents)
    print(f"Created {len(chunks)} chunks.")

    # Save chunks to Chroma
    print("Saving chunks to Chroma...")
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()