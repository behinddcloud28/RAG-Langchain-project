# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import openai 
from dotenv import load_dotenv
import shutil
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFDirectoryLoader


import os
print("API Key:", os.getenv("OPENAI_API_KEY"))
# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "Data"

#Define the embedding function
embedding_function = OpenAIEmbeddings()

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader("./Data")
    return document_loader.load()


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


# def save_to_chroma(chunks: list[Document]):
#     # Clear out the database first.
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)

#     # Create a new DB from the documents.
#     db = Chroma.from_documents(
#         chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
#     )
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

#save to chroma
def save_to_chroma(chunks: list[Document]) -> None:
    # clear out the database first if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    #create a chroma client with proper settings
    client = chromadb.Client(Settings(
        is_persistent=True,
        persist_directory=CHROMA_PATH,
        allow_reset=True
    ))

    #Create a new DB from the documents
    db =Chroma(
        client=client,
        collection_name ="alice_in_wonderland",
        embedding_function= embedding_function,
    )

    #Add the document to the database
    db.add_documents(chunks)

    #Confim the chunks were daved
    print(f"Saved{len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()