import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os 


# Load environment variables
load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

#Initialize model (update model if needed)
model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """"
Answer the question based on the following context:
{context}

Question: {question}
Answer:
"""
prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
# Create a prompt template
PROMPT_TEMPLATE= PromptTemplate(
    input_variables=["context","question"],
    template="Given the context: {context}\n\nAnswer the question: {question}"
)

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text


    # Search the DB.
    db = Chroma(persist_directory="./chroma", embedding_function=embeddings)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
   
    response = model.invoke(prompt)
    response_text = response.content
   

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    print(response_text)


if __name__ == "__main__":
    main()