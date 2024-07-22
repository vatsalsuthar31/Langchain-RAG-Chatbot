import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Enter OpenAI API key
os.environ["OPENAI_API_KEY"] = "ENTER_YOUR_API_KEY_HERE"

# Start ChatBot
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Sample documents (This is what it takes as knowledge)
documents = [
    "This is a sample document about AI.",
    "This document explains how to build chatbots using Langchain.",
    "Another document talks about retrieval-augmented generation."
]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
docs = [text_splitter.split_text(doc) for doc in documents]

# Flatten the list of document chunks
flat_docs = [item for sublist in docs for item in sublist]

# Create embeddings for documents
embeddings = OpenAIEmbeddings()

# Use FAISS to create a vector store
vector_store = FAISS.from_texts(flat_docs, embeddings)

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

# Set up RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # Specify the type of chain, e.g., "stuff", "map_reduce"
)


# Function to get a response from the bot
def chat_with_bot(query):
    response = rag_chain.invoke({"query": query})
    return response


# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    bot_response = chat_with_bot(user_input)
    print(f"Bot: {bot_response}")
