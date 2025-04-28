# Install necessary libraries
# pip install streamlit langchain chromadb tiktoken python-dotenv groq

import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # Using OpenAI embeddings as Groq doesn't directly offer embeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Groq  # Changed from OpenAI to Groq
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"] # Changed to GROQ_API_KEY

# 1. Custom Agentic Task (Add List of Numbers)
def add_numbers(numbers):
    """Adds a list of numbers."""
    return sum(numbers)

# 2. Multiply Two Numbers
def multiply_numbers(a, b):
    """Multiplies two numbers."""
    return a * b

# 3. RAG Pipeline
def rag_pipeline(file_path, query):
    """
    Performs RAG (Retrieval-Augmented Generation) on a given file.
    """
    # Load document
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split documents
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )
    texts = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings() # Using OpenAI embeddings

    # Vectorstore
    db = Chroma.from_documents(texts, embeddings)

    # Retriever
    retriever = db.as_retriever()

    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=Groq(), # Changed to Groq
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Run query
    result = qa({"query": query})
    return result["result"]

# 4. Streamlit UI
def main():
    st.title("Agentic Task, Multiplication, and RAG Pipeline with Groq") # Updated title

    # Custom Agentic Task UI
    st.header("Custom Agentic Task: Add Numbers")
    number_input = st.text_input("Enter numbers separated by commas (e.g., 1,2,3):")
    if number_input:
        numbers = [float(x.strip()) for x in number_input.split(",")]
        sum_result = add_numbers(numbers)
        st.write(f"The sum is: {sum_result}")

    # Multiplication UI
    st.header("Multiply Two Numbers")
    num1 = st.number_input("Enter first number:", value=0.0)
    num2 = st.number_input("Enter second number:", value=0.0)
    multiply_result = multiply_numbers(num1, num2)
    st.write(f"The product is: {multiply_result}")

    # RAG Pipeline UI
    st.header("RAG Pipeline")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    query = st.text_input("Enter your query for the document:")

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with open("temp_file.txt", "w") as f:
            f.write(uploaded_file.read().decode())

        if query:
            answer = rag_pipeline("temp_file.txt", query)
            st.write("Answer from RAG Pipeline:")
            st.write(answer)

if __name__ == "__main__":
    main()