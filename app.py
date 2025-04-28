import streamlit as st
from langchain.llms import Groq
from langchain.embeddings import GroqEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
import tempfile

# Set your Groq API key
GROQ_API_KEY = "your-groq-api-key"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Setup Groq LLM
llm = Groq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)

# Utility Functions

def add_numbers(numbers_list):
    return sum(numbers_list)

def multiply_numbers(num1, num2):
    return num1 * num2

def analyze_uploaded_file(file_path, query):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Vectorstore
    vectordb = Chroma.from_documents(docs, GroqEmbeddings(api_key=GROQ_API_KEY))

    # Create RAG pipeline
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    # Run Query
    result = qa.run(query)
    return result

# Streamlit UI

st.set_page_config(page_title="Custom Local Agent", layout="wide")

st.title("🔍 Local Agentic Task Runner (with Groq + Streamlit)")

st.sidebar.header("Select Task")

task = st.sidebar.selectbox("Choose a task", ["Add List of Numbers", "Multiply Two Numbers", "Upload File for Analysis"])

if task == "Add List of Numbers":
    st.subheader("➕ Add List of Numbers")
    numbers_input = st.text_input("Enter numbers separated by commas", "1,2,3,4,5")
    if st.button("Calculate Sum"):
        try:
            numbers_list = [float(x.strip()) for x in numbers_input.split(",")]
            result = add_numbers(numbers_list)
            st.success(f"The sum is: {result}")
        except Exception as e:
            st.error(f"Error: {e}")

elif task == "Multiply Two Numbers":
    st.subheader("✖️ Multiply Two Numbers")
    num1 = st.number_input("Enter first number", value=1.0)
    num2 = st.number_input("Enter second number", value=1.0)
    if st.button("Multiply"):
        result = multiply_numbers(num1, num2)
        st.success(f"The product is: {result}")

elif task == "Upload File for Analysis":
    st.subheader("📄 Upload a File and Analyze (RAG Pipeline)")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    user_query = st.text_input("Enter your question about the document")

    if uploaded_file is not None and user_query:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if st.button("Analyze"):
            with st.spinner("Analyzing the document..."):
                try:
                    answer = analyze_uploaded_file(tmp_file_path, user_query)
                    st.success("Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
