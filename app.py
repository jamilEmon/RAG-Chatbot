import streamlit as st
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os

st.set_page_config(page_title="Company Policy RAG Chatbot", layout="wide")
st.title("Company Policy RAG Chatbot")
st.caption("Ask questions about company policies based on uploaded or stored documents.")

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload a company policy PDF", type=["pdf"])
url_input = st.sidebar.text_input("Or enter a policy webpage URL")
user_question = st.chat_input("Ask a question about company policies...")

@st.cache_data
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

@st.cache_data
def extract_text_from_url(url):
    response = requests.get(url, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator="\n")

@st.cache_data
def split_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

@st.cache_resource
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma.from_texts(chunks, embeddings)
    return store

@st.cache_resource
def load_chain():
    generator = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
    llm = HuggingFacePipeline(pipeline=generator)
    return load_qa_chain(llm, chain_type="stuff")

if "history" not in st.session_state:
    st.session_state["history"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

uploaded_or_url = False
source_text = ""

if uploaded_file:
    uploaded_or_url = True
    with st.spinner("Processing uploaded PDF..."):
        source_text = extract_text_from_pdf(uploaded_file)

if url_input and not uploaded_file:
    uploaded_or_url = True
    with st.spinner("Fetching and processing URL..."):
        source_text = extract_text_from_url(url_input)

if uploaded_or_url and source_text:
    chunks = split_text(source_text)
    st.session_state["vectorstore"] = build_vectorstore(chunks)
    st.success("Document indexed successfully. You can now ask questions.")
else:
    sample_files = [
        "hr_leave_policy.txt",
        "it_acceptable_use_policy.txt",
        "remote_work_policy.txt"
    ]
    combined_text = ""
    for file_name in sample_files:
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                combined_text += f.read() + "\n"
    if combined_text and st.session_state["vectorstore"] is None:
        chunks = split_text(combined_text)
        st.session_state["vectorstore"] = build_vectorstore(chunks)
        st.info("Using default sample policy documents for Q&A.")

if user_question:
    if not st.session_state["vectorstore"]:
        st.warning("No data available.")
    else:
        docs = st.session_state["vectorstore"].similarity_search(user_question, k=3)
        chain = load_chain()
        response = chain.run(input_documents=docs, question=user_question)
        st.session_state["history"].append({"q": user_question, "a": response, "sources": [d.page_content for d in docs]})
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(response)
        with st.expander("Source References"):
            for i, d in enumerate(docs):
                excerpt = d.page_content[:600].strip().replace("\n", " ")
                st.markdown(f"**Source {i+1}:** {excerpt}...")

if st.session_state["history"]:
    st.sidebar.header("Conversation History")
    for i, chat in enumerate(reversed(st.session_state["history"][-10:])):
        st.sidebar.markdown(f"**Q:** {chat['q']}")
        st.sidebar.markdown(f"**A:** {chat['a'][:120]}...")
