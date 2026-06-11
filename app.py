import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import os
import gdown
# Download PDFs from Google Drive if not present
pdf_files_to_download = {
    "Cambridge IGCSE Chemistry Coursebook 5e (1).pdf": "1MffiyhSpQTeYHdDgyZyHNVtWC1eg3UNW",
    "Cambridge International AS and A Level Chemistry.pdf": "1gZ5faGDU111Jy4jcSSmuI3R8sBSnjoAM",
    "edexcel-international-gcse-9-1-chemistry-st.pdf": "1dAqheS29hMrSWWAWJQY5KweiLekCka4m"
}

for filename, file_id in pdf_files_to_download.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)

# Configure Groq API
API_KEY = "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"
client = Groq(api_key=API_KEY)

# Function to read all PDFs
def load_pdfs():
    text = ""
    folder = os.path.dirname(os.path.abspath(__file__))
    pdf_files = [f for f in os.listdir(folder) if f.endswith('.pdf')]
    for pdf in pdf_files:
        reader = PdfReader(os.path.join(folder, pdf))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# Function to split text into chunks
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Function to find relevant chunks
def find_relevant_chunks(question, chunks, top_k=5):
    question_words = set(question.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words.intersection(chunk_words))
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k]]

# Function to get answer from Groq
def get_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an expert Chemistry tutor for A-Level and IGCSE students.
                Answer questions using only the textbook content provided.
                Give clear step by step explanations with examples.
                If the answer is not in the textbooks say 'This topic is not covered in the loaded textbooks.'"""
            },
            {
                "role": "user",
                "content": f"Textbook Content:\n{context}\n\nStudent Question: {question}"
            }
        ]
    )
    return response.choices[0].message.content

# Streamlit app
st.set_page_config(page_title="Chemistry RAG Tutor", page_icon="🧪")
st.title("🧪 AI Chemistry Tutor")
st.write("Powered by real Chemistry textbooks — Edexcel, Cambridge IGCSE & A-Level")

# Load and process PDFs
if "chunks" not in st.session_state:
    with st.spinner("📚 Loading Chemistry textbooks... Please wait..."):
        raw_text = load_pdfs()
        st.session_state.chunks = get_chunks(raw_text)
    st.success(f"✅ Textbooks loaded! {len(st.session_state.chunks)} sections ready!")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

# User input
question = st.chat_input("Ask a Chemistry question...")

if question:
    st.chat_message("user").write(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("🔍 Searching textbooks..."):
        relevant_chunks = find_relevant_chunks(question, st.session_state.chunks)
        answer = get_answer(question, relevant_chunks)

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})