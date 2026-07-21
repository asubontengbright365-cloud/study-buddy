import streamlit as st
import os
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# --- CONFIGURATION ---
st.set_page_config(page_title="NotebookLM Clone", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "sources" not in st.session_state:
    st.session_state.sources = []

# --- HELPER FUNCTIONS: PARSERS ---
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_youtube_transcript(url):
    try:
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get('v')
        if not video_id:
            video_id = parsed_url.path.split('/')[-1]
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id[0] if isinstance(video_id, list) else video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# --- HELPER FUNCTIONS: RAG ENGINE ---
def process_documents(files, yt_url):
    texts = []
    
    # Process uploaded files
    if files:
        for file in files:
            content = ""
            if file.name.endswith(".pdf"):
                content = extract_text_from_pdf(file)
            elif file.name.endswith(".docx"):
                content = extract_text_from_docx(file)
            elif file.name.endswith(".txt"):
                content = file.getvalue().decode("utf-8")
            
            texts.append({"text": content, "source": file.name})
            if file.name not in st.session_state.sources:
                st.session_state.sources.append(file.name)

    # Process YouTube URL
    if yt_url:
        content = extract_youtube_transcript(yt_url)
        texts.append({"text": content, "source": yt_url})
        if yt_url not in st.session_state.sources:
            st.session_state.sources.append("YouTube: " + yt_url)

    if not texts:
        return None

    # Chunking & Embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    metadatas = []
    
    for item in texts:
        splits = text_splitter.split_text(item["text"])
        chunks.extend(splits)
        metadatas.extend([{"source": item["source"]} for _ in splits])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001") 
# Note: if that gives you any grief, "gemini-embedding-2" also works on the newest SDK!
    
    # Create in-memory Chroma database
    vector_store = Chroma.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)
    return vector_store

# --- UI: SIDEBAR (Source Panel) ---
with st.sidebar:
    bot_name = st.text_input("Assistant Name", value="Study Buddy")
    st.markdown("---")
    
    st.header("Upload Sources")
    uploaded_files = st.file_uploader("Upload PDFs, DOCX, TXT", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    youtube_url = st.text_input("Paste YouTube URL")
    
    if st.button("Process Materials", type="primary"):
        with st.spinner("Analyzing materials..."):
            vs = process_documents(uploaded_files, youtube_url)
            if vs:
                st.session_state.vector_store = vs
                st.success("Materials processed successfully!")
            else:
                st.warning("Please upload files or provide a valid link.")
    
    st.markdown("---")
    st.header("Your Sources")
    for source in st.session_state.sources:
        st.write(f"- {source}")

# --- UI: MAIN PANEL (Chat Interface) ---
st.title(f"Chat with {bot_name}")

web_search_enabled = st.toggle("Enable Web Search (DuckDuckGo)")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your materials..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)        
        response_text = ""
        citations = []

        if web_search_enabled:
            # DuckDuckGo Fallback
            search = DuckDuckGoSearchRun()
            search_results = search.run(prompt)
            context = f"Web Search Results:\n{search_results}"
            citations = ["DuckDuckGo Web Search"]
        else:
            # RAG via ChromaDB
            if st.session_state.vector_store is None:
                context = "No documents uploaded. Please tell the user to upload documents."
            else:
                docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                citations = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))

        # Prompt Engineering for NotebookLM Behavior
        template = """
        You are an AI tutor named {bot_name}.
        Answer the user's question ONLY using the provided Context. 
        If the answer is not in the Context, politely state that you cannot find the information in the provided materials.
        
        Context:
        {context}
        
        User Question: {question}
        """
        
        qa_prompt = PromptTemplate(template=template, input_variables=["bot_name", "context", "question"])
        chain = qa_prompt | llm
        
        with st.spinner("Thinking..."):
            result = chain.invoke({"bot_name": bot_name, "context": context, "question": prompt})
            response_text = result.content
            
            # Format output with citations
            if citations and st.session_state.vector_store is not None:
                response_text += "\n\n**Sources:**\n" + "\n".join([f"- {c}" for c in citations])
                
            message_placeholder.markdown(response_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response_text})
