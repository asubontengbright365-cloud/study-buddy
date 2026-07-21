import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from youtube_transcript_api import YouTubeTranscriptApi
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
def process_documents(uploaded_files, youtube_url):
    docs = []
    sources = []
    
    # Process uploaded files
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        
        docs.extend(loader.load())
        sources.append(file.name)
        os.remove(tmp_path)
    
    # Process YouTube
    if youtube_url:
        try:
            video_id = youtube_url.split("v=")[1].split("&")[0]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t['text'] for t in transcript])
            from langchain.schema import Document
            docs.append(Document(page_content=text, metadata={"source": youtube_url}))
            sources.append(youtube_url)
        except Exception as e:
            st.error(f"Could not get YouTube transcript: {e}")
    
    if not docs:
        return None
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Use Google embeddings instead of sentence-transformers
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    st.session_state.sources = sources
    return vectorstore

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
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)        
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
