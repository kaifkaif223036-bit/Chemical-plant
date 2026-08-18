import os
import io
import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="AI Plant Virtual Assistant", page_icon="🏭", layout="wide")
st.title("🏭 AI Virtual Technical Assistant - Fertilizer Plant")
st.caption("EndTerm Exam Project | ChE-445: AI in Chemical Engineering")

# 2. Safe Imports with Error Handling
try:
    from gTTS import gTTS
    import speech_recognition as sr
    from streamlit_mic_recorder import mic_recorder
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import CSVLoader, PyPDFLoader
    from groq import Groq
except ImportError as e:
    st.error(f"⚠️ Library Loading Error: {e}. Please check requirements.txt.")
    st.stop()

# 3. Groq API Key Handling
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY nahi mili! App Settings -> Secrets mein GROQ_API_KEY add karein.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# 4. State Management
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# 5. Sidebar - Operational Data Upload
st.sidebar.header("📁 Plant Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV or PDF Operational Logs", type=["csv", "pdf"])

if uploaded_file and st.sidebar.button("Process & Index Dataset"):
    with st.spinner("Processing Operational Dataset..."):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            if uploaded_file.name.endswith(".csv"):
                loader = CSVLoader(temp_path)
            else:
                loader = PyPDFLoader(temp_path)

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.sidebar.success("✅ Plant Data successfully indexed in FAISS!")
        except Exception as err:
            st.sidebar.error(f"Error processing file: {err}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# 6. User Interaction Panel
st.subheader("💬 Plant Manager Technical Query Panel")

col1, col2 = st.columns([1, 2])
user_query = ""

with col1:
    st.write("🎙️ **Voice Query (Microphone):**")
    audio = mic_recorder(start_prompt="Record 🎤", stop_prompt="Stop 🔴", key='recorder')
    if audio and 'bytes' in audio:
        recognizer = sr.Recognizer()
        audio_file = io.BytesIO(audio['bytes'])
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                user_query = recognizer.recognize_google(audio_data)
                st.info(f"**Recognized Query:** {user_query}")
            except Exception:
                st.warning("Voice samajh nahi aayi, please dobara try karein.")

with col2:
    st.write("⌨️ **Text Query:**")
    text_input = st.text_input("Ask a technical question about plant conditions:", placeholder="e.g., What caused the pressure drop before the shutdown?")
    if text_input:
        user_query = text_input

# 7. AI Analysis & Voice Generation Pipeline
if user_query:
    with st.spinner("Analyzing Operational Data..."):
        context = ""
        if st.session_state.vector_store:
            docs = st.session_state.vector_store.similarity_search(user_query, k=3)
            context = "\n".join([doc.page_content for doc in docs])
        else:
            context = "No operational dataset uploaded yet. Answer based on general chemical engineering standards."

        system_prompt = f"""You are an expert Virtual Technical Assistant representing an absent Section Manager in a Fertilizer Manufacturing Complex (Ammonia, Urea, Utilities).
        Assist the Plant Manager by analyzing operational logs, identifying process anomalies, and proposing data-driven troubleshooting solutions.

        Uploaded Operational Context:
        {context}
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content

            # Text to Speech Conversion
            tts = gTTS(text=answer, lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            st.session_state.chat_history.append({"query": user_query, "answer": answer, "audio": audio_bytes})
        except Exception as e:
            st.error(f"Error generating response: {e}")

# 8. Output Display
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📋 Technical Analysis & Audio Response")
    latest = st.session_state.chat_history[-1]
    
    st.markdown(f"**Query:** {latest['query']}")
    st.markdown(f"**Assistant Response:**\n{latest['answer']}")
    st.audio(latest['audio'], format='audio/mp3')
