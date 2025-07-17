import streamlit as st
import tempfile
import os
from google import genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# === Load & configure API key ===
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# === Helper functions ===

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

def transcribe_audio(audio_file_path):
    myfile = client.files.upload(file=audio_file_path)
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents= ["Transcribe this complete audio clip", myfile]
    )
    return response.text.strip()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def save_to_faiss(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_audio_index")

def get_answer(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_audio_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say: "Answer not available in the context."
    \n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return result["output_text"]

# === Streamlit UI ===

st.set_page_config(page_title="🎙️ Audio QA App with Gemini")
st.title("🎙️ Audio Q&A with Gemini")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Upload your audio and ask me anything from it!"}
    ]

with st.expander("📖 About"):
    st.write("Upload your MP3/WAV file → Transcribe with Gemini → Ask questions via chat.")

audio_file = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])

if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)
    st.audio(audio_path, format="audio/mp3")

    if st.button("🔁 Transcribe and Index"):
        with st.spinner("Transcribing with Gemini..."):
            transcript = transcribe_audio(audio_path)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": "Audio successfully transcribed! You can now ask questions based on the transcription."}
        )

        with st.spinner("Processing & saving to FAISS..."):
            chunks = chunk_text(transcript)
            save_to_faiss(chunks)
            st.success("✅ Audio processed and ready for Q&A!")

# === Display Chat History (Top) ===
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# === Chat Input (Bottom) ===
if prompt := st.chat_input("Ask a question about your audio..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_answer(prompt)
            except Exception as e:
                response = f"Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.write(response)   


  

