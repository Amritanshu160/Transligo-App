import streamlit as st
import tempfile
import os
from google import genai

from dotenv import load_dotenv

load_dotenv()

# Configure Google API for audio summarization
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def transcribe_audio(audio_file_path):
    myfile = client.files.upload(file=audio_file_path)
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = ["Transcribe the following audio completely and accurately."
        "Distinguish between different speakers and label each of their dialogues clearly (e.g., Speaker 1, Speaker 2)."
        "Include background comments, unless unintelligible."
        "Maintain a clear dialogue format and paragraph breaks when the speaker changes."
        ,myfile]
    )
    return response.text.strip()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Streamlit app interface
st.title('Audio Transcription App')

with st.expander("About this app"):
    st.write("""
        This app uses Google's generative AI to transcribe audio files. 
        Upload your audio file in WAV or MP3 format and get complete transcription of its content.
    """)

audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)  # Save the uploaded file and get the path
    st.audio(audio_path)

    if st.button('Transcribe Audio'):
        with st.spinner('Transcribing...'):
            transcribed_text = transcribe_audio(audio_path)
            st.info(transcribed_text)