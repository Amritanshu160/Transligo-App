import streamlit as st
import tempfile
import os
from google import genai
from gtts.lang import tts_langs

from dotenv import load_dotenv

load_dotenv()

# Configure Google API for audio summarization
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def summarize_audio(audio_file_path,selected_lang):
    myfile = client.files.upload(file=audio_file_path)
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = [f"Please give detailed summary notes(pointwise) of the following audio, without missing anything important.",
                    f"Make ensure that you have to give the summary strictly in the language {selected_lang}",
            myfile]
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
st.title('Language Specific Summarization')

with st.expander("About this app"):
    st.write("""
        This app uses Google's generative AI to summarize uploaded audio files , in a specific language user chosses to. 
        Upload your audio file in WAV or MP3 format and get a concise summary of its content.
    """)

audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)  # Save the uploaded file and get the path
    st.audio(audio_path)

    # Get supported languages and create a mapping of language names to codes
    supported_langs = tts_langs()
    # Create a sorted list of (code, name) tuples
    sorted_langs = sorted(supported_langs.items(), key=lambda x: x[1])
    # Create a dictionary for display names to codes
    lang_display_to_code = {name: code for code, name in sorted_langs}

    # Language selection
    selected_lang_name = st.selectbox(
        "Language to translate to:",
        options=list(lang_display_to_code.keys()),
        index=None,
        placeholder="Select language...",
    )

    if st.button('Summarize Audio'):
        with st.spinner('Summarizing...'):
            summary_text = summarize_audio(audio_path,selected_lang_name)
            st.info(summary_text)      