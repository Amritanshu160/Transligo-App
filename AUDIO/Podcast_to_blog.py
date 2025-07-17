import streamlit as st
import tempfile
import os
from google import genai

from dotenv import load_dotenv

load_dotenv()

# Configure Google API for audio summarization
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def podcast_to_blog(audio_file_path):
    myfile = client.files.upload(file=audio_file_path)
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = ["Convert this podcast audio file into a creative, engaging, and well-structured blog post."
                    "The blog should be written in a conversational tone, highlight the key ideas and insights, and feel like a story, not a transcript."
                    "Make it easy to read, fun to follow, and valuable for someone who didn’t listen to the podcast.",
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
st.title('Podcast To Blog Generator App')

with st.expander("About this app"):
    st.write("""
        This app uses Google's generative AI for converting audio files to creative and engaging Blog post.
        Upload your audio file in WAV or MP3 format and get a well crafted blog.
    """)

audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)  # Save the uploaded file and get the path
    st.audio(audio_path)

    if st.button('Convert'):
        with st.spinner('Converting To Blog...'):
            blog_text = podcast_to_blog(audio_path)
            st.info(blog_text)