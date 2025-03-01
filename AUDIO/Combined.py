# main.py
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import os
from groq_translation import groq_translate
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title='Transligo App', page_icon='🎤')

# Load whisper model
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2))

# Configure Google API for audio summarization and transcription
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def front():
    import streamlit as st

    # Title and description
    st.title("🎧 Transligo - Your Audio Companion")
    st.write("Welcome to Transligo! Explore the following tools to simplify your audio processing and translation tasks.")

    # App descriptions
    st.subheader("Available Features")
    st.write("""
    1. **Multilanguage Translator**: Translate audio content into multiple languages seamlessly.
    2. **Podcast Summarizer**: Generate concise summaries of podcast episodes for quick insights.
    3. **Audio Transcription (Upload)**: Transcribe audio files by uploading them to the app.
    4. **Real-Time Audio Transcription**: Transcribe audio in real-time as it is being recorded or played.
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.sidebar.write("Select a feature from the list above to get started.")

    # Footer
    st.write("---")
    st.write("Made with ❤️ by Amritanshu Bhardwaj")
    st.write("© 2025 Transligo. All rights reserved.")

# Speech to text
def speech_to_text(audio_chunk):
    segments, info = model.transcribe(audio_chunk, beam_size=5)
    speech_text = " ".join([segment.text for segment in segments])
    return speech_text

# Text to speech
def text_to_speech(translated_text, language):
    file_name = "speech.mp3"
    my_obj = gTTS(text=translated_text, lang=language)
    my_obj.save(file_name)
    return file_name

# Summarize audio
def summarize_audio(audio_file_path):
    """Summarize the audio using Google's Generative API."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Please summarize the following audio.",
            audio_file
        ]
    )
    return response.text

# Transcribe audio
def transcribe_audio(audio_file_path):
    """Transcribe the audio using Google's Generative API."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Please transcribe the following audio.",
            audio_file
        ]
    )
    return response.text

# Save uploaded file
def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Initialize session state
if "transcription" not in st.session_state:
    st.session_state["transcription"] = ""

# Language selection
languages = {
   "Portuguese": "pt",
   "Spanish": "es",
   "German": "de",
   "French": "fr",
   "Italian": "it",
   "Dutch": "nl",
   "Russian": "ru",
   "Japanese": "ja",
   "Chinese": "zh",
   "Korean": "ko"
}

# Sidebar for app selection
app_choice = st.sidebar.selectbox(
    "Choose an App",
    ("Home Page","Multilanguage Translator", "Podcast Summarizer", "Audio Transcription", "Real-Time Transcription")
)

if app_choice == "Home Page":
    front()

elif app_choice == "Multilanguage Translator":
    st.header("Multilanguage Translator")
    option = st.selectbox(
       "Language to translate to:",
       languages,
       index=None,
       placeholder="Select language...",
    )

    # Record audio
    audio_bytes = audio_recorder()
    if audio_bytes and option:
        # Display audio player
        st.audio(audio_bytes, format="audio/wav")

        # Save audio to file
        with open('audio.wav', mode='wb') as f:
            f.write(audio_bytes)

        # Speech to text
        st.divider()
        with st.spinner('Transcribing...'):
            text = speech_to_text('audio.wav')
        st.subheader('Transcribed Text')
        st.write(text)

        # Groq translation
        st.divider()
        with st.spinner('Translating...'):
            translation = groq_translate(text, 'en', option)
        st.subheader('Translated Text to ' + option)
        st.write(translation.text)

        # Text to speech
        audio_file = text_to_speech(translation.text, languages[option])
        st.audio(audio_file, format="audio/mp3")

elif app_choice == "Podcast Summarizer":
    st.header("Podcast Summarizer")
    with st.expander("About this app"):
        st.write("""
            This app uses Google's generative AI to summarize audio files. 
            Upload your audio file in WAV or MP3 format and get a concise summary of its content.
        """)

    audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
    if audio_file is not None:
        audio_path = save_uploaded_file(audio_file)  # Save the uploaded file and get the path
        st.audio(audio_path)

        if st.button('Summarize Audio'):
            with st.spinner('Summarizing...'):
                summary_text = summarize_audio(audio_path)
                st.info(summary_text)

elif app_choice == "Audio Transcription":
    st.header("Audio Transcription")
    with st.expander("About this app"):
        st.write("""
            This app uses Google's generative AI to summarize audio files. 
            Upload your audio file in WAV or MP3 format and get a transcription of its content.
        """)

    audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
    if audio_file is not None:
        audio_path = save_uploaded_file(audio_file)  # Save the uploaded file and get the path
        st.audio(audio_path)

        if st.button('Transcribe Audio'):
            with st.spinner('Transcribing Audio...'):
                summary_text = transcribe_audio(audio_path)
                st.info(summary_text)

elif app_choice == "Real-Time Transcription":
    st.header("Real-Time Transcription")
    st.divider()
    st.subheader("🎙️ Record Your Audio")
    audio_bytes = audio_recorder()

    if audio_bytes:
        # Save audio to file
        audio_file_path = "recorded_audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(audio_bytes)

        # Display audio player
        st.audio(audio_file_path, format="audio/wav")

        # Transcribe audio
        st.divider()
        with st.spinner("Transcribing..."):
            transcription_text = speech_to_text(audio_file_path)
            st.session_state["transcription"] = transcription_text

        # Display transcription
        st.subheader("📝 Transcribed Text")
        st.text_area("Transcription", value=st.session_state["transcription"], height=200)

        # Download transcription
        if st.session_state["transcription"]:
            filename = f"transcription_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            st.download_button(
                label="📥 Download Transcription",
                data=st.session_state["transcription"].encode("utf-8"),
                file_name=filename,
                mime="text/plain",
            )

