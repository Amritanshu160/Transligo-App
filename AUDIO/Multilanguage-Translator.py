# main.py
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from groq_translation import groq_translate
from gtts import gTTS
from gtts.lang import tts_langs
from google import genai

from dotenv import load_dotenv

load_dotenv()

# Configure Google API for audio summarization
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page config
st.set_page_config(page_title='Multilanguage Translator', page_icon='🎤')

# Set page title
st.title('Multilanguage Translator')

def speech_to_text(audio_file_path):
    myfile = client.files.upload(file=audio_file_path)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Transcribe the entire audio accurately. Only return the transcribed text, without any additional explanation or metadata. "
            "Ensure correct punctuation is included—such as question marks (?), exclamation marks (!), full stops (.), commas (,), and other necessary symbols—for natural readability and clarity.",
            myfile
        ]
    )
    return response.text.strip()

# Text to speech
def text_to_speech(translated_text, language):
    file_name = "speech.mp3"
    my_obj = gTTS(text=translated_text, lang=language)
    my_obj.save(file_name)
    return file_name

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
# Record audio
audio_bytes = audio_recorder(pause_threshold=86400)
if audio_bytes and selected_lang_name:  # And that option is defined.
    # Display audio player
    st.audio(audio_bytes, format="audio/wav")

    # Save audio to file
    audio_file_path = 'audio.wav'
    with open(audio_file_path, mode='wb') as f:
        f.write(audio_bytes)

    # Speech to text
    st.divider()
    with st.spinner('Transcribing...'):
        text = speech_to_text(audio_file_path) # Use the new function
    st.subheader('Transcribed Text')
    if text:
        st.write(text)
    else:
        st.write("Transcription failed.")

    # Groq translation
    st.divider()
    with st.spinner('Translating...'):
        # Get the language code for the selected language name
        lang_code = lang_display_to_code[selected_lang_name]
        translation = groq_translate(text, lang_code)
    st.subheader('Translated Text to ' + selected_lang_name)
    st.write(translation.text)

    # Text to speech
    audio_file = text_to_speech(translation.text, lang_code)
    st.audio(audio_file, format="audio/mp3")
