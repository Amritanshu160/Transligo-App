import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import os
import datetime

# Set page config
st.set_page_config(page_title="Real-Time Transcription", page_icon="🎤")

# Set page title
st.title("Real-Time Transcription App")

st.info("To remove the previous transcriptions completely, kindly refresh the app.")

# Load Faster Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2))

# Function to transcribe speech
def speech_to_text(audio_file):
    segments, _ = model.transcribe(audio_file, beam_size=5)
    return " ".join([segment.text for segment in segments])

# Initialize session state
if "transcription" not in st.session_state:
    st.session_state["transcription"] = ""

# Record audio
st.divider()
st.subheader("🎙️ Record Your Audio")

# Audio recorder with continuous option
audio_bytes = audio_recorder(pause_threshold=86000)  # 24-hour pause threshold for continuous recording

if audio_bytes:
    # Save audio to file
    audio_file_path = "recorded_audio.wav"
    with open(audio_file_path, "wb") as f:
        f.write(audio_bytes)

    # Transcribe audio
    st.divider()
    with st.spinner("Transcribing..."):
        new_transcription = speech_to_text(audio_file_path)
        st.session_state["transcription"] += " " + "\n\n" + new_transcription

    # Display transcription
    st.subheader("📝 Transcribed Text")
    st.text_area("Transcription", value=st.session_state["transcription"].strip(), height=200)

# Download transcription
if st.session_state["transcription"]:
    filename = f"transcription_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button(
        label="📥 Download Transcription",
        data=st.session_state["transcription"].encode("utf-8"),
        file_name=filename,
        mime="text/plain",
    )






