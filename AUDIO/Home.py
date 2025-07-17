import streamlit as st

# Set page config once at the beginning
st.set_page_config(
    page_title="Transligo App",
    page_icon="🎧",
    layout="wide"
)

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
    3. **Podcast To Blog**: Generate well crafted, creative and engaging Blog post from podcasts.         
    4. **Audio Transcription (Upload)**: Transcribe audio files by uploading them to the app.
    5. **Chat With Audio File**: Upload an audio file and chat with its transcribed content.
    6. **Language Specific Summarizer**: Summarize your audio files in any specific language.          
    7. **Real-Time Audio Transcription**: Transcribe audio in real-time as it is being recorded or played.
    8. **Voice Cover Generator**: Generate voice covers for various characters from your text script.
    9. **Text To Music**: Generate/Compose piano covers from your text input.       
    """)

    # Footer
    st.write("---")
    st.write("Made with ❤️ by Amritanshu Bhardwaj")
    st.write("© 2025 Transligo. All rights reserved.")

def multilanguage_translator():
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
    
def podcast_summarizer():
    import tempfile
    import os
    from google import genai

    from dotenv import load_dotenv

    load_dotenv()

    # Configure Google API for audio summarization
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def summarize_audio(audio_file_path):
        myfile = client.files.upload(file=audio_file_path)
        response = client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = ["Please give detailed summary notes(pointwise) of the following audio, without missing anything important.",
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
    st.title('Audio Summarization App')

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

def podcast_transcriber():
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

def chat_with_audio():
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
    
def real_time_transcription():
    from audio_recorder_streamlit import audio_recorder
    from faster_whisper import WhisperModel
    import os
    import datetime

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

def voice_cover():
    import os
    import json
    from pathlib import Path
    from dotenv import load_dotenv
    from utils.elevenlabs_api import ElevenLabsAPI
    from utils.script_parser import ScriptParser

    # Load environment variables
    load_dotenv()

    # Initialize API and parser
    api = ElevenLabsAPI()
    parser = ScriptParser()

    def load_voices():
        """Load or fetch available voices"""
        if not os.path.exists("voices.json"):
            st.info("Fetching available voices...")
            voices = api.save_voices_to_json()
            st.success("Voices fetched and saved!")
        else:
            with open("voices.json", 'r') as f:
                voices = json.load(f)
        return voices

    def list_audio_files():
        """List all generated audio files"""
        output_dir = Path("output")
        if output_dir.exists():
            return sorted(list(output_dir.glob("*.mp3")))
        return []

    def delete_audio_file(file_path):
        """Delete an audio file"""
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            st.error(f"Error deleting file: {str(e)}")
            return False

    def main():
        st.title("Voiceover Generator")
        
        # Sidebar for voice assignments
        st.sidebar.title("Voice Settings")
        
        # Load available voices
        voices = load_voices()
        voice_options = {voice["name"]: voice["voice_id"] for voice in voices}
        
        with st.expander("📄 Click to see example script format"):
            st.markdown("""
            Please upload your script using the following format:
            
            ```xml
            <character>John</character>
            <dialogue>Hello there! How are you doing today?</dialogue>

            <character>Sarah</character>
            <dialogue>I'm doing great, thanks for asking! How about you?</dialogue>

            <character>John</character>
            <dialogue>I'm fantastic! The weather is perfect for a walk in the park.</dialogue>

            <character>Sarah</character>
            <dialogue>You're right! We should definitely enjoy this beautiful day.</dialogue>
            ```
            """)
            
        # File upload
        uploaded_file = st.file_uploader("Upload your script file", type=['txt'])
        
        if uploaded_file:
            content = uploaded_file.getvalue().decode()
            
            # Parse the script
            script_lines = parser.parse_script(content)
            
            # Get unique characters
            characters = list(set(char for char, _ in script_lines))
            
            # Voice assignment
            st.sidebar.subheader("Assign Voices to Characters")
            character_voices = {}
            for character in characters:
                voice_id = st.sidebar.selectbox(
                    f"Voice for {character}",
                    options=list(voice_options.keys()),
                    key=character
                )
                character_voices[character] = voice_options[voice_id]
            
            # Generate button
            if st.button("Generate Voiceovers"):
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (character, text) in enumerate(script_lines):
                    status_text.text(f"Generating audio for {character}...")
                    output_file = output_dir / f"{idx+1:03d}_{character}.mp3"
                    
                    try:
                        api.generate_audio(
                            text=text,
                            voice_id=character_voices[character],
                            output_file=str(output_file)
                        )
                        progress_bar.progress((idx + 1) / len(script_lines))
                    except Exception as e:
                        st.error(f"Error generating audio for {character}: {str(e)}")
                        return
                
                status_text.text("All audio files generated successfully!")
                st.success(f"Generated {len(script_lines)} audio files in the 'output' directory")

        # Display generated audio files
        st.subheader("Generated Audio Files")
        audio_files = list_audio_files()
        
        if not audio_files:
            st.info("No audio files generated yet.")
        else:
            for audio_file in audio_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.audio(str(audio_file))
                
                with col2:
                    with open(audio_file, 'rb') as f:
                        st.download_button(
                            label="Download",
                            data=f,
                            file_name=audio_file.name,
                            mime="audio/mpeg",
                            key=f"download_{audio_file.name}"
                        )
                
                with col3:
                    if st.button("Delete", key=f"delete_{audio_file.name}"):
                        if delete_audio_file(audio_file):
                            st.success(f"Deleted {audio_file.name}")
                            st.rerun()

    if __name__ == "__main__":
        main()

def text_to_music():
    from midiutil import MIDIFile
    import io
    import random
    import hashlib

    # Expanded musical configuration with more moods
    MUSIC_PROFILES = {
        'Happy': {
            'scale': [60, 62, 64, 65, 67, 69, 71],  # C Major
            'chords': [
                [60, 64, 67],  # C
                [62, 65, 69],  # Dm
                [67, 71, 74],  # G
                [69, 72, 76],  # Am
                [60, 64, 67, 70],  # Cmaj7
                [65, 69, 72]   # F
            ],
            'tempo_range': (100, 160),
            'patterns': [[0,2,4,2], [0,4,7,4], [0,1,2,3,2,1]]
        },
        'Sad': {
            'scale': [60, 62, 63, 65, 67, 68, 70],  # A Minor
            'chords': [
                [60, 63, 67],  # Am
                [62, 65, 68],  # Bdim
                [65, 68, 72],  # Dm
                [67, 70, 74],  # Em
                [60, 63, 67, 70],  # Am7
                [63, 67, 70]   # C
            ],
            'tempo_range': (60, 100),
            'patterns': [[0,3,5,3], [0,1,0,2], [6,4,2,0]]
        },
        'Epic': {
            'scale': [60, 62, 64, 66, 67, 69, 71],  # C Lydian
            'chords': [
                [60, 64, 67],  # C
                [62, 66, 69],  # D
                [64, 67, 71],  # Em
                [66, 69, 72],  # F#
                [60, 64, 67, 70],  # Cmaj7
                [62, 66, 69, 72]   # Dmaj7
            ],
            'tempo_range': (80, 140),
            'patterns': [[0,4,6,4], [0,2,4,6,4,2], [0,3,6,3]]
        },
        'Mysterious': {
            'scale': [60, 61, 63, 65, 66, 68, 70],  # C Harmonic Minor
            'chords': [
                [60, 63, 67],  # Cm
                [61, 65, 68],  # Dbmaj7
                [63, 66, 70],  # Eb+
                [68, 72, 75],  # Fm
                [60, 63, 66, 70],  # Cdim7
                [65, 68, 72]   # Fm
            ],
            'tempo_range': (70, 110),
            'patterns': [[0,1,3,1], [0,3,6,3], [6,3,0,3]]
        },
        'Energetic': {
            'scale': [60, 62, 64, 65, 67, 69],  # C Pentatonic
            'chords': [
                [60, 64, 67],  # C
                [62, 65, 69],  # Dm
                [65, 69, 72],  # F
                [60, 64, 67, 70],  # Cmaj7
                [62, 65, 69, 72]   # Dm7
            ],
            'tempo_range': (120, 180),
            'patterns': [[0,2,4,2], [0,4,0,4], [0,2,4,5,4,2]]
        }
    }

    def generate_melody(profile, text, duration_beats, selected_chords):
        """Generate melody strictly based on text input"""
        melody = []
        time = 0
        
        # Create unique seed from text
        text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        random.seed(text_hash)
        
        # Generate note choices based on text characters
        char_notes = [(ord(c) % len(profile['scale'])) for c in text if c.isalpha()]
        if not char_notes:
            char_notes = [0, 2, 4]  # Default if no letters
        
        # Create rhythm pattern from text punctuation and spaces
        rhythm_map = {'.': 1.0, '!': 0.75, '?': 0.5, ' ': 0.25}
        rhythm_pattern = [rhythm_map.get(c, 0.33) for c in text if c in rhythm_map or c.isspace()]
        if not rhythm_pattern:
            rhythm_pattern = [0.5]  # Default rhythm
        
        # Generate melody notes strictly from text
        chord_notes = set(note for chord in selected_chords for note in chord)
        scale_notes = profile['scale']
        note_index = 0
        
        while time < duration_beats:
            # Get next note from text-derived sequence
            char_note = char_notes[note_index % len(char_notes)]
            note = scale_notes[char_note % len(scale_notes)]
            
            # Ensure note fits with current chords
            if note not in chord_notes:
                note = random.choice([n for n in scale_notes if n in chord_notes] or scale_notes)
            
            # Get duration from text-derived rhythm
            duration = rhythm_pattern[note_index % len(rhythm_pattern)]
            
            melody.append((note, time, duration))
            time += duration
            note_index += 1
        
        return melody

    def generate_chords(profile, text, duration_beats):
        """Generate chord progression based on text"""
        # Remove text hash seeding for true randomization
        chords = []
        time = 0
        progression = random.sample(profile['chords'], min(4, len(profile['chords'])))
        
        while time < duration_beats:
            for chord in progression:
                if time >= duration_beats:
                    break
                duration = random.choice([1, 2, 4])
                chords.append((chord, time, duration))
                time += duration
        
        return chords

    def create_midi(melody, chords, tempo, output_file):
        midi = MIDIFile(2)
        track, channel = 0, 0
        midi.addTempo(track, 0, tempo)
        
        # Add melody
        for note, time, duration in melody:
            midi.addNote(track, channel, note, time, duration, 100)
        
        # Add chords
        track = 1
        for chord, time, duration in chords:
            for i, pitch in enumerate(chord):
                velocity = 80 - (i * 10)  # Root note louder
                midi.addNote(track, channel, pitch, time, duration, velocity)
        
        midi.writeFile(output_file)

    # Streamlit UI
    st.title("🎹 Mood-Based Piano Composer")

    # User inputs
    text = st.text_area("Enter your text:", "A beautiful day full of possibilities")
    selected_mood = st.selectbox("Select Mood:", list(MUSIC_PROFILES.keys()))

    # Display chord selection for the chosen mood
    profile = MUSIC_PROFILES[selected_mood]
    chord_names = [f"Chord {i+1}" for i in range(len(profile['chords']))]
    selected_chord_indices = st.multiselect(
        f"Select chords for {selected_mood} mood:",
        options=range(len(profile['chords'])),
        format_func=lambda x: f"{chord_names[x]} ({profile['chords'][x]})",
        default=[0, 1, 2]  # Default first 3 chords
    )

    custom_tempo = st.slider("Tempo (BPM)", 
                            profile['tempo_range'][0],
                            profile['tempo_range'][1], 
                            (profile['tempo_range'][0] + profile['tempo_range'][1])//2)
    duration = st.slider("Duration (seconds)", 10, 120, 30)

    if st.button("Generate Music"):
        if not text.strip():
            st.error("Please enter some text!")
        elif not selected_chord_indices:
            st.error("Please select at least one chord!")
        else:
            selected_chords = [profile['chords'][i] for i in selected_chord_indices]
            duration_beats = duration * (custom_tempo/60)
            
            melody = generate_melody(profile, text, duration_beats, selected_chords)
            chords = []
            time = 0
            while time < duration_beats:
                for chord in selected_chords:
                    if time >= duration_beats:
                        break
                    duration = random.choice([1, 2, 4])
                    chords.append((chord, time, duration))
                    time += duration
            
            midi_bytes = io.BytesIO()
            create_midi(melody, chords, custom_tempo, midi_bytes)
            midi_bytes.seek(0)
            
            st.download_button(
                label="Download MIDI",
                data=midi_bytes,
                file_name=f"{selected_mood.lower()}_composition.mid",
                mime="audio/midi"
            )

def language_specific_summarizer():
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

def podcast():
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

# Main App
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app", ["Home Page", "Multilanguage Translator", "Podcast Summarizer", "Podcast To Blog Generator", "Podcast Transcriber", "Language Specific Summarizer", "Chat With Audio", "Real Time Transcription", "Voice Cover Generator", "Text To Music"])
    
    if app_mode == "Home Page":
        front()
    elif app_mode == "Multilanguage Translator":
        multilanguage_translator()
    elif app_mode == "Podcast Summarizer":
        podcast_summarizer()
    elif app_mode == "Podcast To Blog Generator":
        podcast()    
    elif app_mode == "Podcast Transcriber":
        podcast_transcriber()
    elif app_mode == "Language Specific Summarizer":
        language_specific_summarizer()   
    elif app_mode == "Chat With Audio":
        chat_with_audio()
    elif app_mode == "Real Time Transcription":
        real_time_transcription()
    elif app_mode == "Voice Cover Generator":
        voice_cover()
    elif app_mode == "Text To Music":
        text_to_music()        

if __name__ == "__main__":
    main()

