import os
import streamlit as st
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