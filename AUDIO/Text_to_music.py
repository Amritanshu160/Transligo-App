import streamlit as st
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



