import streamlit as st
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load the tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Define a function to perform Speech-to-Text
def speech_to_text(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    input_values = tokenizer(speech, return_tensors='pt').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

# Streamlit UI
st.title("Speech to Text")

# Input section 
audio_file_path = st.text_input("Enter the path of the audio file:", "")

# Convert button
if st.button("Convert"):
    if audio_file_path:
        # Remove quotes from the input path
        audio_file_path = audio_file_path.strip('"')
        try:
            # Call the conversion function and display the result
            transcription = speech_to_text(audio_file_path)
            st.write("Transcription:")
            st.success(transcription)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid audio file path.")