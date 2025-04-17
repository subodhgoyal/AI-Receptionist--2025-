import streamlit as st
import sys
import os
from datetime import datetime

# Add the src directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.llm_service import LLMService
from src.voice_interface import VoiceInterface

# Initialize LLM Service
llm_service = LLMService()

# Initialize Voice Interface
@st.cache_resource
def get_voice_interface():
    return VoiceInterface()

voice_interface = get_voice_interface()

# App Title and Description
st.title("AI Receptionist Voice Interface")
st.markdown("""
    This interface allows you to interact with the AI Receptionist using voice commands.
    You can either:
    - Use the voice recording feature to speak with the assistant
    - Type your message in the text input below
""")

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = llm_service.generate_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recording" not in st.session_state:
    st.session_state.recording = False

# Settings and Controls Section
with st.sidebar:
    st.header("Controls")
    # Reset Session Button
    if st.button("Reset Conversation"):
        llm_service.reset_session(st.session_state.session_id)
        st.session_state.messages = []
        st.success("Conversation has been reset.")
    
    # Display Session ID
    st.text(f"Session ID: {st.session_state.session_id}")

# Voice Recording Interface
st.header("Voice Controls")
col1, col2 = st.columns(2)

with col1:
    if st.button(
        "🎤 Start Recording" if not st.session_state.recording else "⏹️ Stop Recording",
        type="primary" if not st.session_state.recording else "secondary"
    ):
        if not st.session_state.recording:
            voice_interface.start_recording()
            st.session_state.recording = True
        else:
            voice_interface.stop_recording()
            st.session_state.recording = False
            
            with st.spinner("Transcribing..."):
                transcribed_text = voice_interface.transcribe_audio()
                if transcribed_text:
                    st.chat_message("user").markdown(transcribed_text)
                    st.session_state.messages.append({"role": "user", "content": transcribed_text})

                    with st.spinner("Generating response..."):
                        response = llm_service.handle_chat(transcribed_text, st.session_state.session_id)
                        st.chat_message("assistant").markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        audio_file = voice_interface.text_to_speech(response)
                        voice_interface.play_audio_response(audio_file)

with col2:
    if st.session_state.recording:
        st.markdown("🔴 Recording in progress...")

# Chat History Display
st.header("Conversation History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Text Input Alternative
if text_input := st.chat_input("Type your message here..."):
    st.chat_message("user").markdown(text_input)
    st.session_state.messages.append({"role": "user", "content": text_input})
    
    with st.spinner("Generating response..."):
        response = llm_service.handle_chat(text_input, st.session_state.session_id)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        audio_file = voice_interface.text_to_speech(response)
        voice_interface.play_audio_response(audio_file)
