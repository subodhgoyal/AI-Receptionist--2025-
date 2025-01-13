import streamlit as st
import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.llm_service import LLMService, generate_session_id

# Initialize LLM Service
@st.cache_resource
def get_llm_service():
    return LLMService()

# Get or create LLM service instance
llm_service = get_llm_service()

# App Title
st.title("Business AI Assistant Chat Interface")

# Initialize Session State for Chat
if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset Session Button
if st.button("Reset Conversation"):
    response = llm_service.reset_session(st.session_state.session_id)
    st.session_state.messages = []
    st.success(response)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Section
if user_input := st.chat_input("Ask your question here..."):
    # Display User's Message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate Assistant's Response
    with st.spinner("Assistant is typing..."):
        response = llm_service.handle_chat(user_input, st.session_state.session_id)

    # Display Assistant's Response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})