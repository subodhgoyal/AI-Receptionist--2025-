import streamlit as st
import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.llm_service import handle_chat, reset_session, generate_new_session_id

# App Title
st.title("Business AI Assistant Chat Interface")

# Initialize Session State for Chat
if "session_id" not in st.session_state:
    st.session_state.session_id = generate_new_session_id()  # Generate unique session ID
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store conversation messages

# Reset Session Button
if st.button("Reset Conversation"):
    reset_session(st.session_state.session_id)
    st.session_state.messages = []  # Clear the messages
    st.success("Conversation has been reset.")

# Display Chat History in a Scrollable Chat Container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Use chat_message container
        st.markdown(message["content"])

# User Input Section
if user_input := st.chat_input("Ask your question here..."):
    # Display User's Message in the Chat Container
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})  # Add to chat history

    # Generate Assistant's Response
    with st.spinner("Assistant is typing..."):
        response = handle_chat(user_input, st.session_state.session_id)

    # Display Assistant's Response in the Chat Container
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})  # Add to chat history
