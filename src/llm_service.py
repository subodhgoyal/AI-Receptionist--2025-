import os
import json
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# File paths
EMBEDDINGS_FILE = "data/structured_embeddings.pt"
CHAT_HISTORY_DIR = "data/chat_sessions"

# Initialize OpenAI client and embedding model
client = OpenAI()
model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with the actual model used for embeddings

def load_embeddings():
    """
    Load embeddings and their associated texts from the embeddings file.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_FILE}.")
    embedding_data = torch.load(EMBEDDINGS_FILE, map_location=torch.device("cpu"))
    return embedding_data["embeddings"], embedding_data["texts"]

def search_embeddings(query, top_k=3, min_similarity=0.5):
    """
    Search for the most relevant chunks using embeddings with a similarity threshold.
    """
    embeddings, texts = load_embeddings()
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    # Filter results by similarity threshold
    results = [
        texts[idx] for idx, score in enumerate(cos_scores)
        if score >= min_similarity
    ]

    # Return top-k filtered results
    return results[:top_k] if results else ["No relevant information found."]

def get_system_message():
    """
    Load the system message from the file in the data folder.
    If the file doesn't exist, use a default message.
    """
    system_message_file = "data/system_message.txt"

    # Default system message
    default_message = (
        "You are a helpful assistant representing the business. "
        "Always respond in the first person using 'we,' 'our,' or 'I' to reflect that you are part of the business. "
        "Keep your tone friendly and professional. Be concise but include necessary details to address user queries effectively."
    )

    # Load the message from the file if it exists
    if os.path.exists(system_message_file):
        with open(system_message_file, "r", encoding="utf-8") as file:
            return file.read().strip()

    # If the file doesn't exist, return the default message
    return default_message

def generate_response(user_input, session_history, relevant_chunks, max_history=5):
    """
    Generate a conversational response using OpenAI GPT-4 with the system message loaded dynamically.
    """
    # Truncate session history
    truncated_history = session_history[-max_history:]
    history = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in truncated_history])

    # Combine relevant chunks (truncate if necessary)
    context = "\n".join(relevant_chunks[:3])  # Use only the top 3 chunks for context

    # Fetch the dynamic system message from file
    system_message = get_system_message()

    # Build prompt
    prompt = f"""
    {system_message}

    Use the following context to answer the user's query concisely:

    Relevant Information:
    {context}

    Session History:
    {history}

    Current User Input:
    {user_input}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def handle_chat(user_input, session_id):
    """
    Handle a chat session, including retrieving embeddings, generating a response, and saving the session.
    """
    # Ensure chat history folder exists
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

    # Load session history or initialize it
    session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as file:
            session_history = json.load(file)
    else:
        session_history = []

    # Retrieve relevant chunks from embeddings
    relevant_chunks = search_embeddings(user_input)

    # Generate response using OpenAI API
    if "No relevant information found." in relevant_chunks:
        response = "I'm sorry, I couldn't find this information. Can you please rephrase?"
    else:
        response = generate_response(user_input, session_history, relevant_chunks)

    # Append the current interaction to the session history
    session_history.append({
        "user": user_input,
        "assistant": response,
        "retrieved_chunks": relevant_chunks
    })

    # Save the updated session history
    with open(session_file, "w", encoding="utf-8") as file:
        json.dump(session_history, file, indent=4)

    return response

def reset_session(session_id):
    """
    Reset the user context and session history for the LLM while keeping the existing session file.
    """
    session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(session_file):
        # Clear the session history in memory (reset context)
        with open(session_file, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4)
        print(f"Session {session_id} has been reset. Existing file is kept.")
    else:
        print(f"No session file found for {session_id}. Initializing a new session.")
        os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        with open(session_file, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4)

def generate_new_session_id():
    """
    Generate a unique session ID with a timestamp.
    """
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
