import os
import json
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import random

# File paths
EMBEDDINGS_FILE = "data/faq_responses_embeddings.pt"
CHAT_HISTORY_DIR = "data/chat_sessions"

# Initialize OpenAI client and embedding model
client = OpenAI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class ConversationalEnhancement:
    """Helper class for managing conversational elements"""
    
    @staticmethod
    def get_greeting_phrase():
        greetings = [
            "Thank you for reaching out to us! ",
            "I'm happy to help you today! ",
            "Thanks for getting in touch! ",
            "Welcome! I'm here to assist you. "
        ]
        return random.choice(greetings)
    
    @staticmethod
    def get_confusion_response():
        responses = [
            "I want to make sure I give you accurate information. Could you please provide more details about what you're looking for?",
            "I'd love to help you with that. Could you elaborate a bit more so I can better understand your needs?",
            "To better assist you, could you share a bit more information about what you're specifically interested in?",
            "I want to ensure I address your question fully. Could you rephrase that or provide additional context?"
        ]
        return random.choice(responses)
    
    @staticmethod
    def get_closing_phrase():
        closings = [
            "Is there anything else I can help you with?",
            "Please let me know if you have any other questions!",
            "Don't hesitate to ask if you need any clarification.",
            "I'm here if you need any additional assistance!"
        ]
        return random.choice(closings)

def load_embeddings():
    """Load embeddings and their associated texts from the embeddings file."""
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_FILE}.")
    embedding_data = torch.load(EMBEDDINGS_FILE, map_location=torch.device("cpu"))
    return embedding_data["embeddings"], embedding_data["texts"]

def search_embeddings(query, top_k=3, min_similarity=0.5):
    """Search for the most relevant chunks using embeddings with a similarity threshold."""
    embeddings, texts = load_embeddings()
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    
    # Get both scores and indices for threshold filtering
    results = [(texts[idx], score.item()) for idx, score in enumerate(cos_scores)]
    filtered_results = [text for text, score in results if score >= min_similarity]
    
    return filtered_results[:top_k] if filtered_results else []

def get_system_message():
    """Enhanced system message to encourage more natural conversations"""
    return """You are an intelligent, friendly, and professional AI receptionist for our business. 
    Follow these guidelines for natural conversation:
    
    1. Use a warm, welcoming tone while maintaining professionalism
    2. Show empathy and understanding in your responses
    3. Use natural conversation patterns and appropriate small talk
    4. Handle uncertainty gracefully by asking clarifying questions
    5. Personalize responses based on the context of the conversation
    6. Use "we" and "our" to represent the business
    7. Mirror the user's level of formality while staying professional
    8. Acknowledge user's emotions when appropriate
    
    Remember to be helpful, clear, and super concise while making the interaction feel natural and personal."""

def analyze_user_intent(user_input):
    """Analyze user input for intent and emotion"""
    # Simple keyword-based intent analysis
    intents = {
        'greeting': any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']),
        'urgent': any(word in user_input.lower() for word in ['urgent', 'emergency', 'asap', 'right away']),
        'frustrated': any(word in user_input.lower() for word in ['wrong', 'not working', 'bad', 'unhappy', 'frustrated']),
        'inquiry': '?' in user_input,
    }
    return intents

def generate_response(user_input, session_history, relevant_chunks, max_history=5):
    """Generate a more natural conversational response"""
    truncated_history = session_history[-max_history:]
    history = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in truncated_history])
    
    # Analyze user intent
    intents = analyze_user_intent(user_input)
    
    # Prepare context with relevant chunks
    context = "\n".join(relevant_chunks) if relevant_chunks else "No specific information found."
    
    # Build dynamic prompt based on conversation state
    prompt = f"""
    {get_system_message()}

    Conversation Context:
    {context}

    Previous Conversation:
    {history}

    User Intent Analysis:
    - Greeting: {'Yes' if intents['greeting'] else 'No'}
    - Urgent: {'Yes' if intents['urgent'] else 'No'}
    - Frustrated: {'Yes' if intents['frustrated'] else 'No'}
    - General Inquiry: {'Yes' if intents['inquiry'] else 'No'}

    Current User Input:
    {user_input}

      
    Instructions:
    1. If this is a new conversation, start with a warm greeting
    2. If the user seems frustrated, acknowledge their feelings
    3. If it's an urgent matter, prioritize addressing the urgency
    4. Maintain conversation flow while providing accurate information
    5. Use natural transitions between topics
    6. End with an appropriate closing if the response seems final

    Please provide a natural, conversational response:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def handle_chat(user_input, session_id):
    """Enhanced chat handler with improved conversation flow"""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    
    # Load or initialize session history
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as file:
            session_history = json.load(file)
    else:
        session_history = []

    # Get relevant information
    relevant_chunks = search_embeddings(user_input)
    
    # Generate appropriate response
    if not relevant_chunks:
        # No relevant information found - generate a more natural confusion response
        response = ConversationalEnhancement.get_confusion_response()
    else:
        response = generate_response(user_input, session_history, relevant_chunks)

    # Save interaction
    session_history.append({
        "user": user_input,
        "assistant": response,
        "retrieved_chunks": relevant_chunks,
        "timestamp": datetime.now().isoformat()
    })

    with open(session_file, "w", encoding="utf-8") as file:
        json.dump(session_history, file, indent=4)

    return response

def reset_session(session_id):
    """Reset session with confirmation message"""
    session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4)
        return "I've reset our conversation. How can I assist you today?"
    else:
        os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        with open(session_file, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4)
        return "I've started a new conversation. How can I help you?"

def generate_new_session_id():
    """Generate a unique session ID"""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"