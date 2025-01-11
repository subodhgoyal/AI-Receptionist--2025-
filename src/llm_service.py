import os
import json
import torch
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import random
import pytz

# File paths
EMBEDDINGS_FILE = "data/faq_responses_embeddings.pt"
CHAT_HISTORY_DIR = "data/chat_sessions"

# Initialize OpenAI client and embedding model
client = OpenAI()
model = SentenceTransformer("all-MiniLM-L6-v2")

TIMEZONE = 'America/New_York'  # Change to your business timezone

class ConversationalEnhancement:
    """Helper class for managing conversational elements"""
    
    @staticmethod
    def get_greeting_phrase(current_time):
        hour = current_time.hour
        time_of_day = (
            "morning" if 5 <= hour < 12
            else "afternoon" if 12 <= hour < 17
            else "evening"
        )
        
        greetings = [
            f"Good {time_of_day}! Thank you for reaching out to us. ",
            f"Good {time_of_day}! I'm happy to help you today. ",
            f"Good {time_of_day}! Thanks for getting in touch. ",
            f"Good {time_of_day}! Welcome to our service. "
        ]
        return random.choice(greetings)
    
    @staticmethod
    def get_uncertainty_response():
        responses = [
            "I apologize, but I'm not entirely sure about this specific detail. Would you like me to have someone from our team contact you about this? Meanwhile, is there something else I can help you with?",
            "I want to ensure you get the most accurate information. Let me have our team contact you about this specifically. Meanwhile, is there anything else I can assist you with?",
        ]
        return random.choice(responses)
    
    @staticmethod
    def get_farewell_response():
        responses = [
            "You're welcome! Have a great rest of your day!",
            "Thank you for reaching out. Have a wonderful day!",
            "It was my pleasure helping you. Take care!",
            "Glad I could help. Have a great day!"
        ]
        return random.choice(responses)
    
    @staticmethod
    def get_out_of_scope_response():
        responses = [
            "I apologize, but I can only assist with questions related to our business services, appointments, and general information. For this specific query, I recommend consulting appropriate resources or professionals. Is there anything about our services I can help you with?",
            "I'm specifically trained to help with our business services and appointments. For this type of question, I'd recommend reaching out to relevant experts or services. Would you like to know about our services instead?",
            "I need to stay focused on helping you with our business services. This question falls outside my scope - would you like to know about our services or schedule an appointment instead?"
        ]
        return random.choice(responses)

def expand_query_with_synonyms(query):
    """Expand search query with basic synonyms"""
    synonyms = {
        'appointment': ['booking', 'reservation', 'schedule', 'meeting'],
        'cost': ['price', 'fee', 'charge', 'rate'],
        'hours': ['timing', 'schedule', 'open', 'close'],
        'location': ['address', 'place', 'directions', 'where'],
        'parking': ['park', 'garage', 'lot'],
        'thanks': ['thank', 'thanking', 'appreciate', 'grateful'],
        'bye': ['goodbye', 'bye', 'see you', 'talk to you', 'take care']
    }
    
    expanded_query = query
    for word in query.lower().split():
        for key, values in synonyms.items():
            if word in values or word == key:
                expanded_query += f" {key} " + " ".join(values)
    
    return expanded_query

def analyze_user_intent(user_input):
    """Enhanced user intent analysis with topic filtering"""
    user_input_lower = user_input.lower()
    
    # Define out-of-scope topics
    out_of_scope_topics = {
        'weather': ['weather', 'temperature', 'rain', 'sunny', 'forecast'],
        'health_advice': ['symptoms', 'treatment', 'medicine', 'disease', 'pain', 'hurt', 'medical', 'diagnosis'],
        'general_knowledge': ['history', 'science', 'politics', 'news', 'sports']
    }
    
    # Check if query is out of scope
    is_out_of_scope = any(
        any(topic in user_input_lower for topic in topics)
        for topics in out_of_scope_topics.values()
    )
    
    # Detect closing intent with expanded phrases
    closing_phrases = {
        'thanks': ['thank', 'thanks', 'appreciate', 'grateful', 'thx'],
        'goodbye': ['bye', 'goodbye', 'see you', 'talk to you', 'take care'],
        'ok': ['okay', 'ok', 'alright', 'sure', 'got it']
    }
    
    intents = {
        'greeting': any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']),
        'urgent': any(word in user_input_lower for word in ['urgent', 'emergency', 'asap', 'right away']),
        'frustrated': any(word in user_input_lower for word in ['wrong', 'not working', 'bad', 'unhappy', 'frustrated']),
        'inquiry': '?' in user_input,
        'time_related': any(word in user_input_lower for word in ['open', 'close', 'hours', 'today', 'tomorrow', 'time']),
        'appointment': any(word in user_input_lower for word in ['book', 'appointment', 'schedule', 'meet']),
        'closing': any(any(phrase in user_input_lower for phrase in phrases) for phrases in closing_phrases.values()),
        'thanks': any(word in user_input_lower for word in closing_phrases['thanks']),
        'goodbye': any(word in user_input_lower for word in closing_phrases['goodbye']),
        'out_of_scope': is_out_of_scope
    }
    return intents

def load_embeddings():
    """Load embeddings and their associated texts from the embeddings file."""
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_FILE}.")
    embedding_data = torch.load(EMBEDDINGS_FILE, map_location=torch.device("cpu"))
    return embedding_data["embeddings"], embedding_data["texts"]

def search_embeddings(query, top_k=3, min_similarity=0.45):
    """Search for the most relevant chunks using embeddings with expanded query"""
    embeddings, texts = load_embeddings()
    expanded_query = expand_query_with_synonyms(query)
    query_embedding = model.encode(expanded_query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    
    results = [(texts[idx], score.item()) for idx, score in enumerate(cos_scores)]
    filtered_results = [text for text, score in results if score >= min_similarity]
    
    return filtered_results[:top_k] if filtered_results else []

def generate_response(user_input, session_history, relevant_chunks, current_time, max_history=5):
    """Generate a more natural conversational response with time awareness"""
    truncated_history = session_history[-max_history:]
    history = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in truncated_history])
    
    intents = analyze_user_intent(user_input)
    context = "\n".join(relevant_chunks) if relevant_chunks else ""
    
    prompt = f"""
    {get_system_message()}

    Current Time and Date: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}

    Conversation Context:
    {context}

    Previous Conversation:
    {history}

    User Intent Analysis:
    - Greeting: {'Yes' if intents['greeting'] else 'No'}
    - Urgent: {'Yes' if intents['urgent'] else 'No'}
    - Frustrated: {'Yes' if intents['frustrated'] else 'No'}
    - Time Related: {'Yes' if intents['time_related'] else 'No'}
    - Appointment Related: {'Yes' if intents['appointment'] else 'No'}
    - Closing Conversation: {'Yes' if intents['closing'] else 'No'}
    - Expressing Thanks: {'Yes' if intents['thanks'] else 'No'}
    - Saying Goodbye: {'Yes' if intents['goodbye'] else 'No'}

    Current User Input:
    {user_input}
    
    Instructions:
    1. If this is a new conversation, start with a warm greeting
    2. If the user seems frustrated, acknowledge their feelings
    3. If it's an urgent matter, prioritize addressing it
    4. For medical emergencies, direct users to call emergency services
    5. If you're uncertain about business matters, offer to have the team contact them
    6. If the user is closing the conversation, respond naturally without asking follow-up questions
    7. Maintain natural conversation flow
    8. End with an appropriate closing unless user is ending the conversation

    Please provide a natural, conversational response:
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def handle_chat(user_input, session_id, current_time=None):
    """Enhanced chat handler with improved intent detection"""
    if current_time is None:
        current_time = datetime.now(pytz.UTC)
    
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as file:
            session_history = json.load(file)
    else:
        session_history = []

    intents = analyze_user_intent(user_input)
    
    if intents['out_of_scope']:
        response = ConversationalEnhancement.get_out_of_scope_response()
    elif intents['closing'] and len(user_input.split()) <= 4:  # Short closing phrases
        response = ConversationalEnhancement.get_farewell_response()
    else:
        relevant_chunks = search_embeddings(user_input)
        response = generate_response(user_input, session_history, relevant_chunks, current_time)

    session_history.append({
        "user": user_input,
        "assistant": response,
        "timestamp": current_time.isoformat()
    })

    with open(session_file, "w", encoding="utf-8") as file:
        json.dump(session_history, file, indent=4)

    return response

def get_system_message():
    """Enhanced system message with strict topic boundaries"""
    return """You are an intelligent, friendly, and professional AI receptionist for our business. 
    
    IMPORTANT CONSTRAINTS:
    1. ONLY answer questions directly related to our business services, appointments, hours, locations, and policies
    2. DO NOT provide any health, medical, or treatment advice whatsoever
    3. DO NOT answer questions about weather, news, general knowledge, or any topics unrelated to our business
    4. For out-of-scope questions, politely explain that you can only assist with business-related matters
    
    Follow these guidelines for natural conversation:
    1. Use a warm, welcoming tone while maintaining professionalism
    2. Show empathy and understanding in your responses
    3. Use natural conversation patterns
    4. If uncertain about business matters, offer to have the team contact the customer
    5. Personalize responses based on conversation context
    6. Use "we" and "our" to represent the business
    7. Mirror the user's level of formality while staying professional
    8. For medical emergencies, direct users to call emergency services
    9. When users are closing the conversation, respond naturally without asking follow-up questions
    
    Remember to be helpful, clear, and concise while making the interaction feel natural and personal."""

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