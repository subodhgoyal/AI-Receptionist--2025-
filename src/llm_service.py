import json
from datetime import datetime
from typing import List, Optional
import pytz
from openai import OpenAI
import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.conversation_flows import ConversationManager

CHAT_HISTORY_DIR = "data/chat_sessions"

# Initialize OpenAI client
client = OpenAI()

def get_system_message():
    """Define core system message with constraints"""
    return """You are an intelligent, friendly, and professional AI receptionist for our business. 
    
    IMPORTANT CONSTRAINTS:
    1. ONLY answer questions directly related to our business services, appointments, hours, locations, and policies
    2. DO NOT provide any health, medical, or treatment advice whatsoever
    3. DO NOT answer questions about weather, news, general knowledge, or any topics unrelated to our business
    4. For out-of-scope questions, politely explain that you can only assist with business-related matters
    
    Remember to be helpful, clear, and concise while keeping the interaction professional and friendly."""

class LLMService:
    def __init__(self):
        self.conversation_manager = ConversationManager()
        os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    
    def load_session_history(self, session_id: str) -> List[dict]:
        """Load chat history for a session"""
        session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file, "r", encoding="utf-8") as file:
                return json.load(file)
        return []
    
    def save_session_history(self, session_id: str, history: List[dict]):
        """Save chat history for a session"""
        session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
        with open(session_file, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)
    
    def generate_llm_response(self, 
                              user_input: str, 
                              conversation_result: dict,
                              session_history: List[dict],
                              current_time: Optional[datetime] = None) -> str:
        """Generate response using LLM with conversation context"""
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
            
        # Prepare conversation history
        history = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
            for msg in session_history[-5:]  # Keep last 5 messages for context
        ])
        
        prompt = f"""
        {get_system_message()}

        Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
        Detected Intent: {conversation_result['intent']}
        Confidence: {conversation_result['confidence']}
        
        Previous Conversation:
        {history}

        Current User Input: {user_input}
        
        Rule-Based Response: {conversation_result['response']}
        
        Please provide a natural, conversational response that incorporates the context and rule-based response while following the system constraints.
        If the rule-based response is appropriate, you can enhance it. If it needs modification, please adjust it while maintaining the same intent.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def handle_chat(self, user_input: str, session_id: str) -> str:
        """Main chat handling function"""
        # Load session history
        session_history = self.load_session_history(session_id)
        
        # Process through conversation manager
        conversation_result = self.conversation_manager.process_message(user_input)
        
        # Generate enhanced response using LLM
        final_response = self.generate_llm_response(
            user_input,
            conversation_result,
            session_history
        )
        
        # Update session history
        session_history.append({
            "user": user_input,
            "assistant": final_response,
            "intent": conversation_result["intent"],
            "timestamp": datetime.now(pytz.UTC).isoformat()
        })
        
        # Save updated history
        self.save_session_history(session_id, session_history)
        
        return final_response
    
    def reset_session(self, session_id: str) -> str:
        """Reset a conversation session"""
        session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file, "w", encoding="utf-8") as file:
                json.dump([], file, indent=4)
        return "Session reset. How can I assist you today?"

    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
