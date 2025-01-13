from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random
from datetime import datetime

@dataclass
class Intent:
    name: str
    keywords: List[str]
    responses: List[str]
    conditions: Optional[List[str]] = None
    next_intents: List[str] = field(default_factory=list)
    
@dataclass
class ConversationFlow:
    intents: Dict[str, Intent]
    fallback_responses: List[str]
    
    def analyze_intent(self, user_input: str) -> tuple[str, float]:
        user_input = user_input.lower()
        
        # Check each intent
        max_score = 0
        detected_intent = "fallback"
        
        for intent_name, intent in self.intents.items():
            score = 0
            # Check keywords
            for keyword in intent.keywords:
                if keyword in user_input:
                    score += 1
            
            # Check conditions if they exist
            if intent.conditions:
                for condition in intent.conditions:
                    if condition.lower() in user_input:
                        score += 0.5
                        
            # Normalize score
            score = score / (len(intent.keywords) + (len(intent.conditions or [])) * 0.5)
            
            if score > max_score:
                max_score = score
                detected_intent = intent_name
        
        return detected_intent, max_score
    
    def get_response(self, intent_name: str) -> str:
        if intent_name == "fallback":
            return random.choice(self.fallback_responses)
        return random.choice(self.intents[intent_name].responses)
    
    def get_next_intents(self, current_intent: str) -> List[str]:
        if current_intent in self.intents:
            return self.intents[current_intent].next_intents
        return []

# Define default conversation flow
default_flow = ConversationFlow(
    intents={
        "greeting": Intent(
            name="greeting",
            keywords=["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            responses=[
                "Hello! How can I assist you today?",
                "Welcome! What can I help you with?",
                "Hi there! How may I help you?"
            ],
            next_intents=["appointment", "business_hours", "location"]
        ),
        "appointment": Intent(
            name="appointment",
            keywords=["appointment", "book", "schedule", "reservation"],
            responses=[
                "I can help you schedule an appointment. What date and time works best for you?",
                "I'd be happy to help you book an appointment. When would you like to come in?"
            ],
            conditions=["urgent", "asap", "emergency"],
            next_intents=["confirm_appointment", "business_hours"]
        ),
        "business_hours": Intent(
            name="business_hours",
            keywords=["hours", "open", "close", "timing", "schedule"],
            responses=[
                "Our business hours are Monday-Friday, 9 AM to 5 PM.",
                "We're open from 9 AM to 5 PM, Monday through Friday."
            ],
            next_intents=["appointment", "location"]
        ),
        "location": Intent(
            name="location",
            keywords=["location", "address", "directions", "where", "place"],
            responses=[
                "We're located at 123 Business Street, Suite 100.",
                "Our address is 123 Business Street, Suite 100. Would you like directions?"
            ],
            next_intents=["business_hours", "appointment"]
        ),
        "farewell": Intent(
            name="farewell",
            keywords=["bye", "goodbye", "thanks", "thank you", "good night"],
            responses=[
                "Thank you for reaching out! Have a great day!",
                "You're welcome! Have a wonderful day!",
                "Goodbye! Feel free to contact us if you need anything else."
            ],
            next_intents=[]
        )
    },
    fallback_responses=[
        "I apologize, but I'm not sure I understood. Could you please rephrase that?",
        "I want to help, but I'm not quite sure what you're asking. Could you provide more details?",
        "I'm having trouble understanding your request. Could you try explaining it differently?"
    ]
)

class ConversationManager:
    def __init__(self, flow: ConversationFlow = default_flow):
        self.flow = flow
        self.confidence_threshold = 0.3
    
    def process_message(self, user_input: str) -> dict:
        # Detect intent
        intent, confidence = self.flow.analyze_intent(user_input)
        
        # Get response based on confidence
        if confidence >= self.confidence_threshold:
            response = self.flow.get_response(intent)
            next_intents = self.flow.get_next_intents(intent)
        else:
            intent = "fallback"
            response = self.flow.get_response("fallback")
            next_intents = []
            
        return {
            "intent": intent,
            "confidence": confidence,
            "response": response,
            "next_intents": next_intents
        }