import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# test_llm_service.py
from src.llm_service import LLMService

def main():
    try:
        # Instantiate the LLMService class
        llm_service = LLMService()
        
        # Check if ConversationManager is loaded properly
        if llm_service.conversation_manager:
            print("ConversationManager loaded successfully!")
        else:
            print("ConversationManager failed to load.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
