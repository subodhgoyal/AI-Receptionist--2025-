import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.llm_service import handle_chat, generate_new_session_id

# Start a new session
session_id = generate_new_session_id()

# User input
user_input = "do you offer any marketing services for small and medium businesses?"

# Get AI response
response = handle_chat(user_input, session_id)
print(response)
