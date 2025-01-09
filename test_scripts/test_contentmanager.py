import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.content_manager import generate_faq_responses

generate_faq_responses()

from src.content_manager import modify_chunk

modify_chunk("Business Name", "Clarnmain Dental Centre")

from src.content_manager import create_embeddings_for_faq_responses

create_embeddings_for_faq_responses()
