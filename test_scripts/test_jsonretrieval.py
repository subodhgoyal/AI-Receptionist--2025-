import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


from src.json_manager import extract_key_data, save_to_json

# Extract key data and save it to JSON
data = extract_key_data()
save_to_json(data)