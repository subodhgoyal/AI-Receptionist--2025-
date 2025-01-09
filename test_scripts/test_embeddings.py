import torch
import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


# Path to the saved embeddings file
EMBEDDINGS_FILE = "data/embeddings.pt"

# Load the embeddings
embedding_data = torch.load(EMBEDDINGS_FILE, map_location=torch.device('cpu'))

# Extract embeddings and associated text
embeddings = embedding_data['embeddings']  # Tensor of embeddings
texts = embedding_data['texts']  # List of corresponding text chunks

# Print details
print(f"Number of embeddings: {embeddings.shape[0]}")  # Number of embeddings
print(f"Embedding dimensions: {embeddings.shape[1]}")  # Dimensionality of each embedding
