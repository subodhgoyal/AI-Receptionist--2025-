import os
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

def load_environment_variables():
    """
    Loads environment variables from a .env file.
    """
    load_dotenv()

def load_embeddings(embeddings_file="data/embeddings.pt"):
    """
    Loads embeddings and their associated text chunks from a .pt file.
    Returns the SentenceTransformer model, embeddings tensor, and associated texts.
    """
    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Check if the embeddings file exists
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_file}")

    # Load embeddings and texts
    embedding_data = torch.load(embeddings_file, map_location=torch.device('cpu'))
    embeddings = embedding_data['embeddings']
    texts = embedding_data['texts']

    return model, embeddings, texts
