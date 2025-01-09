import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# File paths
EMBEDDINGS_FILE = "data/embeddings.pt"
CHUNKS_FILE = "data/structured_chunks.json"

# Fields to extract and organize into structured chunks
FIELDS = [
    "Business Name",
    "Phone Number",
    "Email",
    "Address/Location",
    "Business Hours",
    "Products & Services Overview",
    "Team Overview",
    "How to Schedule an Appointment"
]

# Initialize OpenAI client and embedding model
client = OpenAI()
model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with the actual model used

def load_embeddings():
    """
    Load embeddings and their associated texts from the embeddings file.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_FILE}.")
    embedding_data = torch.load(EMBEDDINGS_FILE, map_location=torch.device("cpu"))
    return embedding_data["embeddings"], embedding_data["texts"]

def search_embeddings(query, top_k=5):
    """
    Search for the most relevant chunks using embeddings.
    """
    embeddings, texts = load_embeddings()

    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    # Retrieve the top-k relevant chunks
    results = [texts[idx] for idx in top_results.indices]
    return results

def generate_structured_chunks():
    """
    Generate `structured_chunks.json` by using embeddings search and OpenAI API for refinement.
    """
    structured_chunks = []

    for field in FIELDS:
        # Step 1: Search embeddings for the top-k most relevant chunks
        top_chunks = search_embeddings(field, top_k=5)

        # Step 2: Use OpenAI API to refine the chunk content
        prompt = f"""
        Based on the following relevant information, extract concise and specific details for the field "{field}".

        Relevant Information:
        {"\n".join(top_chunks)}
        
        Provide the result as plain text.
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in extracting structured business information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        refined_content = response.choices[0].message.content.strip()

        # Step 3: Append the field and refined content to the structured chunks
        structured_chunks.append({"header": field, "content": refined_content})

    # Step 4: Save structured chunks to JSON
    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(structured_chunks, file, indent=4)
    print("Structured chunks created and saved successfully!")

def load_chunks():
    """
    Load structured chunks from the JSON file.
    """
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    raise FileNotFoundError(f"Structured chunks file not found at {CHUNKS_FILE}.")

def save_chunks(chunks):
    """
    Save the updated chunks to the JSON file.
    """
    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4)
    print("Chunks saved successfully!")

def modify_chunk(header, new_content):
    """
    Modify the content of a specific chunk by its header.
    """
    chunks = load_chunks()
    for chunk in chunks:
        if chunk["header"].lower() == header.lower():
            chunk["content"] = new_content
            save_chunks(chunks)
            return f"Updated chunk: {header}"
    return f"Header '{header}' not found."

def view_chunks():
    """
    View all chunks as a formatted string.
    """
    chunks = load_chunks()
    for chunk in chunks:
        print(f"{chunk['header']}:\n{chunk['content']}\n{'-' * 40}")

def create_embeddings_for_chunks():
    """
    Generate embeddings for structured chunks (header + content) and save them to a file.
    """
    # Load structured chunks
    chunks = load_chunks()

    # Prepare combined texts (header + content) for embeddings
    combined_texts = [f"{chunk['header']}: {chunk['content']}" for chunk in chunks]

    # Generate embeddings
    embeddings = model.encode(combined_texts, convert_to_tensor=True)

    # Save embeddings and corresponding texts
    embeddings_file = "data/structured_embeddings.pt"
    torch.save({"embeddings": embeddings, "texts": combined_texts}, embeddings_file)
    print(f"Embeddings for structured chunks saved to {embeddings_file}.")

def save_chunks(chunks):
    """
    Save the updated chunks to the JSON file and regenerate embeddings.
    """
    # Save the updated chunks to structured_chunks.json
    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4)
    print("Chunks saved successfully!")

    # Regenerate embeddings for the updated chunks
    create_embeddings_for_chunks()
