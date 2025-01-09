import streamlit as st
import json
import torch
import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from sentence_transformers import SentenceTransformer
from src.webscraping_agent import scrape_website, chunk_text, generate_embeddings
from src.content_manager import generate_structured_chunks, save_chunks, load_chunks

# File paths
EMBEDDINGS_FILE = "data/structured_embeddings.pt"
STRUCTURED_CHUNKS_FILE = "data/structured_chunks.json"
ADDITIONAL_INFO_FILE = "data/additional_info.json"

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with the correct model

# Utility functions
def load_additional_info():
    """Load additional info from the JSON file or initialize an empty list."""
    if os.path.exists(ADDITIONAL_INFO_FILE):
        with open(ADDITIONAL_INFO_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []  # Return an empty list if the file doesn't exist

def save_additional_info(additional_info):
    """Save additional info to the JSON file."""
    os.makedirs(os.path.dirname(ADDITIONAL_INFO_FILE), exist_ok=True)
    with open(ADDITIONAL_INFO_FILE, "w", encoding="utf-8") as file:
        json.dump(additional_info, file, indent=4)

def generate_combined_embeddings():
    """
    Generate embeddings by combining structured chunks and additional info.
    """
    # Load structured chunks and additional info
    structured_chunks = load_chunks() if os.path.exists(STRUCTURED_CHUNKS_FILE) else []
    additional_info = load_additional_info() if os.path.exists(ADDITIONAL_INFO_FILE) else []

    # Combine both datasets
    combined_chunks = structured_chunks + additional_info

    # Create embeddings
    texts = [f"{chunk['header']}: {chunk['content']}" for chunk in combined_chunks]
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Save embeddings and text
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    torch.save({"embeddings": embeddings, "texts": texts}, EMBEDDINGS_FILE)
    st.success("Unified embeddings generated successfully!")

# App Title
st.title("Business Interface - AI Assistant Setup")

# Step 1: Input Business Website
st.header("Step 1: Enter Business Website")
business_url = st.text_input("Website URL", placeholder="Enter your business website URL here")

if st.button("Scrape Website"):
    if business_url:
        with st.spinner("Scraping website..."):
            # Scrape raw text
            raw_text = scrape_website(business_url)
            st.success("Website content scraped successfully!")

            # Chunk text and generate embeddings
            with st.spinner("Processing chunks and generating embeddings..."):
                chunks = chunk_text(raw_text)
                generate_embeddings(chunks)
                st.success("Chunks and embeddings generated successfully!")

            # Generate structured chunks
            with st.spinner("Creating structured chunks..."):
                generate_structured_chunks()
                st.success("Structured chunks created successfully!")
        st.info("Proceed to Step 2 to review and edit structured chunks.")
    else:
        st.error("Please enter a valid website URL.")

# Step 2: Review and Edit Structured Chunks
st.header("Step 2: Review and Edit Structured Chunks")
if os.path.exists(STRUCTURED_CHUNKS_FILE):
    chunks = load_chunks()
    for chunk in chunks:
        st.subheader(chunk["header"])
        chunk["content"] = st.text_area(f"Edit {chunk['header']}", value=chunk["content"], height=100)

    if st.button("Save Changes to Structured Chunks"):
        with st.spinner("Saving changes and regenerating structured embeddings..."):
            save_chunks(chunks)
            st.success("Structured chunks updated successfully!")
else:
    st.info("No structured chunks available. Please complete Step 1 first.")

# Step 3: Add Additional Info
st.header("Step 3: Add Additional Info")
additional_info = load_additional_info()
new_header = st.text_input("New Header", placeholder="Enter the header (e.g., FAQs, Offers, etc.)")
new_content = st.text_area("New Content", placeholder="Enter the content for the new header", height=100)

if st.button("Add Additional Info"):
    if new_header and new_content:
        additional_info.append({"header": new_header, "content": new_content})
        save_additional_info(additional_info)
        st.success("Additional info added successfully!")
    else:
        st.error("Both header and content are required.")

# Display existing additional info
if additional_info:
    st.subheader("Existing Additional Info")
    for info in additional_info:
        st.subheader(info["header"])
        info["content"] = st.text_area(f"Edit {info['header']}", value=info["content"], height=100)

    if st.button("Save Changes to Additional Info"):
        save_additional_info(additional_info)
        st.success("Changes to additional info saved successfully!")

# Step 4: Complete Setup
st.header("Step 4: Complete AI Assistant Setup")

if st.button("Complete Setup"):
    with st.spinner("Generating unified embeddings for your AI Assistant..."):
        generate_combined_embeddings()
        st.success("Your AI Assistant is ready to rock!")
