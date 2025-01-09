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
from src.content_manager import generate_faq_responses, save_autogen_responses, load_autogen_responses

# File paths
EMBEDDINGS_FILE = "data/faq_responses_embeddings.pt"
AUTOGEN_FAQS_FILE = "data/faq_autogen_responses.json"
MANUAL_FAQS_FILE = "data/faq_manual_responses.json"
BUSINESS_CONFIG_FILE = "data/business_config.json"

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with the correct model

# Utility functions
def save_business_config(domain_type):
    """
    Save the selected business domain type to business_config.json.
    """
    config = {"domain_type": domain_type}
    os.makedirs(os.path.dirname(BUSINESS_CONFIG_FILE), exist_ok=True)
    with open(BUSINESS_CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)
    st.success(f"Business domain type '{domain_type}' saved successfully!")

def load_manual_faqs():
    """Load additional (manual) Q&A info from the JSON file or initialize an empty list."""
    if os.path.exists(MANUAL_FAQS_FILE):
        with open(MANUAL_FAQS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []  # Return an empty list if the file doesn't exist

def save_manual_faqs(manual_faqs):
    """Save additional (manual) Q&A info to the JSON file."""
    os.makedirs(os.path.dirname(MANUAL_FAQS_FILE), exist_ok=True)
    with open(MANUAL_FAQS_FILE, "w", encoding="utf-8") as file:
        json.dump(manual_faqs, file, indent=4)

def generate_combined_embeddings():
    """
    Generate embeddings by combining auto-generated FAQs and additional (manual) Q&A.
    """
    # Load auto-generated FAQs and additional (manual) Q&A
    autogen_faqs = load_autogen_responses() if os.path.exists(AUTOGEN_FAQS_FILE) else []
    manual_faqs = load_manual_faqs() if os.path.exists(MANUAL_FAQS_FILE) else []

    # Combine both datasets
    combined_faqs = autogen_faqs + manual_faqs

    # Prepare texts for embeddings (using question and answer)
    texts = [f"Question: {faq['question']} Answer: {faq['answer']}" for faq in combined_faqs]

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Save embeddings and text
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    torch.save({"embeddings": embeddings, "texts": texts}, EMBEDDINGS_FILE)
    st.success("Unified embeddings generated successfully!")

# App Title
st.title("Business Interface - AI Assistant Setup")

# Step 1: Select Business Type
st.header("Step 1: Select Business Type")
domain_type = st.selectbox(
    "Select your business type",
    ["Dental Clinic", "Veterinary Clinic"],
    help="Choose the type of business for the AI Assistant setup."
)

if st.button("Save Business Type"):
    if domain_type:
        save_business_config(domain_type.lower().replace(" ", "_"))
    else:
        st.error("Please select a business type.")

# Step 2: Input Business Website
st.header("Step 2: Enter Business Website")
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

            # Generate FAQ-based structured chunks
            with st.spinner("Generating FAQ responses..."):
                generate_faq_responses()
                st.success("FAQ responses generated successfully!")
        st.info("Proceed to Step 3 to review and edit FAQs.")
    else:
        st.error("Please enter a valid website URL.")

# Step 3: Review and Edit Auto-Generated FAQ Responses
st.header("Step 3: Review and Edit Auto-Generated FAQ Responses")
if os.path.exists(AUTOGEN_FAQS_FILE):
    autogen_faqs = load_autogen_responses()
    for idx, faq in enumerate(autogen_faqs):
        st.subheader(f"FAQ #{idx + 1}")
        st.text(f"Question: {faq['question']}")
        faq["answer"] = st.text_area(
            f"Edit Answer for Question #{idx + 1}", value=faq["answer"], height=100
        )
        faq["metadata"] = st.text_area(
            f"Edit Metadata Tags for Question #{idx + 1} (comma-separated)",
            value=", ".join(faq["metadata"]),
            height=70,
        ).split(", ")

    if st.button("Save Changes to Auto-Generated FAQs"):
        save_autogen_responses(autogen_faqs)
        st.success("Changes to auto-generated FAQs saved successfully!")
else:
    st.info("No auto-generated FAQs available. Please complete Step 2 first.")

# Step 4: Add Additional Q&A
st.header("Step 4: Add Additional Q&A")
manual_faqs = load_manual_faqs()  # Updated function name
new_question = st.text_input("New Question", placeholder="Enter the question (e.g., What services do you offer?)")
new_answer = st.text_area("New Answer", placeholder="Enter the answer for the question", height=100)

if st.button("Add Additional Q&A"):
    if new_question and new_answer:
        manual_faqs.append({"question": new_question, "answer": new_answer, "metadata": []})
        save_manual_faqs(manual_faqs)  # Updated function name
        st.success("Additional Q&A added successfully!")
    else:
        st.error("Both question and answer are required.")

# Display existing additional Q&A
if manual_faqs:
    st.subheader("Existing Additional Q&A")
    for faq in manual_faqs:
        st.subheader(faq["question"])  # Display the question
        faq["answer"] = st.text_area(
            f"Edit Answer for '{faq['question']}'", 
            value=faq["answer"], 
            height=100
        )
        faq["metadata"] = st.text_area(
            f"Edit Metadata for '{faq['question']}'",
            value=", ".join(faq.get("metadata", [])),
            height=70
        )

    if st.button("Save Changes to Additional Q&A"):
        save_manual_faqs(manual_faqs)  # Updated function name
        st.success("Changes to additional Q&A saved successfully!")


# Step 5: Complete Setup
st.header("Step 5: Complete AI Assistant Setup")

if st.button("Complete Setup"):
    with st.spinner("Generating unified embeddings for your AI Assistant..."):
        generate_combined_embeddings()
        st.success("Your AI Assistant is ready to rock!")