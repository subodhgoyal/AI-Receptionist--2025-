import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
EXCLUDED_PAGES = ["terms", "privacy", "cookie-policy", "blog", "newsletter", "testimonials"]
MODEL_NAME = 'all-MiniLM-L6-v2'
RAW_DATA_FILE = "data/raw_data.txt"
CHUNKS_FILE = "data/chunks.txt"
EMBEDDINGS_FILE = "data/embeddings.pt"

# Initialize model
model = SentenceTransformer(MODEL_NAME)

def scrape_website(url):
    """
    Scrapes text content from the specified website URL, prefixes each text body with its URL,
    and saves all raw data in a single file.
    """
    # Define headers with User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Make the main request with headers
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all links
    links = [a['href'] for a in soup.find_all('a', href=True) if not any(excluded in a['href'] for excluded in EXCLUDED_PAGES)]

    # Normalize relative URLs to absolute URLs
    base_url = url.rstrip("/")
    subpages = [link if link.startswith("http") else f"{base_url}/{link.lstrip('/')}" for link in links]

    print("Subpages to be scraped:")
    for subpage in subpages:
        print(subpage)

    # Initialize a variable to store all raw data
    raw_data = ""

    # Scrape each subpage
    for subpage in subpages:
        try:
            # Make subpage requests with headers
            sub_response = requests.get(subpage, headers=headers, timeout=10)
            sub_response.raise_for_status()
            sub_soup = BeautifulSoup(sub_response.content, 'html.parser')

            # Extract text from subpage
            sub_text = "\n".join(
                tag.get_text(strip=True)
                for tag in sub_soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            )

            # Append URL-prefixed text to raw_data
            raw_data += f"URL: {subpage}\n{sub_text}\n\n"
            print(f"Scraped: {subpage}")
        except Exception as e:
            print(f"Failed to scrape {subpage}: {e}")

    # Also scrape the main page
    main_text = "\n".join(
        tag.get_text(strip=True)
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
    )
    raw_data = f"URL: {url}\n{main_text}\n\n" + raw_data

    # Save the combined raw data to a file
    os.makedirs(os.path.dirname(RAW_DATA_FILE), exist_ok=True)
    with open(RAW_DATA_FILE, "w", encoding="utf-8") as file:
        file.write(raw_data)

    print(f"Raw data saved to {RAW_DATA_FILE}")
    return raw_data


def chunk_text(raw_text, chunk_size=500, overlap=100):
    """
    Splits raw text into chunks using LangChain's RecursiveCharacterTextSplitter
    and saves the chunks with IDs.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(raw_text)

    # Add chunk IDs
    chunks_with_ids = [f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)]

    # Save chunks with IDs to the file
    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        file.write("\n\n".join(chunks_with_ids))

    print(f"Text chunks saved to {CHUNKS_FILE}")
    return chunks

def generate_embeddings(chunks):
    """
    Generates embeddings for text chunks and saves them to a file.
    """
    embeddings = model.encode(chunks, convert_to_tensor=True)

    # Save embeddings and chunks
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    torch.save({"embeddings": embeddings, "texts": chunks}, EMBEDDINGS_FILE)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")
