import sys
import os

# Add the root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import functions from webscraping_agent
from src.webscraping_agent import scrape_website, chunk_text, generate_embeddings

# Test the functions
url = "https://www.dentistryatlasalle.com"  # Replace with your target URL
raw_text = scrape_website(url)
print("Scraping completed!")

chunks = chunk_text(raw_text)
print(f"Chunking completed! {len(chunks)} chunks created.")

generate_embeddings(chunks)
print("Embeddings generated successfully!")
