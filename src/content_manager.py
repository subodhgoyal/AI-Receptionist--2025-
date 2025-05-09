import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# File paths
EMBEDDINGS_FILE = "data/web_scraped_data_embeddings.pt"
FAQ_AUTOGEN_RESPONSES_FILE = "data/faq_autogen_responses.json"
FAQ_MANUAL_RESPONSES_FILE = "data/faq_manual_responses.json"
DOMAIN_FAQ_FILE = "data/domain_faqs.json"
BUSINESS_CONFIG_FILE = "data/business_config.json"
FAQ_RESPONSES_EMBEDDINGS_FILE = "data/faq_responses_embeddings.pt"

# Initialize OpenAI client and embedding model
client = OpenAI()
model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with the actual model used

def load_embeddings():
    """
    Load embeddings and their associated texts from the embeddings file.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_FILE}.")
    embedding_data = torch.load(EMBEDDINGS_FILE, map_location=torch.device("cpu"), weights_only=True)
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

def load_domain_faqs(domain_type):
    """
    Load domain-specific FAQs from the domain_faqs.json file.
    """
    if not os.path.exists(DOMAIN_FAQ_FILE):
        raise FileNotFoundError(f"Domain FAQs file not found at {DOMAIN_FAQ_FILE}.")
    with open(DOMAIN_FAQ_FILE, "r", encoding="utf-8") as file:
        faqs = json.load(file)
    return faqs.get(domain_type, [])

def load_business_config():
    """
    Load business configuration, including domain type.
    """
    if not os.path.exists(BUSINESS_CONFIG_FILE):
        raise FileNotFoundError(f"Business config file not found at {BUSINESS_CONFIG_FILE}.")
    with open(BUSINESS_CONFIG_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def generate_faq_responses():
    """
    Generate FAQ responses using domain-specific questions and embeddings search.
    """
    business_config = load_business_config()
    domain_type = business_config.get("domain_type")
    if not domain_type:
        raise ValueError("Domain type is missing in the business config file.")

    faqs = load_domain_faqs(domain_type)
    faq_responses = []

    for question in faqs:
        top_chunks = search_embeddings(question, top_k=5)

        prompt = f"""
        Based on the following relevant information, provide a concise and specific answer to the question: "{question}".

        Relevant Information:
        {"\n".join(top_chunks)}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in generating FAQs for businesses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()

        metadata_tags = generate_tags(question)
        faq_responses.append({
            "question": question,
            "answer": answer,
            "metadata": metadata_tags
        })

    save_autogen_responses(faq_responses)
    print("FAQ responses generated and saved successfully!")

def generate_tags(question):
    """
    Generate metadata tags for a given question by inferring variations.
    """
    synonyms = {
        "operating hours": ["working hours", "business hours", "clinic hours"],
        "location": ["address", "clinic location"],
        "services": ["offerings", "treatments"]
    }
    tags = []
    for key, variations in synonyms.items():
        if key in question.lower():
            tags.extend([key] + variations)
    return tags

def load_autogen_responses():
    """
    Load auto-generated FAQ responses from the JSON file.
    """
    if os.path.exists(FAQ_AUTOGEN_RESPONSES_FILE):
        with open(FAQ_AUTOGEN_RESPONSES_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    raise FileNotFoundError(f"Auto-generated FAQ responses file not found at {FAQ_AUTOGEN_RESPONSES_FILE}.")

def save_autogen_responses(responses):
    """
    Save auto-generated FAQ responses to the JSON file.
    """
    os.makedirs(os.path.dirname(FAQ_AUTOGEN_RESPONSES_FILE), exist_ok=True)
    with open(FAQ_AUTOGEN_RESPONSES_FILE, "w", encoding="utf-8") as file:
        json.dump(responses, file, indent=4)

def create_embeddings_for_faq_responses():
    """
    Generate embeddings for FAQ responses (question + answer) and save them to a file.
    """
    autogen_responses = load_autogen_responses()
    manual_responses = []
    if os.path.exists(FAQ_MANUAL_RESPONSES_FILE):
        with open(FAQ_MANUAL_RESPONSES_FILE, "r", encoding="utf-8") as file:
            manual_responses = json.load(file)

    all_responses = autogen_responses + manual_responses

    combined_texts = [f"Question: {faq['question']} Answer: {faq['answer']}" for faq in all_responses]

    embeddings = model.encode(combined_texts, convert_to_tensor=True)

    torch.save({"embeddings": embeddings, "texts": combined_texts}, FAQ_RESPONSES_EMBEDDINGS_FILE)
    print(f"Embeddings for FAQ responses saved to {FAQ_RESPONSES_EMBEDDINGS_FILE}.")
