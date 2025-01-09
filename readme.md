project/
├── applications/
│   ├── business_interface.py      # Streamlit app for data ingestion and inspection
│   ├── chat_agent.py              # Streamlit app for querying the AI assistant
├── output/
│   ├── raw_content.txt            # Raw scraped content
│   ├── chunks.txt                 # Preprocessed text chunks
│   ├── embeddings.faiss           # FAISS vector store
│   ├── key_info.json              # Extracted key business information
│   └── app.log                    # Log file for debugging
├── src/
│   ├── chunking_agent.py          # Text chunking logic using LangChain splitter
│   ├── data_ingestion.py          # Scrapes website data and orchestrates ingestion
│   ├── embeddings_agent.py        # Embedding creation and vector store management
│   ├── json_manager.py            # Key info extraction using Generative AI
│   ├── retrieval_agent.py         # Hybrid retriever combining BM25 and FAISS
│   └── utils.py                   # Helper functions (logging, file handling, etc.)
├── requirements.txt               # Python dependencies
└── .env                           # Environment variables (e.g., API keys)
