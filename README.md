# Knowledge Assistant for Tucows Domains Support Team

## Overview
This repository contains a Knowledge Assistant designed to help the Tucows Domains Support Team quickly find answers to common questions and issues related to domain management. The assistant leverages a comprehensive knowledge base and advanced search capabilities to provide accurate and timely responses.
With the help of a RAG (Retrieval-Augmented Generation) architecture, the assistant can retrieve relevant information from the knowledge base and generate coherent answers to user queries.

## Problem Statement
You will build a Knowledge Assistant that can analyze customer support queries and return structured, relevant, and helpful responses. The assistant should use a Retrieval-Augmented Generation (RAG) pipeline powered by an LLM and follow the Model Context Protocol (MCP) to produce structured output.

### Sample Input (Support Ticket)
```
My domain was suspended and I didn’t get any notice. How can I reactivate it?
```

### Expected Output (MCP-compliant JSON)
```json
{
  "answer": "Your domain may have been suspended due to a violation of policy or missing WHOIS information. Please update your WHOIS details and contact support.",
  "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
  "action_required": "escalate_to_abuse_team"
}
```

## Features
- Natural Language Processing (NLP) to understand and analyze support queries.
- Retrieval-Augmented Generation (RAG) to fetch relevant information from the knowledge base.
- Model Context Protocol (MCP) compliance for structured output.
- FAISS (Facebook AI Similarity Search) for efficient vector-based search and retrieval.
- Integration with Ollama's Mistral model for generating responses.
- Confidence scoring to assess the reliability of generated answers.
- Fully testable FastAPI backend.
- Model deployment using Docker.
- Pydantic models for data validation and serialization.
- Basic HTML frontend for user interaction.

## Project Structure
```
├── src/
│   ├── api/              # FastAPI endpoints and routes
│   ├── llm/              # LLM integration and prompt management
│   ├── embeddings/       # FAISS vector store and embedding logic
│   ├── utils/            # Data loading, preprocessing, and confidence scoring
│   └── config.py         # Configuration settings
│
├── tests/                # API and LLM unit tests
├── data/                 # FAQ dataset
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
├── Dockerfile            # Containerization setup
├── docker-compose.yml    # Docker Compose configuration
├── static/               # Basic HTML frontend
├── scripts/              # Utility scripts
├── faiss_index/          # Generated FAISS index files
└── README.md
```

## Getting Started
1. **Clone the repository:**
```bash
git clone https://github.com/aabiyah/aabiyah-interview-exercise-ai-tucows.git
cd aabiyah-interview-exercise-ai-tucows
```

2. **Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
# or
.venv\Scripts\activate    # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file and add your Ollama configurations:
```
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral
```

5. **Run the application:**
To run locally:
```bash
PYTHONPATH=src python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
Or using Docker Compose:
```bash
docker-compose up --build
```
Then open your browser and navigate to [http://localhost:8000](http://localhost:8000)

6. **Access the application:**
- FastAPI backend: [http://localhost:8000](http://localhost:8000)
- HTML frontend: [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

7. **Run tests:**
```bash
pytest tests/
```

## Technologies Used
- **FastAPI** — RESTful API Framework  
- **FAISS** — Vector similarity search for FAQ retrieval  
- **Ollama** — LLM response generation  
- **Pydantic** — Data validation and modeling  
- **Docker** — Containerization
- **Uvicorn** — ASGI server for FastAPI

## Challenges & Lessons Learned
Building this system required balancing model performance, latency, and reliability while integrating local and cloud-based LLMs.

- **OpenAI API:** Initially used for structured JSON responses, but the usage quota was limited, causing frequent `rate_limit` errors.  
- **Llama 3.2 (Ollama):** Produced strong reasoning responses, but response latency was high and sometimes unresponsive during concurrent API calls.  
- **Phi-3 (Mini Model, Ollama):** Lightweight and efficient, but failed to maintain consistent output format (JSON parsing errors).  
- **Mistral (Ollama):** Struck the right balance — fast inference, structured responses, and stable local execution. It became the final choice for the deployment.  

Overall, the lesson learned was to prioritize controllability and stability of the LLM over sheer capability, especially when integrating it into a backend API pipeline.

## Example Outputs

### Example 1: Console JSON Response
![Example JSON Response](./images/Console%20Output.png)

### Example 2: Frontend Responses
![Frontend Response 1](./images/Frontend%20Output%201.png)
![Frontend Response 2](./images/Frontend%20Output%202.png)

### Example 3: Error Handling
![Model Unresponsive](./images/Model%20Unresponsive.png)

---
## Author
**Aabiyah Zehra**  
Master of Applied Computing @ Wilfrid Laurier University  
AI Engineer Candidate @ Tucows Inc.
[GitHub](https://github.com/aabiyah) | [LinkedIn](https://www.linkedin.com/in/aabiyah-zehra-526528202)