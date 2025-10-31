# FastAPI application for the Tucows Knowledge Assistant with frontend.
import json
import os
import os as _os
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from .response_models import TicketRequest, TicketResponse
from embeddings.embedder import FAQEmbedder
from embeddings.vector_store import FAISSVectorStore
from llm.ollama_client import TucowsSupportLLM
from utils.confidence import calculate_confidence, should_escalate
from config import TOP_K_RETRIEVAL, CONFIDENCE_THRESHOLD, STATIC_DIR

# Global instances
embedder: FAQEmbedder = None
vector_store: FAISSVectorStore = None
llm_client: TucowsSupportLLM = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, vector_store, llm_client

    print("Starting Tucows Domains Knowledge Assistant...")

    # Loading embedding model
    embedder = FAQEmbedder()

    # Loading FAISS index
    vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    try:
        vector_store.load_index()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python scripts/build_index.py' first!")
        raise

    # Load Ollama LLM client
    llm_client = TucowsSupportLLM()
    print("Ready to process tickets!\n")

    yield
    print("Shutting down...")

# App setup
app = FastAPI(
    title="Tucows Domains Knowledge Assistant",
    description="An LLM-powered customer support ticket assistant using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Enabling CORS in case of frontend being served from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serving frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return _os.path.join(STATIC_DIR, "index.html")

# Ticket resolution endpoint
@app.post("/resolve-ticket", response_model=TicketResponse)
async def resolve_ticket(
        request: TicketRequest,
        debug: bool = Query(False, description="Include reasoning_trace in response")
) -> TicketResponse:
    try:
        # Step 1: Embedding query
        query_embedding = embedder.embed_query(request.ticket_text)

        # Step 2: Retrieving top-K FAQs
        retrieved_faqs = vector_store.search(query_embedding, top_k=TOP_K_RETRIEVAL)
        if not retrieved_faqs:
            raise HTTPException(status_code=500, detail="No FAQs retrieved. Index may be empty.")

        # Step 3: Generating LLM response using Ollama
        llm_response = llm_client.generate_response(request.ticket_text, retrieved_faqs)

        # Validating and ensuring required keys exist before any downstream uses
        if not isinstance(llm_response, dict):
            raise HTTPException(status_code=500, detail="Invalid LLM response format")

        llm_response.setdefault("answer", "No answer generated.")
        llm_response.setdefault("references", [])
        llm_response.setdefault("action_required", "none")
        llm_response.setdefault("reasoning_trace", None)

        # Step 4: Calculating confidnce (based on similarity scores)
        similarity_scores = [r.get('similarity_score', 0.0) for r in retrieved_faqs]
        confidence = calculate_confidence(similarity_scores, llm_response, len(retrieved_faqs))

        # Step 5: Determining action required safely
        action = should_escalate(confidence, llm_response["action_required"], CONFIDENCE_THRESHOLD)

        # Building the response
        return TicketResponse(
            answer=llm_response["answer"],
            references=llm_response["references"],
            action_required=action,
            confidence_score=confidence,
            reasoning_trace=llm_response.get("reasoning_trace") if debug else None
        )

    except Exception as e:
        print(f"Error processing ticket: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process ticket: {str(e)}")


@app.post("/api/ask", response_model=TicketResponse)
async def api_ask(request: Request):
    try:
        payload = await request.json()
        query = payload.get("query", "")

        if not query:
            raise HTTPException(status_code=400, detail="Missing query field")

        ticket_request = TicketRequest(ticket_text=query)
        response = await resolve_ticket(ticket_request)
        return response

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        print(f"Error in /api/ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

