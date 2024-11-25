from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings
import pinecone
import os
import httpx

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PINECONE_HOST = os.getenv("PINECONE_HOST")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Global variables
pinecone_embeddings = None
client = None

# Initialize FastAPI app
app = FastAPI()


# Root path
@app.get("/")
async def root():
    return {"message": "Hello, World!"}


# Pydantic model for query
class QueryRequest(BaseModel):
    query: str


# Pinecone initialization
def initialize_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, dimension=1024, metric="euclidean")
    return PineconeEmbeddings(model="multilingual-e5-large")


# Startup and shutdown events
@app.on_event("startup")
async def startup():
    global pinecone_embeddings, client
    pinecone_embeddings = initialize_pinecone()
    print("Pinecone initialized.")
    client = httpx.AsyncClient()


@app.on_event("shutdown")
async def shutdown():
    global client
    if client:
        await client.aclose()
        client = None


# Format context from Pinecone query results
def format_context(chunks):
    context = ""
    for chunk in chunks[:3]:
        context += f"{chunk['metadata'].get('text', '')}\n\n"
    return context.strip()


# Query Pinecone and generate response
async def process_query(query: str) -> str:
    try:
        embedding = pinecone_embeddings.embed_query(query)
        index = pinecone.Index(
            index_name=INDEX_NAME, host=PINECONE_HOST, api_key=PINECONE_API_KEY
        )
        results = index.query(vector=embedding, top_k=3, include_metadata=True)
        chunks = results.get("matches", [])
        if not chunks:
            return "No relevant information found."
        context = format_context(chunks)
        return f"Response generated using context:\n\n{context}"
    except Exception as e:
        return f"Error processing query: {str(e)}"


# Telegram webhook endpoint
@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    global client
    data = await request.json()
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if not (chat_id and text):
        return {"status": "ignored"}

    try:
        response_text = await process_query(text)
        await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": response_text},
        )
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


# Query endpoint for manual testing
@app.post("/query")
async def query_endpoint(query: QueryRequest):
    try:
        response = await process_query(query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
