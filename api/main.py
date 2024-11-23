# Imports and Configuration
import os
import re
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pinecone
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request

# Load environment variables
load_dotenv()

app = FastAPI()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PINECONE_HOST = os.getenv("PINECONE_HOST")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Global AsyncClient
client = None

# Set up Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

# Pinecone Initialization
def initialize_pinecone():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region=PINECONE_API_ENV)
        )
    return PineconeEmbeddings(model="multilingual-e5-large")

initialize_pinecone()

# Startup event to initialize the HTTP client
@app.on_event("startup")
async def startup():
    global client
    if client is None:  # Ensure client is initialized only once
        client = httpx.AsyncClient()

# Shutdown event to close the HTTP client session
@app.on_event("shutdown")
async def shutdown():
    global client
    if client is not None:
        await client.aclose()
        client = None

# Home endpoint
@app.get("/")
def ret():
    return {"Hello": "World"}

# Query Pinecone index directly
@app.get("/{query}")
async def query_pinecone(query: str):
    embedding = PineconeEmbeddings(model="multilingual-e5-large").embed_query(query)

    index = pinecone.Index(
        index_name=INDEX_NAME,
        host=PINECONE_HOST,
        api_key=PINECONE_API_KEY
    )
    
    results = index.query(
        vector=embedding,
        top_k=7,
        include_values=False,
        include_metadata=True
    )

    top_chunks = results.get('matches', [])
    response = []
    for i, chunk in enumerate(top_chunks, start=1):
        response.append({
            "rank": i,
            "date": chunk['metadata'].get('date', 'N/A'),
            "title": chunk['metadata'].get('title', 'N/A'),
            "text": chunk['metadata'].get('text', 'N/A')
        })
    
    return response

# Telegram webhook endpoint
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if chat_id and text:
        response_text = await process_query(text)
        await send_message(chat_id, response_text)

    return {"status": "ok"}

# Process the query using Pinecone
async def process_query(query: str) -> str:
    try:
        embedding = PineconeEmbeddings(model="multilingual-e5-large").embed_query(query)

        index = pinecone.Index(
            index_name=INDEX_NAME,
            host=PINECONE_HOST,
            api_key=PINECONE_API_KEY
        )

        results = index.query(
            vector=embedding,
            top_k=7,
            include_values=False,
            include_metadata=True
        )

        top_chunks = results.get('matches', [])
        if not top_chunks:
            return "No relevant information found."

        response = ""
        for i, chunk in enumerate(top_chunks, start=1):
            date = chunk['metadata'].get('date', 'N/A')
            title = chunk['metadata'].get('title', 'N/A')
            text = chunk['metadata'].get('text', 'N/A')
            response += f"Rank {i}:\nDate: {date}\nTitle: {title}\nText: {text}\n\n"

        return response.strip()

    except Exception as e:
        return f"Error processing query: {str(e)}"

# Send a message to the Telegram user
async def send_message(chat_id, text):
    global client
    if client is None:  # Reinitialize client if missing
        client = httpx.AsyncClient()

    try:
        response = await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
        response.raise_for_status()  # Raise an error if the request failed
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")