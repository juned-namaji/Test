from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import os
import json
from typing import List
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pinecone
from ctransformers import AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_HOST = os.getenv("PINECONE_HOST")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Telegram Webhook URL (should point to your deployed endpoint)
WEBHOOK_URL = f"https://<your-vercel-deployment>/telegram_webhook"

# Set up Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_API_ENV"] = PINECONE_API_ENV

app = FastAPI()

# Telegram API URL
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


class QueryRequest(BaseModel):
    query: str


# Global variables for embeddings, LLM, and Pinecone index
embeddings = None
llm = None
index = None


# Pinecone Initialization
def initialize_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, dimension=1024, metric="euclidean")
    return PineconeEmbeddings(model="multilingual-e5-large")


# Initialize Llama Model
def initialize_llm():
    """Initialize the Llama 2 model using ctransformers."""
    try:
        print("Initializing LLM")
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            model_file="llama-2-7b-chat.ggmlv3.q4_K_M.bin",
            max_new_tokens=512,
            temperature=0.5,
            repetition_penalty=1.15,
            context_length=1024,
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return None


def format_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into a context string."""
    context = ""
    for chunk in chunks[:3]:  # Using top 3 most relevant chunks
        if "metadata" in chunk and "text" in chunk["metadata"]:
            context += f"{chunk['metadata']['text']}\n\n"
    return context.strip()


def generate_llama2_response(llm, query: str, context: str) -> str:
    """Generate response using Llama 2 based on query and context."""
    try:
        prompt_template = """<s>[INST] You are a helpful spiritual assistant. Use the provided context to answer questions accurately and concisely.
        If you can't find the answer in the context, say so honestly.
        Context:
        {context}
        Question: {query}
        Answer: [/INST]"""

        full_prompt = prompt_template.format(context=context, query=query)
        response = llm(full_prompt, max_new_tokens=512)
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


@app.on_event("startup")
def startup_event():
    global embeddings, llm, index
    embeddings = initialize_pinecone()
    llm = initialize_llm()
    index = pinecone.Index(
        index_name=INDEX_NAME, api_key=PINECONE_API_KEY, host=PINECONE_HOST
    )
    set_webhook()


@app.get("/query")
async def chatbot_query_endpoint(query: str):
    """
    This endpoint handles GET requests for querying the chatbot.
    The query parameter is passed in the URL like: /query?query=<your-query>
    """
    try:
        global embeddings, llm, index

        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding, top_k=3, include_values=False, include_metadata=True
        )
        chunks = results.get("matches", [])

        if not chunks:
            return {
                "response": "I couldn't find any relevant information to answer your question."
            }

        context = format_context(chunks)
        response = generate_llama2_response(llm, query, context)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    try:
        global embeddings, llm, index

        # Parse the incoming Telegram update
        update = await request.json()
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        query = message.get("text")

        if not query or not chat_id:
            return {"status": "ignored"}

        # Process the query
        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding, top_k=3, include_values=False, include_metadata=True
        )
        chunks = results.get("matches", [])
        context = format_context(chunks) if chunks else ""
        response = generate_llama2_response(llm, query, context)

        # Send the response back to the user
        send_telegram_message(chat_id, response)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def send_telegram_message(chat_id: int, text: str):
    """Send a message to a Telegram chat."""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)


def set_webhook():
    """Set the Telegram webhook."""
    url = f"{TELEGRAM_API_URL}/setWebhook"
    payload = {"url": WEBHOOK_URL}
    response = requests.post(url, json=payload)
    print("Webhook set:", response.json())
