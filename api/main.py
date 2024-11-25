import os
import time
from fastapi import FastAPI, HTTPException
from typing import List
import pinecone
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeEmbeddings
from ctransformers import AutoModelForCausalLM
from fastapi.responses import JSONResponse
from pinecone import Pinecone, ServerlessSpec

# Configuration
PINECONE_API_KEY = "624cd15e-b2fc-4e1b-99f4-71e0edb92447"
PINECONE_API_ENV = "us-east-1"
INDEX_NAME = "pinecone"

# Initialize FastAPI app
app = FastAPI()

# Set up Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


# Initialize Pinecone embeddings
def initialize_pinecone():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENV),
        )
    return PineconeEmbeddings(model="multilingual-e5-large")


# Initialize Llama model (ensure it's compatible with your environment)
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


# Initialize LLM and embeddings
llm = initialize_llm()
if llm is None:
    raise Exception("Failed to initialize LLM model")

embeddings = initialize_pinecone()

# Set up Pinecone index
index = pinecone.Index(
    index_name=INDEX_NAME,
    host="https://pinecone-azpdmbh.svc.aped-4627-b74a.pinecone.io",
    api_key="1f5403e4-2faa-481a-814d-19b3204261a8",
)


# Format context for the response
def format_context(chunks: List[dict]) -> str:
    context = ""
    for chunk in chunks[:3]:  # Using top 3 most relevant chunks
        if "metadata" in chunk and "text" in chunk["metadata"]:
            context += f"{chunk['metadata']['text']}\n\n"
    return context.strip()


# Generate response using Llama 2
def generate_llama2_response(llm, query: str, context: str) -> str:
    try:
        prompt_template = """<s>[INST] You are a helpful spiritual assistant. Use the provided context to answer questions accurately and concisely.
        If you can't find the answer in the context, say so honestly.
        Context:
        {context}
        Question: {query}
        Answer: [/INST]"""

        full_prompt = prompt_template.format(context=context, query=query)
        print("Sending to LLM")
        response = llm(full_prompt, max_new_tokens=512)
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Main function to process the query and generate a response
def chatbot_query(query: str, index, embeddings, llm) -> str:
    try:
        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding, top_k=3, include_values=False, include_metadata=True
        )
        chunks = results.get("matches", [])
        if not chunks:
            return "I couldn't find any relevant information to answer your question."
        context = format_context(chunks)
        print(context)
        response = generate_llama2_response(llm, query, context)
        return response
    except Exception as e:
        return f"An error occurred while processing your query: {str(e)}"


# FastAPI GET Endpoint to handle chatbot queries
@app.get("/query")
async def chatbot_query_endpoint(query: str):
    """GET endpoint to process the query."""
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required.")

        print(f"\nProcessing query: {query}")
        start_time = time.time()

        # Process the query using the chatbot logic
        response = chatbot_query(query, index, embeddings, llm)

        # Calculate the time taken to process the query
        end_time = time.time()
        time_taken = end_time - start_time

        return JSONResponse(
            content={"response": response, "time_taken": f"{time_taken:.2f} seconds"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
