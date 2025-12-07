from typing import Union, List, Dict, Any
from fastapi import FastAPI

# Import routers
from routers import probability, embedding, neural_networks

app = FastAPI(
    title="APAN5560 GenAI API", 
    description="API for probability, text processing, embeddings, neural networks, and generative models",
    version="1.0.0"
)

# Include routers
app.include_router(probability.router)
app.include_router(embedding.router)
app.include_router(neural_networks.router)

@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint providing API information."""
    return {"message": "APAN5560 GenAI API", "version": "1.0.0"}