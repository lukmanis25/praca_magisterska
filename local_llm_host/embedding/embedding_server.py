from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]

@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    vectors = model.encode(request.input).tolist()
    return {
        "data": [
            {
                "object": "embedding",
                "embedding": vectors[i],
                "index": i
            } for i in range(len(vectors))
        ],
        "object": "list",
        "model": request.model
    }
