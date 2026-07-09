from google import genai
from google.genai import types
from typing import List
from app.core.config import settings

# Configure the new Gemini Client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Takes a list of text chunks and returns a list of 768-dimensional embeddings.
    Leverages Matryoshka Representation Learning (MRL) to truncate vectors.
    """
    if not chunks:
        return []

    try:
        # Using the active 'gemini-embedding-001' model instead of the deprecated one
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunks,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768  # <-- THE MATRYOSHKA MAGIC
            )
        )
        
        # The new SDK returns an object where .embeddings is a list of embeddings, 
        # and .values contains the actual floats.
        return [embedding.values for embedding in result.embeddings]
        
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings from Gemini API: {str(e)}")