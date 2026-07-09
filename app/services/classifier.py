import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from app.core.config import settings
from app.utils.prompts import CLASSIFIER_PROMPT

# Initialize the new Gemini Client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Define the exact JSON structure we expect from the AI
class IntentResponse(BaseModel):
    category: str

def classify_intent(query: str) -> dict:
    """
    Uses Gemini Flash to quickly route the user's query into one of three categories:
    SIMPLE_QA, REASONING, or RAG_LOOP.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFIER_PROMPT,
                temperature=0.1, # Low temp for consistent routing
                response_mime_type="application/json",
                response_schema=IntentResponse,
            ),
        )
        
        # Parse the JSON string returned by the model into a dictionary
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Classification failed: {e}")
        # Safe fallback: if the classifier crashes, send it to the RAG Loop
        return {"category": "RAG_LOOP"}