from google import genai
from google.genai import types
from app.core.config import settings
from app.utils.prompts import STUDENT_PROMPT

# We initialize the client using the key from our .env file
client = genai.Client(api_key=settings.GEMINI_API_KEY)

def generate_student_draft(query: str, context: str) -> str:
    """
    Uses Gemini Flash (The Student) to draft an answer based ONLY on the provided context.
    This is designed to be fast and cost-effective.
    """
    # Combine the context and query into a single prompt for the model
    prompt = f"Context Information:\n{context}\n\nUser Query:\n{query}"
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=STUDENT_PROMPT,
                temperature=0.3, # Low temperature keeps it focused on the facts
            ),
        )
        return response.text
    except Exception as e:
        print(f"Student generation failed: {e}")
        return "I apologize, but I encountered an error while trying to generate an answer."

def generate_pro_response(query: str) -> str:
    """
    Uses Gemini Pro directly for complex reasoning, math, or coding questions.
    This bypasses the RAG context entirely.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",  # Using the heavy-duty model
            contents=query,
            config=types.GenerateContentConfig(
                temperature=0.5, # Slightly higher temperature for better reasoning/creativity
            ),
        )
        return response.text
    except Exception as e:
        print(f"Pro generation failed: {e}")
        return "I apologize, but I encountered an error while processing your complex query."