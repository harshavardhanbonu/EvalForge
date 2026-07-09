import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from app.core.config import settings
from app.utils.prompts import JUDGE_PROMPT

# Initialize the Gemini Client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Define the exact JSON structure we demand from the Judge
class JudgeEvaluation(BaseModel):
    is_approved: bool
    feedback: str

def evaluate_draft(query: str, context: str, draft: str) -> dict:
    """
    The Judge (Gemini Pro) evaluates the Student's draft.
    Returns a dictionary with 'is_approved' (boolean) and 'feedback' (string).
    """
    prompt = f"Context:\n{context}\n\nUser Query:\n{query}\n\nStudent Draft:\n{draft}"
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_PROMPT,
                temperature=0.1, # Extremely low temperature for strict, robotic evaluation
                response_mime_type="application/json",
                response_schema=JudgeEvaluation, # Forces the output to match our Pydantic model
            ),
        )
        
        # Parse the JSON string returned by the model into a Python dictionary
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        # Fail safe: if the judge crashes, we do NOT approve the draft.
        return {"is_approved": False, "feedback": f"Judge encountered a system error: {str(e)}"}