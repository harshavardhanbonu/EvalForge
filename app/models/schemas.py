from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import UUID

# --- Document Ingestion Schemas ---

class DocumentResponse(BaseModel):
    """Schema for the response returned after a PDF is successfully uploaded."""
    id: UUID
    project_id: str
    filename: str
    message: str
    chunks_created: int

    class Config:
        from_attributes = True  # Allows Pydantic to read SQLAlchemy models directly

# --- Query Routing & Evaluation Schemas ---

class QueryRequest(BaseModel):
    """Schema for the incoming user question."""
    project_id: str
    question: str

class QueryResponse(BaseModel):
    """Schema for the final answer returned to the user."""
    answer: str
    route_taken: str  # SIMPLE_QA, REASONING, or RAG_LOOP
    student_attempts: int
    judge_feedback: Optional[List[Dict[str, Any]]] = None  # History of critiques if RAG_LOOP was used