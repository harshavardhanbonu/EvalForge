from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid

from app.core.db import get_db
from app.models.schemas import QueryRequest, QueryResponse
from app.models.db_models import EvaluationLog
from app.services.agent_workflow import process_query

router = APIRouter()

@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Takes a user question, runs it through the Agentic Workflow, 
    logs the entire process for observability, and returns the answer.
    """
    try:
        # 1. Run the master workflow
        result = process_query(request.question, request.project_id, db)
        
        # 2. Log the execution for observability
        log_entry = EvaluationLog(
            id=uuid.uuid4(),
            project_id=request.project_id,
            user_query=request.question,
            route_taken=result["route_taken"],
            student_attempts=result["student_attempts"],
            final_answer=result["answer"],
            was_successful=result["route_taken"] != "FALLBACK",
            judge_feedback_history=result["judge_feedback_history"]
        )
        
        db.add(log_entry)
        db.commit()
        
        # 3. Return the result to the user
        return QueryResponse(
            answer=result["answer"],
            route_taken=result["route_taken"],
            student_attempts=result["student_attempts"],
            judge_feedback=result["judge_feedback_history"]
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")