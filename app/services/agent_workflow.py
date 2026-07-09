from sqlalchemy.orm import Session
from app.models.db_models import Document, DocumentChunk
from app.services.embeddings import generate_embeddings
from app.services.classifier import classify_intent
from app.services.generator import generate_student_draft, generate_pro_response
from app.services.evaluator import evaluate_draft

def retrieve_context(query: str, project_id: str, db: Session, limit: int = 5) -> str:
    """
    Converts the user's query into a vector and performs a Cosine Similarity
    search in pgvector to find the most relevant document chunks.
    """
    # 1. Embed the user's query
    query_embedding = generate_embeddings([query])[0]
    
    # 2. Search PostgreSQL using pgvector's cosine distance operator (<=>)
    results = db.query(DocumentChunk).join(DocumentChunk.document)\
        .filter(Document.project_id == project_id)\
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))\
        .limit(limit).all()
        
    if not results:
        return "No relevant documents found in the database."
        
    # 3. Combine the chunks into a single text block
    return "\n\n---\n\n".join([chunk.text_content for chunk in results])

def process_query(query: str, project_id: str, db: Session) -> dict:
    """
    The master workflow. Routes the query, manages the Student-Judge loop, 
    and enforces the retry guardrails.
    """
    # Step 1: Classify the Intent
    intent_data = classify_intent(query)
    route = intent_data.get("category", "RAG_LOOP")
    
    # Step 2: Handle Math/Logic Bypass
    if route == "REASONING":
        answer = generate_pro_response(query)
        return {
            "answer": answer,
            "route_taken": route,
            "student_attempts": 0,
            "judge_feedback_history": []
        }
        
    # Step 3: Retrieve Context for Document-based tasks
    context = retrieve_context(query, project_id, db)
    
    # Step 4: Handle Simple QA (No Judge needed)
    if route == "SIMPLE_QA":
        answer = generate_student_draft(query, context)
        return {
            "answer": answer,
            "route_taken": route,
            "student_attempts": 1,
            "judge_feedback_history": []
        }
        
    # Step 5: The Evaluator-Optimizer Loop (RAG_LOOP)
    max_retries = 2
    attempts = 0
    feedback_history = []
    current_context = context
    
    while attempts < max_retries:
        attempts += 1
        
        # Student Drafts Answer
        draft = generate_student_draft(query, current_context)
        
        # Judge Evaluates
        evaluation = evaluate_draft(query, context, draft)
        feedback_history.append(evaluation)
        
        # If approved, break the loop and return the winning draft!
        if evaluation.get("is_approved"):
            return {
                "answer": draft,
                "route_taken": route,
                "student_attempts": attempts,
                "judge_feedback_history": feedback_history
            }
            
        # If rejected, append the Judge's feedback to the context so the Student learns
        current_context = context + f"\n\n[SYSTEM NOTE: Your previous answer was rejected. Judge's Feedback: {evaluation.get('feedback')}. Please correct this.]"
        
    # Step 6: Safe Fallback (If the loop maxes out)
    return {
        "answer": "I cannot answer this confidently based on the provided documents. The system rejected the drafts for safety.",
        "route_taken": "FALLBACK",
        "student_attempts": attempts,
        "judge_feedback_history": feedback_history
    }