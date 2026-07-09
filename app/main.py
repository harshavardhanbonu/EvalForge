from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import os

from app.core.db import engine, Base, get_db

# Import your routers
from app.api import routes_ingest
from app.api import routes_query

# Initialize FastAPI
app = FastAPI(
    title="EvalForge API", 
    description="Adaptive RAG Evaluation Platform",
    version="1.0.0"
)

# CORS Middleware (Allows your frontend to talk to the backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API Routes
app.include_router(routes_ingest.router, prefix="/api/v1/documents", tags=["Ingestion"])
app.include_router(routes_query.router, prefix="/api/v1/query", tags=["Query"])

@app.on_event("startup")
def on_startup():
    """
    Runs when the server starts. 
    Ensures the pgvector extension is installed in Postgres, then creates tables.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    
    # Reset tables on startup (Useful for local development)
    Base.metadata.drop_all(bind=engine)
    
    # Create all tables defined in db_models.py
    Base.metadata.create_all(bind=engine)


@app.get("/")
def serve_ui():
    """Serves the beautiful HTML UI when you visit the root URL."""
    # Find the path to the frontend/index.html file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "frontend", "index.html")
    
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "UI file not found. Please ensure frontend/index.html exists."}


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """
    Simple health check to verify API and Database connectivity.
    """
    try:
        db.execute(text("SELECT 1;"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")