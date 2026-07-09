import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.db import Base

class Document(Base):
    """Stores metadata about uploaded PDF files."""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(String, index=True)
    filename = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to the chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    """Stores the extracted text and the 768D vector embeddings."""
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    chunk_index = Column(Integer)
    text_content = Column(Text)
    
    # 768 Dimensions matching our Matryoshka-optimized Gemini output
    embedding = Column(Vector(768)) 

    document = relationship("Document", back_populates="chunks")

class EvaluationLog(Base):
    """The core observability table for our Evaluator-Optimizer loop."""
    __tablename__ = "evaluation_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(String, index=True)
    user_query = Column(Text)
    
    # Track which path the classifier chose: 'SIMPLE_QA', 'REASONING', 'RAG_LOOP'
    route_taken = Column(String) 
    
    # Track how many times the Student had to try (max 2)
    student_attempts = Column(Integer, default=1)
    final_answer = Column(Text)
    was_successful = Column(Boolean, default=False)
    
    # Store the exact critiques the Pro Judge gave the Student
    judge_feedback_history = Column(JSONB, default=list) 
    created_at = Column(DateTime, default=datetime.utcnow)