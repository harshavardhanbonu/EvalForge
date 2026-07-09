from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import uuid

from app.core.db import get_db
from app.models.db_models import Document, DocumentChunk
from app.models.schemas import DocumentResponse
from app.utils.pdf_parser import extract_text_from_pdf, get_sliding_window_chunks
from app.services.embeddings import generate_embeddings

router = APIRouter()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to upload a PDF document, chunk it, embed the chunks, and store them in pgvector.
    """
    # 1. Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # 2. Read and extract text
        pdf_bytes = await file.read()
        text_content = extract_text_from_pdf(pdf_bytes)
        
        if not text_content:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")
            
        # 3. Chunk the text
        chunks = get_sliding_window_chunks(text_content, chunk_size=1000, overlap=200)
        
        # 4. Generate embeddings (768D)
        embeddings = generate_embeddings(chunks)
        
        # 5. Save to Database
        # Create Document record
        new_doc = Document(
            id=uuid.uuid4(),
            project_id=project_id,
            filename=file.filename
        )
        db.add(new_doc)
        db.flush() # Get the new_doc.id before committing to use in the chunks
        
        # Create Chunk records
        db_chunks = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            new_chunk = DocumentChunk(
                id=uuid.uuid4(),
                document_id=new_doc.id,
                chunk_index=i,
                text_content=chunk_text,
                embedding=embedding
            )
            db_chunks.append(new_chunk)
            
        db.add_all(db_chunks)
        db.commit()
        db.refresh(new_doc)
        
        return DocumentResponse(
            id=new_doc.id,
            project_id=new_doc.project_id,
            filename=new_doc.filename,
            message="Document successfully processed and embedded.",
            chunks_created=len(db_chunks)
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred during ingestion: {str(e)}")