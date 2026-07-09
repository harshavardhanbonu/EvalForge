import fitz  # PyMuPDF
from typing import List

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts all text from a PDF file byte stream.
    """
    try:
        # Open the PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            # Extract text from each page
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")

def get_sliding_window_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits a large text block into smaller chunks using a sliding window.
    The overlap ensures context (like pronouns or mid-sentence cuts) is not lost.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only append if the chunk actually contains text
        if chunk.strip():
            chunks.append(chunk.strip())
            
        # Move the starting point forward, but step back by the overlap amount
        start += (chunk_size - overlap)

    return chunks