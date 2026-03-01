"""
Document Ingestion and Corpus Assembly Module.

This module is responsible for loading PDF research papers, extracting their 
textual content, and preparing a unified corpus for traditional NLP analysis.
We use pdfplumber for high-fidelity text extraction from academic formats.
"""

import os
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of a PDF, discarding empty or unreadable pages.
    Ensures robustness by processing page-by-page.
    """
    full_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                # Remove empty pages or those with only whitespace
                if page_text and page_text.strip():
                    full_text.append(page_text.strip())
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

    # Return documents as single space-joined string for the PDF
    return " ".join(full_text)

def load_dataset(folder_path):
    """
    Iterates through a folder, extracts text from all available PDFs, 
    and returns a list of processed documents and a combined corpus string.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    documents = []
    
    # Sort files to ensure deterministic behavior across runs
    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
    
    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        content = extract_text_from_pdf(path)
        if content:
            documents.append(content)
            
    combined_corpus = " ".join(documents)
    
    return documents, combined_corpus

if __name__ == "__main__":
    # Internal test for local verification
    RAW_PATH = os.path.join(os.path.dirname(__file__), "../data/raw")
    docs, corpus = load_dataset(RAW_PATH)
    print(f"Extraction complete: Total Documents = {len(docs)}")
    print(f"Corpus size (characters): {len(corpus)}")