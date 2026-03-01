import PyPDF2
import re

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            
    # Try to start at Abstract
    abstract_match = re.search(r'\b(?:ABSTRACT|Abstract)\b', text)
    if abstract_match:
        text = text[abstract_match.start():]
        
    # Try to end after Introduction / before Methods or References
    # We start searching 500 characters after the start so we don't match the intro header itself too closely
    search_start = min(500, len(text))
    end_pattern = r'\n\s*(?:[IVX0-9]+\.?\s*)?(?:Methods|Methodology|Materials and Methods|Background|Related Work|Literature Review|References|REFERENCES|Bibliography)\b'
    end_match = re.search(end_pattern, text[search_start:], re.IGNORECASE)
    
    if end_match:
        text = text[:search_start + end_match.start()]
        
    return text