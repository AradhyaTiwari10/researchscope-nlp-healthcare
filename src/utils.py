"""
utils.py
--------
Utility functions for the ResearchScope pipeline.
"""

def is_medical_query(query: str) -> bool:
    """
    Check if a query is related to healthcare, medicine, or medical research.
    Uses a broad keyword-based intent classification.
    """
    medical_keywords = [
        "cancer", "treatment", "medicine", "medical", "healthcare", "health", 
        "clinical", "trial", "research", "drug", "vaccine", "disease", "patient",
        "symptom", "diagnosis", "therapy", "surgery", "breakthrough", "cardiology",
        "oncology", "biology", "genetic", "virus", "infection", "pharmacy",
        "pharmaceutical", "anatomy", "physiology", "hospital", "doctor", "nurse",
        "physician", "neurology", "brain", "heart", "lung", "kidney", "liver",
        "diabetes", "covid", "hiv", "aids", "tumor", "biotech"
    ]
    
    query_lower = query.lower()
    return any(word in query_lower for word in medical_keywords)
