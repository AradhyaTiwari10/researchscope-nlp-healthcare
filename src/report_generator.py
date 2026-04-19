"""
report_generator.py
-------------------
Module: LLM Report Generation
Phase 5 – ResearchScope NLP Healthcare

Responsibilities:
  - Generate a structured research report using summaries from multiple sources.
  - Utilize HuggingFace Flan-T5 model.
  - Safely combine input texts and manually map source attributions.

Public API:
  generate_report(query: str, summaries: list) -> str
"""

from transformers import pipeline

generator = None

def get_generator():
    global generator
    if generator is None:
        print("  [*] Loading HuggingFace model (Flan-T5-base)...")
        # Let pipeline auto-infer the task type from the model architecture to prevent deprecated task string errors
        generator = pipeline(
            model="google/flan-t5-base",
            max_new_tokens=400
        )
    return generator

def generate_report(query: str, summaries: list) -> str:
    """
    Generate a formatted report answering the user query grounded directly 
    by summary intelligence.
    
    Args:
        query (str): The initial user query context.
        summaries (list): Extracted NLP constraints: [{"url": "...", "summary": "..."}]
        
    Returns:
        str: Multiline string containing the final report with source links appended.
    """
    if not summaries:
        return "Insufficient valid text extracted to generate a report."
        
    # 1. Collate Texts
    combined_texts = []
    urls = []
    
    unique_summaries = list(set([item["summary"] for item in summaries]))
    
    for idx, text in enumerate(unique_summaries):
        combined_texts.append(f"{idx+1}. {text}")
        
    for item in summaries:
        if item["url"] not in urls:
            urls.append(item["url"])
            
    # Combine securely
    combined_context = "\n".join(combined_texts)
    
    # Truncation safety: Flan-T5 restricts max token capacity to 512 normally
    # We clip rough characters (1500 chars ~ 350-400 tokens) to ensure prompt fits
    if len(combined_context) > 1500:
        combined_context = combined_context[:1500] + "..."
        
    # 2. Design Prompt
    prompt = (
        f"Given the following research summaries about: {query}\n\n"
        f"Summaries:\n{combined_context}\n\n"
        "Generate the report STRICTLY in this format:\n\n"
        "Title:\n"
        "...\n\n"
        "Abstract:\n"
        "...\n\n"
        "Key Findings:\n"
        "- ...\n"
        "- ...\n\n"
        "Conclusion:\n"
        "...\n\n"
        "Keep it factual and concise based ONLY on the summaries provided."
    )
    
    # 3. Generate Text via LLM
    print("  [*] Generating formal structure via LLM...")
    gen = get_generator()
    
    response = gen(prompt)
    generated_text = response[0]["generated_text"].strip()
    
    if not generated_text or len(generated_text) < 50:
        return "Report generation failed. Insufficient structured output."
    
    # 4. Append Authentic Manual Sources
    source_append = "\n\nSources:\n"
    for url in urls:
        source_append += f"* {url}\n"
        
    return generated_text + source_append
