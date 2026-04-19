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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = None
model = None

def get_generator():
    global tokenizer, model
    if model is None:
        print("  [*] Loading HuggingFace model (Flan-T5-base)...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

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
    prompt = f"""You are a medical research assistant.

Using the summaries below, generate a clear and structured report.

STRICT FORMAT:

Title:
<clear topic title>

Abstract:
Write 2-3 simple sentences explaining the topic in plain language.

Key Findings:
- Explain important insights in simple terms
- Avoid technical jargon
- Focus on meaning, not raw data

Conclusion:
Summarize overall importance and real-world impact

Query: {query}

Summaries:
{combined_context}

IMPORTANT:
- Simplify complex medical language
- Do NOT copy sentences directly
- Make it readable for a general audience
"""
    
    # 3. Generate Text via LLM
    print("  [*] Generating formal structure via LLM...")
    tok, mod = get_generator()
    
    inputs = tok(prompt, return_tensors="pt", max_length=1500, truncation=True)
    outputs = mod.generate(**inputs, max_new_tokens=400)
    generated_text = tok.decode(outputs[0], skip_special_tokens=True).strip()
    
    if not generated_text or len(generated_text) < 50:
        return "Report generation failed. Insufficient structured output."
    
    # 4. Append Authentic Manual Sources
    source_append = "\n\nSources:\n"
    for url in urls:
        source_append += f"* {url}\n"
        
    return generated_text + source_append
