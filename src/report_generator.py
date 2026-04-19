"""
report_generator.py
-------------------
Module: LLM Report Generation
Phase 5 – ResearchScope NLP Healthcare

Responsibilities:
  - Generate a structured research report using summaries from multiple sources.
  - Utilize HuggingFace Flan-T5 model via MULTI-STEP prompt decomposition.
  - Each report section (abstract, findings, conclusion) is generated independently
    to work within Flan-T5's tight token window constraints.

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


def _run_prompt(prompt: str) -> str:
    """
    Core inference helper. Runs a single focused prompt through Flan-T5.
    Uses a short max_length input and max_new_tokens to avoid context overflow.
    """
    tok, mod = get_generator()
    inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = mod.generate(**inputs, max_new_tokens=200)
    return tok.decode(outputs[0], skip_special_tokens=True).strip()


def generate_report(query: str, summaries: list) -> str:
    """
    Generate a research report using multi-step prompt decomposition.
    
    Instead of a single large prompt (which Flan-T5 compresses aggressively),
    we independently generate each section and assemble them manually.
    
    Args:
        query (str): The user's research query.
        summaries (list): NLP summaries: [{"url": "...", "summary": "..."}]
        
    Returns:
        str: Fully formatted multi-section report.
    """
    if not summaries:
        return "Insufficient valid text extracted to generate a report."

    print("  [*] Generating formal structure via LLM...")
    
    # 1. Prepare the shared context 
    unique_summaries = list(set([item["summary"] for item in summaries]))
    urls = list(set([item["url"] for item in summaries]))
    
    # Merged dense block — works better than numbered lists for Flan-T5
    combined = " ".join(unique_summaries)
    if len(combined) > 1200:
        combined = combined[:1200] + "..."

    # 2. Generate ABSTRACT
    abstract_prompt = (
        f"Summarize the following medical research into a clear abstract of 3-4 sentences. "
        f"Focus on what the research is about and why it matters:\n\n{combined}"
    )
    print("    [+] Generating Abstract...")
    abstract = _run_prompt(abstract_prompt)
    if not abstract or len(abstract.split()) < 10:
        abstract = "Research summaries provide insights into recent clinical advancements in this domain."

    # 3. Generate KEY FINDINGS  
    findings_prompt = (
        f"Based on the following medical research, list 5 key findings as short bullet points. "
        f"Each point should be a clear, simple insight:\n\n{combined}"
    )
    print("    [+] Generating Key Findings...")
    raw_findings = _run_prompt(findings_prompt)
    # Format bullet points neatly regardless of model output style
    finding_lines = [line.strip() for line in raw_findings.split("\n") if line.strip()]
    if finding_lines:
        findings = "\n".join(
            f"- {line.lstrip('-').lstrip('*').strip()}" for line in finding_lines
        )
    else:
        findings = f"- {raw_findings.strip()}"

    # 4. Generate CONCLUSION
    conclusion_prompt = (
        f"Write a 3-sentence conclusion about the importance of this medical research "
        f"and its future impact on healthcare:\n\n{combined}"
    )
    print("    [+] Generating Conclusion...")
    conclusion = _run_prompt(conclusion_prompt)
    if not conclusion or len(conclusion.split()) < 10:
        conclusion = "These findings represent an important step forward in improving patient outcomes and advancing clinical practice."

    # 5. Auto-generate Title
    title = f"Recent Advances in {query.title()}"

    # 6. Assemble Sources
    sources = "\n".join([f"- {url}" for url in urls])

    # 7. Assemble Final Report
    report = f"""Title:
{title}

Abstract:
{abstract}

Key Findings:
{findings}

Conclusion:
{conclusion}

Sources:
{sources}"""

    return report
