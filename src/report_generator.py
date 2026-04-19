"""
report_generator.py
-------------------
Module: LLM Report Generation
Phase 5 – ResearchScope NLP Healthcare

Responsibilities:
  - Generate a structured research report using summaries from multiple sources.
  - Utilizes Groq API with llama3-70b-8192 for high-quality, fast inference.
  - Uses multi-step prompt decomposition for richer, structured outputs.

Public API:
  generate_report(query: str, summaries: list) -> str
"""

import os
from groq import Groq

_client = None

def get_client():
    """Lazy-load Groq client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "\n\n  ❌ GROQ_API_KEY is not set!\n"
                "  Please set it in your terminal before running:\n\n"
                "      export GROQ_API_KEY='your_key_here'\n\n"
                "  Get a free key at: https://console.groq.com/keys\n"
            )
        print("  [*] Connecting to Groq (llama3-70b-8192)...")
        _client = Groq(api_key=api_key)
    return _client


def _run_prompt(prompt: str, max_tokens: int = 400) -> str:
    """
    Execute a single focused prompt via Groq and return the response text.
    
    Args:
        prompt: The instruction to send to the LLM.
        max_tokens: Max output length.
        
    Returns:
        str: The LLM's generated text.
    """
    client = get_client()
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4,  # Slightly creative but factual
    )
    return response.choices[0].message.content.strip()


def generate_report(query: str, summaries: list) -> str:
    """
    Generate a full structured research report using multi-step prompt decomposition.

    Each section (Abstract, Key Findings, Conclusion) is independently generated
    by a focused prompt, then assembled manually for guaranteed structure.

    Args:
        query (str): The user's research query.
        summaries (list): NLP summaries: [{"url": "...", "summary": "..."}]

    Returns:
        str: Fully formatted, multi-section research report.
    """
    if not summaries:
        return "Insufficient valid text extracted to generate a report."

    print("  [*] Generating formal structure via LLM (Groq)...")

    # 1. Prepare shared context
    unique_summaries = list(set([item["summary"] for item in summaries]))
    urls = list(set([item["url"] for item in summaries]))
    combined = " ".join(unique_summaries)
    if len(combined) > 2000:
        combined = combined[:2000] + "..."

    # 2. Generate ABSTRACT
    abstract_prompt = (
        f"You are a medical research assistant helping a researcher.\n"
        f"Based on the following research summaries about '{query}', "
        f"write a clear, informative abstract of 3-4 sentences. "
        f"Use plain language and focus on what was studied and why it matters.\n\n"
        f"Research Summaries:\n{combined}"
    )
    print("    [+] Generating Abstract...")
    abstract = _run_prompt(abstract_prompt, max_tokens=250)

    # 3. Generate KEY FINDINGS
    findings_prompt = (
        f"You are a medical research assistant.\n"
        f"Based on the following research summaries about '{query}', "
        f"extract and list 5-6 key findings as concise bullet points. "
        f"Each bullet point should be a distinct, clear insight in simple language.\n\n"
        f"Research Summaries:\n{combined}"
    )
    print("    [+] Generating Key Findings...")
    raw_findings = _run_prompt(findings_prompt, max_tokens=350)
    # Normalize bullet point formatting regardless of model output style
    finding_lines = [line.strip() for line in raw_findings.split("\n") if line.strip()]
    findings = "\n".join(
        f"- {line.lstrip('-').lstrip('*').lstrip('•').strip()}"
        for line in finding_lines
        if len(line.strip()) > 5
    )

    # 4. Generate CONCLUSION
    conclusion_prompt = (
        f"You are a medical research assistant.\n"
        f"Based on the following research summaries about '{query}', "
        f"write a conclusion of 3-4 sentences that explains the real-world importance "
        f"of these findings and their potential future impact on healthcare.\n\n"
        f"Research Summaries:\n{combined}"
    )
    print("    [+] Generating Conclusion...")
    conclusion = _run_prompt(conclusion_prompt, max_tokens=250)

    # 5. Auto-generate title from query
    title = f"Recent Advances in {query.title()}"

    # 6. Format sources
    sources = "\n".join([f"- {url}" for url in urls])

    # 7. Assemble full report
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
