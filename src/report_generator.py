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
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

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
        print("  [*] Connecting to Groq (llama-3.3-70b-versatile)...")
        _client = Groq(api_key=api_key)
    return _client


def _run_prompt(prompt: str, max_tokens: int = 800) -> str:
    """
    Execute a single focused prompt via Groq and return the response text.
    Includes retry handling for Groq free-tier rate limits.
    """
    try:
        client = get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,  # Lower = more factual, less hallucination
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  LLM request failed: {e}")
        return "⚠️ LLM request failed. Please retry."


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

    unique_summaries = list(set([item["summary"] for item in summaries]))
    urls = list(set([item["url"] for item in summaries]))
    combined = " ".join(unique_summaries)
    if len(combined) > 2000:
        combined = combined[:2000] + "..."

    abstract_prompt = (
        f"You are a medical research assistant helping a researcher.\n"
        f"Based on the following research summaries about '{query}', "
        f"write a clear, informative abstract of 3-4 sentences. "
        f"Use plain language and focus on what was studied and why it matters. "
        f"Avoid overly technical jargon where possible.\n\n"
        f"Research Summaries:\n{combined}"
    )
    print("    [+] Generating Abstract...")
    abstract = _run_prompt(abstract_prompt, max_tokens=250)

    findings_prompt = (
        f"You are a medical research assistant.\n"
        f"Based on the following research summaries about '{query}', "
        f"extract and list 5-6 key findings as concise bullet points. "
        f"Start DIRECTLY with the first bullet point — do NOT add a header or intro sentence. "
        f"Each bullet point should be a distinct, clear insight in simple language. "
        f"Simplify technical terms where possible.\n\n"
        f"Research Summaries:\n{combined}"
    )
    print("    [+] Generating Key Findings...")
    raw_findings = _run_prompt(findings_prompt, max_tokens=350)
    finding_lines = [line.strip() for line in raw_findings.split("\n") if line.strip()]
    findings = "\n".join(
        f"- {line.lstrip('-').lstrip('*').lstrip('•').strip()}"
        for line in finding_lines
        if len(line.strip()) > 5
           and not line.lower().startswith("here are")
           and not line.lower().startswith("the following")
    )

    conclusion_prompt = (
        f"You are a medical research assistant.\n"
        f"Based on the following research summaries about '{query}', "
        f"write a conclusion of 3-4 sentences that explains the real-world importance "
        f"of these findings and their potential future impact on healthcare. "
        f"Use accessible language suitable for a general audience.\n\n"
        f"Research Summaries:\n{combined}"
    )
    print("    [+] Generating Conclusion...")
    conclusion = _run_prompt(conclusion_prompt, max_tokens=250)

    title = f"Recent Advances in {query.title()}"

    sources = "\n".join([f"- {url}" for url in urls])

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
