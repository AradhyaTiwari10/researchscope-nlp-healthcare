import streamlit as st
import re
import time
from dotenv import load_dotenv

# Import the core NLP pipeline functions for manual execution
from src.web_search import search_web
from src.content_extractor import extract_multiple
from src.nlp_processor import process_articles
from src.report_generator import generate_report
from src.utils import is_medical_query

# Auto-load environment variables
load_dotenv()

st.set_page_config(
    page_title="ResearchScope NLP Healthcare", 
    page_icon="🩺",
    layout="wide"
)

def parse_report(report_text):
    """Safely parse the sections of the text report."""
    sections = {}
    try:
        parts = re.split(r'(Title:|Abstract:|Key Findings:|Conclusion:|Sources:)', report_text)
        current_header = None
        for part in parts:
            part_strip = part.strip()
            if part_strip in ["Title:", "Abstract:", "Key Findings:", "Conclusion:", "Sources:"]:
                current_header = part_strip.replace(":", "")
            elif current_header and part_strip:
                sections[current_header] = part_strip
    except Exception:
        pass
    return sections

# Custom Premium CSS Inject
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800;
        color: #0F172A;
        text-align: center;
        margin-bottom: -15px;
        font-family: 'Inter', sans-serif;
    }
    .sub-title {
        font-size: 1.25rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Modern UI Header
st.markdown('<div class="main-title">🩺 ResearchScope AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous Medical NLP Research Assistant</div>', unsafe_allow_html=True)

# Handle Interactive Step-By-Step State
if "step" not in st.session_state:
    st.session_state.step = 0
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Premium Example Queries (Pill Layout)
st.markdown("""
<div class="glass-card" style="text-align: center; background-color: #F8FAFC;">
    <span style="color: #475569; font-weight: 600; margin-right: 15px;">💡 Try an example query:</span>
    <span style="background: #E0F2FE; color: #0284C7; padding: 6px 14px; border-radius: 20px; font-size: 0.9em; margin-right: 10px; font-weight: 500;">AI in cancer detection</span>
    <span style="background: #E0F2FE; color: #0284C7; padding: 6px 14px; border-radius: 20px; font-size: 0.9em; margin-right: 10px; font-weight: 500;">Diabetes treatment research</span>
    <span style="background: #E0F2FE; color: #0284C7; padding: 6px 14px; border-radius: 20px; font-size: 0.9em; font-weight: 500;">Infectious disease prevention WHO</span>
</div>
""", unsafe_allow_html=True)

col_input, col_clear = st.columns([8, 1])
with col_input:
    query = st.text_input("Enter your research query:", placeholder="e.g. latest cancer treatment research")
with col_clear:
    st.write("")
    st.write("")
    if st.button("Reset Pipeline"):
        st.session_state.step = 0
        st.session_state.cache = {}
        st.session_state.current_query = ""
        st.rerun()

# Reset state if the query changes
if query != st.session_state.current_query and query != "":
    st.session_state.step = 0
    st.session_state.cache = {}
    st.session_state.current_query = query

if query:
    query_cleaned = query.strip().lstrip(">").strip()
    
    # Check Scope First
    if not is_medical_query(query_cleaned):
        st.error("❌ OUT OF SCOPE: The ResearchScope agent is specialized for medical and healthcare research. Please enter a valid query.")
        st.stop()

    st.divider()
    
    # ==========================================
    # STEP 1: Search & Retrieval
    # ==========================================
    col1_th, col1_act = st.columns([1, 1.5])
    with col1_th:
        st.markdown("### 1. Retrieval (RAG)")
        st.markdown("**🔍 Theory & Tools:** We use **DuckDuckGo Search (`ddgs`)** to query the live internet. We enforce a deterministic **ScopeGuard Algorithm** that filters out generic sites and guarantees strict inclusion of verifiable medical domains (like *NIH, WHO, Medical News Today*). This prevents the AI from encountering fake news.")
        if st.session_state.step == 0:
            if st.button("🔍 Search medical sources...", type="primary"):
                with st.spinner("Searching..."):
                    st.session_state.cache["search_results"] = search_web(query_cleaned)
                    st.session_state.step = 1
                    st.rerun()
    with col1_act:
        if st.session_state.step >= 1:
            st.success("✅ Search Complete")
            results = st.session_state.cache.get("search_results", [])
            st.write(f"**Found {len(results)} trusted sources:**")
            for res in results:
                st.write(f"- 🔗 {res['url']}")
                
    # ==========================================
    # STEP 2: Content Extraction
    # ==========================================
    if st.session_state.step >= 1:
        st.divider()
        col2_th, col2_act = st.columns([1, 1.5])
        with col2_th:
            st.markdown("### 2. Payload Extraction")
            st.markdown("**📄 Theory & Tools:** Raw HTML is messy and full of ads. We use the **`newspaper3k`** library to aggressively parse the website's DOM structure. By simulating HTTP headers to bypass bot-checks, it isolates pure academic body text while stripping away useless navigation bars that would bloat the AI memory.")
            if st.session_state.step == 1:
                if st.button("📄 Extract content...", type="primary"):
                    with st.spinner("Downloading HTML and extracting text..."):
                        st.session_state.cache["texts"] = extract_multiple(st.session_state.cache["search_results"])
                        st.session_state.step = 2
                        st.rerun()
        with col2_act:
            if st.session_state.step >= 2:
                st.success("✅ Extraction Complete")
                texts = st.session_state.cache.get("texts", [])
                for txt in texts:
                    count = len(txt['text'].split())
                    st.write(f"- Downloaded **{count} words** from {txt['url'].split('//')[-1].split('/')[0]}")

    # ==========================================
    # STEP 3: NLP Operations (TF-IDF & LDA)
    # ==========================================
    if st.session_state.step >= 2:
        st.divider()
        col3_th, col3_act = st.columns([1, 1.5])
        with col3_th:
            st.markdown("### 3. NLP Analysis")
            st.markdown("**🧠 Theory & Tools:** Before using LLMs, we mathematically reduce the text to prevent 'hallucinations'. Using **Scikit-Learn**, we build a **TF-IDF Vector Matrix** to score word importance. This creates **Extractive Summaries**. We also use **LDA (Latent Dirichlet Allocation)** for unsupervised machine learning to find hidden Topic Clusters.")
            if st.session_state.step == 2:
                if st.button("🧠 Run NLP analysis...", type="primary"):
                    with st.spinner("Building TF-IDF Matrices and running LDA..."):
                        summaries, topics = process_articles(st.session_state.cache["texts"])
                        st.session_state.cache["summaries"] = summaries
                        st.session_state.cache["topics"] = topics
                        st.session_state.step = 3
                        st.rerun()
        with col3_act:
            if st.session_state.step >= 3:
                st.success("✅ NLP Processing Complete")
                topics = st.session_state.cache.get("topics", [])
                st.write("**Discovered Topic Clusters (Latent Themes):**")
                for idx, words in topics:
                    st.write(f"- **Cluster {idx+1}:** {', '.join(words)}")

    # ==========================================
    # STEP 4: LLM Synthesis
    # ==========================================
    if st.session_state.step >= 3:
        st.divider()
        col4_th, col4_act = st.columns([1, 1.5])
        with col4_th:
            st.markdown("### 4. Generative Synthesis")
            st.markdown("**🤖 Theory & Tools:** We feed the dense mathematical summaries into **Groq's LLaMA-3.3-Versatile LPU**. We use an architecture called **Prompt Decomposition**. Instead of 1 giant prompt, the agent asks 3 focused questions to generate the Abstract, Findings, and Conclusion in a perfectly formatted structure.")
            if st.session_state.step == 3:
                if st.button("🤖 Generate report...", type="primary"):
                    with st.spinner("Synthesizing final report with Groq API..."):
                        start_time = time.time()
                        report = generate_report(query_cleaned, st.session_state.cache["summaries"])
                        st.session_state.cache["exec_time"] = time.time() - start_time
                        st.session_state.cache["report"] = report
                        st.session_state.step = 4
                        st.rerun()
        with col4_act:
            if st.session_state.step >= 4:
                st.success(f"✅ Generated in {st.session_state.cache.get('exec_time', 0):.2f} seconds!")
                
                # --- RENDER FINAL PDF EXPORT AND REPORT ---
                st.divider()
                report = st.session_state.cache.get("report", "")
                parsed = parse_report(report)
                
                st.header("Medical Context Report")
                
                # PDF GENERATION
                topics_disp = "\n".join([f"Cluster {idx+1}: {', '.join(words)}" for idx, words in st.session_state.cache.get("topics", [])])
                try:
                    from fpdf import FPDF
                    def generate_pdf(rep_text, top_text):
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.set_font("Arial", size=11)
                        rep_clean = rep_text.replace("📘", "").replace("🔬", "").replace("📊", "").replace("🔗", "")
                        clean_text = (rep_clean + "\n\n--- NLP TOPIC CLUSTERS ---\n" + top_text).encode("latin-1", "replace").decode("latin-1")
                        for line in clean_text.split('\n'):
                            pdf.multi_cell(0, 7, txt=line)
                        return pdf.output(dest="S").encode("latin-1")
                    
                    pdf_bytes = generate_pdf(report, topics_disp)
                    st.download_button("📄 Download Complete Report as PDF", data=pdf_bytes, file_name="researchscope_report.pdf", mime="application/pdf")
                except:
                    pass

                # RENDER TEXT
                if parsed and "Title" in parsed:
                    st.markdown(f"*{parsed.get('Title', '')}*")
                    st.subheader("📘 Abstract")
                    st.markdown(parsed.get("Abstract", ""))
                    st.subheader("🔬 Key Findings")
                    st.markdown(parsed.get("Key Findings", ""))
                    st.subheader("📊 Conclusion")
                    st.markdown(parsed.get("Conclusion", ""))
                    st.subheader("🔗 Verified Sources")
                    st.markdown(parsed.get("Sources", ""))
                else:
                    st.markdown(report)