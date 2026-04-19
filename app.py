import streamlit as st
import re
import os
from dotenv import load_dotenv

# Import the LangGraph agent
from src.agent_graph import run_agent

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
    
    # Split using known headers from the formatting prompt
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

# UI Header Section
st.title("🩺 ResearchScope NLP Healthcare")
st.subheader("AI-powered Healthcare Research Assistant")
st.warning("⚠️ Only healthcare-related queries are supported")

# Input Section
query = st.text_input("Enter your research query:", placeholder="e.g. latest cancer treatment research")

# Execution Flow
if st.button("Analyze Research", type="primary"):
    
    # Validation
    query = query.strip().lstrip(">").strip()
    if not query:
        st.warning("Please enter a research query.")
    else:
        # Progress Indicators
        with st.status("Processing query...", expanded=True) as status:
            st.write("🔍 Searching medical sources...")
            st.write("📄 Extracting content...")
            st.write("🧠 Running NLP analysis...")
            st.write("🤖 Generating report...")
            
            try:
                # 1. Call Backend
                result = run_agent(query)
                report = result.get("report", "")
                
                # 2. Output Handling
                if report.startswith("❌") or "Report generation failed" in report or "Insufficient valid text" in report:
                    status.update(label="Analysis Failed", state="error", expanded=True)
                    st.error(report)
                elif "⚠️ LLM request failed" in report:
                    status.update(label="API Error", state="error", expanded=True)
                    st.error("LLM Generation failed due to rate limits or API errors. Please retry in a moment.")
                else:
                    status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                    
                    # 3. Output Display
                    st.divider()
                    
                    parsed_sections = parse_report(report)
                    
                    if parsed_sections and "Title" in parsed_sections:
                        # Display parsed structured sections
                        st.header(parsed_sections.get("Title", "Research Report"))
                        
                        st.subheader("📘 Abstract")
                        st.markdown(parsed_sections.get("Abstract", "No abstract available."))
                        
                        st.subheader("🔬 Key Findings")
                        st.markdown(parsed_sections.get("Key Findings", "No findings extracted."))
                        
                        st.subheader("📊 Conclusion")
                        st.markdown(parsed_sections.get("Conclusion", "No conclusion drawn."))
                        
                        st.subheader("🔗 Sources")
                        st.markdown(parsed_sections.get("Sources", "No sources provided."))
                    else:
                        # Fallback to display raw output if parsing failed
                        st.write("Report generated, but structured parsing failed. Raw output below:")
                        st.markdown(report)
                        
            except Exception as e:
                status.update(label="Error processing query", state="error", expanded=True)
                st.error(f"An unexpected error occurred: {str(e)}")