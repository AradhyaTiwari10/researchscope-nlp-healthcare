"""
Visualization components for the Streamlit UI.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_global_keywords(global_terms):
    """
    Plots the global keywords using seaborn and matplotlib.
    """
    df_global = pd.DataFrame(global_terms, columns=["Term", "Importance Score"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_global, x="Importance Score", y="Term", palette="Blues_r", ax=ax)
    plt.title("Most Significant Terms Across Corpus", fontsize=14, color="#1e293b")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)

def plot_document_keywords(doc_tfidf, feat_names):
    """
    Plots the top keywords for a specific document using streamlit native bar chart.
    """
    top_indices = doc_tfidf.argsort()[::-1][:10]
    doc_df = pd.DataFrame({"Term": feat_names[top_indices], "Score": doc_tfidf[top_indices]})
    st.bar_chart(doc_df.set_index("Term"))
