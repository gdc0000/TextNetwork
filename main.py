#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit Text Network and Topic Modeling App

This app allows you to upload an Excel file containing text data, extracts hashtags,
builds a text network (with centrality metrics), performs topic modeling using NMF,
and provides downloadable outputs for further analysis.

Requirements:
  - streamlit
  - pandas
  - openpyxl
  - xlsxwriter
  - scikit-learn
  - networkx

Run the app with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
import re
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF

# ---------------------------
# Helper Functions
# ---------------------------
def load_excel_file(file) -> pd.DataFrame:
    """
    Loads an Excel file from an uploaded file-like object.
    """
    try:
        df = pd.read_excel(file, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None


def extract_hashtags(text) -> str:
    """
    Extracts hashtags from a text string and returns them as a space-separated string.
    """
    if pd.isnull(text):
        return ""
    # Extract hashtags without the '#' symbol
    hashtags = re.findall(r'#(\w+)', str(text))
    return " ".join(hashtags)


def preprocess_documents(df: pd.DataFrame, text_column: str) -> pd.Series:
    """
    Applies hashtag extraction to the specified text column.
    """
    if text_column not in df.columns:
        st.error(f"Column '{text_column}' not found in the DataFrame.")
        return None
    documents = df[text_column].apply(extract_hashtags)
    return documents


def build_text_network(documents: pd.Series) -> nx.Graph:
    """
    Builds a text network using CountVectorizer to compute a document-term matrix,
    then creating an adjacency matrix and a networkx Graph.
    """
    cv = CountVectorizer()
    X = cv.fit_transform(documents)
    words = cv.get_feature_names_out()
    # Compute adjacency matrix (co-occurrence)
    adj_matrix = (X.T * X).toarray()
    adj_df = pd.DataFrame(adj_matrix, index=words, columns=words)
    G = nx.from_pandas_adjacency(adj_df)
    return G


def compute_graph_centralities(G: nx.Graph) -> None:
    """
    Computes and assigns degree, eigenvector, closeness, and betweenness centralities to graph nodes.
    """
    degree_dict = dict(G.degree())
    eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000)
    closeness_dict = nx.closeness_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G)

    nx.set_node_attributes(G, degree_dict, 'degree')
    nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
    nx.set_node_attributes(G, closeness_dict, 'closeness')
    nx.set_node_attributes(G, betweenness_dict, 'betweenness')


def perform_topic_modeling(documents: pd.Series,
                           n_topics: int = 10,
                           tfidf_params: dict = None,
                           nmf_params: dict = None):
    """
    Performs topic modeling using TF-IDF vectorization and NMF.
    Returns document-topic and word-topic dataframes along with raw matrices.
    """
    if tfidf_params is None:
        tfidf_params = {
            "ngram_range": (1, 3),
            "max_features": 1000,
            "min_df": 5,
            "max_df": 0.9
        }
    if nmf_params is None:
        nmf_params = {
            "n_components": n_topics,
            "beta_loss": 'kullback-leibler',
            "solver": 'mu',
            "max_iter": 1000,
            "alpha_H": 0.1,
            "l1_ratio": 0.5
        }

    tfidf = TfidfVectorizer(**tfidf_params)
    V = tfidf.fit_transform(documents)
    feature_names = tfidf.get_feature_names_out()

    model = NMF(**nmf_params)
    W = model.fit_transform(V)
    H = model.components_

    # Create DataFrames for topics
    word_topic_df = pd.DataFrame(H.T, index=feature_names)
    document_topic_df = pd.DataFrame(W)
    document_topic_df["Topic"] = document_topic_df.idxmax(axis=1)

    return document_topic_df, word_topic_df, W, H


def get_edge_list_csv(G: nx.Graph) -> bytes:
    """
    Creates a CSV (as bytes) of the network edge list, excluding self-loops.
    """
    edge_list = [(u, v) for u, v in G.edges() if u != v]
    edge_df = pd.DataFrame(edge_list, columns=['Source', 'Target'])
    return edge_df.to_csv(index=False).encode('utf-8')


def get_gephi_gexf(G: nx.Graph) -> bytes:
    """
    Exports the network graph to a GEXF format (as bytes) for use in Gephi.
    """
    buffer = BytesIO()
    nx.write_gexf(G, buffer)
    return buffer.getvalue()


def get_topic_modeling_excel(original_df: pd.DataFrame,
                             document_topic_df: pd.DataFrame,
                             word_topic_df: pd.DataFrame) -> bytes:
    """
    Exports topic modeling results into an Excel file (as bytes) with two sheets:
    'documents' (original data with topic assignments) and 'words' (topic weights per word).
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.concat([original_df, document_topic_df], axis=1).to_excel(writer, sheet_name='documents', index=False)
        word_topic_df.to_excel(writer, sheet_name='words')
    return output.getvalue()

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("Text Network and Topic Modeling Analysis")

# Sidebar Inputs
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])
text_column = st.sidebar.text_input("Text column name", value="text")
n_topics = st.sidebar.number_input("Number of Topics", value=10, step=1)

if uploaded_file is not None:
    df = load_excel_file(uploaded_file)
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())

        if text_column not in df.columns:
            st.error(f"Column '{text_column}' not found in the uploaded file.")
        else:
            # Step 1: Preprocess Documents
            documents = preprocess_documents(df, text_column)
            st.subheader("Hashtag Extraction Sample")
            st.write(documents.head())

            # Step 2: Build Text Network and Compute Centralities
            G = build_text_network(documents)
            compute_graph_centralities(G)
            st.subheader("Network Summary")
            st.write(f"Number of nodes: {G.number_of_nodes()}")
            st.write(f"Number of edges: {G.number_of_edges()}")

            # Download buttons for network outputs
            edge_csv = get_edge_list_csv(G)
            gephi_gexf = get_gephi_gexf(G)
            st.download_button("Download Edge List CSV", data=edge_csv,
                               file_name="Text_Network_Edge_List.csv", mime="text/csv")
            st.download_button("Download Gephi GEXF", data=gephi_gexf,
                               file_name="Text_Network_Gephi_Results.gexf", mime="application/gexf+xml")

            # Step 3: Topic Modeling
            document_topic_df, word_topic_df, _, _ = perform_topic_modeling(documents, n_topics=n_topics)
            st.subheader("Topic Modeling Completed")
            st.write("Document-Topic assignments (first 5 rows):")
            st.dataframe(document_topic_df.head())

            topic_excel = get_topic_modeling_excel(df, document_topic_df, word_topic_df)
            st.download_button("Download Topic Modeling Results (Excel)", data=topic_excel,
                               file_name="NMF_Results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload an Excel file to begin analysis.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
st.markdown("""
[GitHub](https://github.com/gdc0000) | 
[ORCID](https://orcid.org/0000-0002-1439-5790) | 
[LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
""")
