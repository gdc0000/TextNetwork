#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit App for Text Network and Topic Modeling Analysis

This app allows you to upload an Excel file containing text data, then:
  1. Preprocess the text (extract hashtags)
  2. Build a text network with customizable CountVectorizer parameters and compute centralities
  3. Perform topic modeling using customizable TF-IDF and NMF parameters

Downloadable outputs include an edge list CSV, a Gephi GEXF file, and an Excel file for topic modeling results.

To run:
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


def build_text_network(documents: pd.Series, cv_params: dict = None) -> nx.Graph:
    """
    Builds a text network using CountVectorizer with given parameters,
    computes the co-occurrence adjacency matrix, and returns a networkx Graph.
    """
    if cv_params is None:
        cv_params = {}
    cv = CountVectorizer(**cv_params)
    X = cv.fit_transform(documents)
    words = cv.get_feature_names_out()
    adj_matrix = (X.T * X).toarray()
    adj_df = pd.DataFrame(adj_matrix, index=words, columns=words)
    G = nx.from_pandas_adjacency(adj_df)
    return G


def compute_graph_centralities(G: nx.Graph) -> None:
    """
    Computes and assigns centralities (degree, eigenvector, closeness, betweenness) to graph nodes.
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
# Session State Initialization
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "G" not in st.session_state:
    st.session_state.G = None
if "document_topic_df" not in st.session_state:
    st.session_state.document_topic_df = None
if "word_topic_df" not in st.session_state:
    st.session_state.word_topic_df = None

# ---------------------------
# Sidebar Configuration
# ---------------------------
st.sidebar.header("Data Upload & Basic Settings")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])
text_column = st.sidebar.text_input("Text column name", value="text")

st.sidebar.header("Text Network Analysis Options")
st.sidebar.subheader("CountVectorizer Settings")
cv_min_df = st.sidebar.number_input("min_df", value=1, step=1)
cv_max_df = st.sidebar.number_input("max_df", value=1.0, step=0.1)
cv_ngram_lower = st.sidebar.number_input("ngram_range lower", value=1, step=1)
cv_ngram_upper = st.sidebar.number_input("ngram_range upper", value=1, step=1)

st.sidebar.header("Topic Modeling Options")
st.sidebar.subheader("TF-IDF Settings")
tfidf_ngram_lower = st.sidebar.number_input("TFIDF ngram_range lower", value=1, step=1)
tfidf_ngram_upper = st.sidebar.number_input("TFIDF ngram_range upper", value=3, step=1)
tfidf_max_features = st.sidebar.number_input("max_features", value=1000, step=100)
tfidf_min_df = st.sidebar.number_input("min_df", value=5, step=1)
tfidf_max_df = st.sidebar.number_input("max_df", value=0.9, step=0.1)

st.sidebar.subheader("NMF Settings")
n_topics = st.sidebar.number_input("Number of Topics", value=10, step=1)
nmf_beta_loss = st.sidebar.selectbox("beta_loss", options=['kullback-leibler', 'frobenius'], index=0)
nmf_solver = st.sidebar.selectbox("solver", options=['mu', 'cd'], index=0)
nmf_max_iter = st.sidebar.number_input("max_iter", value=1000, step=100)
nmf_alpha_H = st.sidebar.number_input("alpha_H", value=0.1, step=0.1, format="%.2f")
nmf_l1_ratio = st.sidebar.number_input("l1_ratio", value=0.5, step=0.1, format="%.2f")

# ---------------------------
# Main App Layout
# ---------------------------
st.title("Text Network and Topic Modeling Analysis")

# ----- Step 1: Data Upload -----
st.header("1. Data Upload")
if uploaded_file is not None:
    st.session_state.df = load_excel_file(uploaded_file)
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
else:
    st.info("Please upload an Excel file from the sidebar.")

# ----- Step 2: Preprocessing -----
st.header("2. Preprocessing")
if st.session_state.df is not None:
    if st.button("Preprocess Data"):
        st.session_state.documents = preprocess_documents(st.session_state.df, text_column)
        if st.session_state.documents is not None:
            st.success("Preprocessing completed!")
            st.subheader("Sample Processed Text")
            st.write(st.session_state.documents.head())
else:
    st.info("Upload data to enable preprocessing.")

# ----- Step 3: Text Network Analysis -----
st.header("3. Text Network Analysis")
if st.session_state.documents is not None:
    cv_params = {
        "min_df": cv_min_df,
        "max_df": cv_max_df,
        "ngram_range": (cv_ngram_lower, cv_ngram_upper)
    }
    if st.button("Build Text Network"):
        st.session_state.G = build_text_network(st.session_state.documents, cv_params=cv_params)
        compute_graph_centralities(st.session_state.G)
        st.success("Text network built and centralities computed!")
        st.write(f"**Number of nodes:** {st.session_state.G.number_of_nodes()}")
        st.write(f"**Number of edges:** {st.session_state.G.number_of_edges()}")
        
        # Provide download buttons for network outputs
        edge_csv = get_edge_list_csv(st.session_state.G)
        gephi_gexf = get_gephi_gexf(st.session_state.G)
        st.download_button("Download Edge List CSV", data=edge_csv,
                           file_name="Text_Network_Edge_List.csv", mime="text/csv")
        st.download_button("Download Gephi GEXF", data=gephi_gexf,
                           file_name="Text_Network_Gephi_Results.gexf", mime="application/gexf+xml")
else:
    st.info("Preprocess your data to enable network analysis.")

# ----- Step 4: Topic Modeling -----
st.header("4. Topic Modeling")
if st.session_state.documents is not None:
    tfidf_params = {
        "ngram_range": (tfidf_ngram_lower, tfidf_ngram_upper),
        "max_features": tfidf_max_features,
        "min_df": tfidf_min_df,
        "max_df": tfidf_max_df
    }
    nmf_params = {
        "n_components": n_topics,
        "beta_loss": nmf_beta_loss,
        "solver": nmf_solver,
        "max_iter": nmf_max_iter,
        "alpha_H": nmf_alpha_H,
        "l1_ratio": nmf_l1_ratio
    }
    if st.button("Perform Topic Modeling"):
        (st.session_state.document_topic_df,
         st.session_state.word_topic_df,
         _, _) = perform_topic_modeling(st.session_state.documents,
                                        n_topics=n_topics,
                                        tfidf_params=tfidf_params,
                                        nmf_params=nmf_params)
        st.success("Topic modeling completed!")
        st.subheader("Document-Topic Assignments (Sample)")
        st.dataframe(st.session_state.document_topic_df.head())
        
        topic_excel = get_topic_modeling_excel(st.session_state.df,
                                               st.session_state.document_topic_df,
                                               st.session_state.word_topic_df)
        st.download_button("Download Topic Modeling Results (Excel)",
                           data=topic_excel,
                           file_name="NMF_Results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Preprocess your data to enable topic modeling.")

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
