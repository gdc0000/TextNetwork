#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit App for Text Network and Topic Modeling Analysis with Detailed Explanations

This application allows you to:
  1. Upload an Excel file containing text data.
  2. Preprocess the text by extracting hashtags.
  3. Build a text network with customizable CountVectorizer parameters and compute centrality measures.
  4. Perform topic modeling using customizable TF-IDF and NMF parameters.

Each section includes a progress bar and a detailed explanation of the scientific rationale behind the method and parameters.

Downloadable outputs include:
  - Edge list CSV (for network analysis).
  - Gephi GEXF file (for network visualization in Gephi).
  - Excel file containing topic modeling results.

Run the app with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
import re
import time
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
    
    Scientific Explanation:
    Hashtags often capture key topics and sentiments. By extracting hashtags, we focus on the most
    meaningful words in social media or text data, reducing noise and emphasizing relevant features.
    """
    if pd.isnull(text):
        return ""
    hashtags = re.findall(r'#(\w+)', str(text))
    return " ".join(hashtags)

def preprocess_documents(df: pd.DataFrame, text_column: str) -> pd.Series:
    """
    Applies hashtag extraction to the specified text column.
    
    Scientific Explanation:
    Preprocessing reduces complexity by isolating important keywords. Extracting hashtags
    helps in filtering out less relevant words and prepares the data for further network and topic analyses.
    """
    if text_column not in df.columns:
        st.error(f"Column '{text_column}' not found in the DataFrame.")
        return None
    documents = df[text_column].apply(extract_hashtags)
    return documents

def build_text_network(documents: pd.Series, cv_params: dict = None) -> nx.Graph:
    """
    Builds a text network using CountVectorizer with given parameters, computes the co-occurrence matrix,
    and returns a networkx Graph.
    
    Scientific Explanation:
    CountVectorizer converts text into a document-term matrix. Multiplying the matrix by its transpose
    yields a co-occurrence matrix that indicates how frequently terms appear together. This matrix is
    interpreted as an adjacency matrix, where nodes are terms and edges represent co-occurrence.
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
    Computes and assigns centrality measures (degree, eigenvector, closeness, betweenness) to graph nodes.
    
    Scientific Explanation:
    Centrality measures are critical in network analysis:
    - **Degree Centrality:** Counts direct connections (importance by frequency).
    - **Eigenvector Centrality:** Considers the influence of neighbors.
    - **Closeness Centrality:** Measures the average distance to all other nodes.
    - **Betweenness Centrality:** Reflects the extent to which a node lies on paths between others.
    These metrics help identify the most influential terms in the network.
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
    
    Scientific Explanation:
    - **TF-IDF Vectorization:** Weighs words by their frequency in a document relative to the corpus,
      emphasizing terms that are significant to individual documents.
    - **Non-negative Matrix Factorization (NMF):** Decomposes the TF-IDF matrix into topics (latent features),
      allowing us to interpret clusters of words as coherent topics.
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

def simulate_progress(step_description: str, duration: float = 1.0):
    """
    Simulates a progress bar for a given step.
    
    Args:
        step_description (str): Description to display above the progress bar.
        duration (float): Total duration in seconds for the progress simulation.
    """
    st.info(step_description)
    progress_bar = st.progress(0)
    steps = 100
    for i in range(steps):
        time.sleep(duration / steps)
        progress_bar.progress(i + 1)

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
# Sidebar Configuration with Scientific Explanations
# ---------------------------
st.sidebar.header("Configuration Panel & Scientific Explanations")
st.sidebar.markdown("""
This panel allows you to configure the parameters used in the analysis:

**Data Upload:**
- **Upload Excel File:** Provide your file containing text data.

**Text Column:**
- **Text Column Name:** Specify the column name in your file that contains text. This column is where hashtags will be extracted.

**Text Network Analysis Options (CountVectorizer):**
- **min_df:** Minimum document frequency for a term to be included.
- **max_df:** Maximum document frequency for a term to be included.
- **ngram_range:** Determines the range of n-grams (e.g., single words, bi-grams) extracted. This affects the granularity of the network.

**Topic Modeling Options (TF-IDF & NMF):**
- **TF-IDF Parameters:** Control how the importance of terms is measured relative to the entire corpus.
- **NMF Parameters:** Define how topics are extracted from the TF-IDF matrix. The number of topics, loss function, and regularization settings impact the interpretability of the topics.
""")

uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])
text_column = st.sidebar.text_input("Text column name", value="text")

st.sidebar.header("Text Network Analysis Options")
st.sidebar.markdown("""
**CountVectorizer Parameters:**
- **min_df:** Minimum number of documents a term must appear in.
- **max_df:** Maximum proportion of documents a term can appear in.
- **ngram_range:** The lower and upper boundary for the range of n-values for n-grams to be extracted.
""")
cv_min_df = st.sidebar.number_input("min_df", value=1, step=1)
cv_max_df = st.sidebar.number_input("max_df", value=1.0, step=0.1)
cv_ngram_lower = st.sidebar.number_input("ngram_range lower", value=1, step=1)
cv_ngram_upper = st.sidebar.number_input("ngram_range upper", value=1, step=1)

st.sidebar.header("Topic Modeling Options")
st.sidebar.markdown("""
**TF-IDF Parameters:**
- **ngram_range:** The range of n-grams to consider (e.g., 1-gram to 3-gram).
- **max_features:** The maximum number of terms to consider.
- **min_df & max_df:** Thresholds for term frequency to filter out rare or too-common words.

**NMF Parameters:**
- **Number of Topics:** The number of latent topics to extract.
- **beta_loss:** The loss function to be used.
- **solver:** The algorithm used for optimization.
- **max_iter:** Maximum iterations for convergence.
- **alpha_H & l1_ratio:** Regularization parameters that control the sparsity of the model.
""")
tfidf_ngram_lower = st.sidebar.number_input("TFIDF ngram_range lower", value=1, step=1)
tfidf_ngram_upper = st.sidebar.number_input("TFIDF ngram_range upper", value=3, step=1)
tfidf_max_features = st.sidebar.number_input("max_features", value=1000, step=100)
tfidf_min_df = st.sidebar.number_input("min_df", value=5, step=1)
tfidf_max_df = st.sidebar.number_input("max_df", value=0.9, step=0.1)

n_topics = st.sidebar.number_input("Number of Topics", value=10, step=1)
nmf_beta_loss = st.sidebar.selectbox("beta_loss", options=['kullback-leibler', 'frobenius'], index=0)
nmf_solver = st.sidebar.selectbox("solver", options=['mu', 'cd'], index=0)
nmf_max_iter = st.sidebar.number_input("max_iter", value=1000, step=100)
nmf_alpha_H = st.sidebar.number_input("alpha_H", value=0.1, step=0.1, format="%.2f")
nmf_l1_ratio = st.sidebar.number_input("l1_ratio", value=0.5, step=0.1, format="%.2f")

# ---------------------------
# Main App Layout with Explanations and Progress Bars
# ---------------------------
st.title("Text Network and Topic Modeling Analysis")
st.markdown("""
This application performs several analyses on your text data:

1. **Preprocessing:** Extracts hashtags from the specified text column.
2. **Text Network Analysis:** Builds a network from co-occurrence of terms and computes centrality measures to identify important terms.
3. **Topic Modeling:** Applies TF-IDF and NMF to uncover latent topics within your text.

Each processing step is accompanied by a progress bar and scientific explanation to help you understand the methods used.
""")

# ----- Step 1: Data Upload -----
st.header("1. Data Upload")
if uploaded_file is not None:
    st.session_state.df = load_excel_file(uploaded_file)
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
else:
    st.info("Please upload an Excel file from the sidebar.")

# ----- Step 2: Preprocessing (Hashtag Extraction) -----
st.header("2. Preprocessing (Hashtag Extraction)")
if st.session_state.df is not None:
    if st.button("Preprocess Data"):
        simulate_progress("Preprocessing data: Extracting hashtags from the text...", duration=1.0)
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
        simulate_progress("Building text network: Converting text to a network based on co-occurrence...", duration=1.5)
        st.session_state.G = build_text_network(st.session_state.documents, cv_params=cv_params)
        compute_graph_centralities(st.session_state.G)
        st.success("Text network built and centralities computed!")
        st.write(f"**Number of nodes:** {st.session_state.G.number_of_nodes()}")
        st.write(f"**Number of edges:** {st.session_state.G.number_of_edges()}")
        
        # Downloadable network outputs
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
        simulate_progress("Performing topic modeling: Extracting latent topics from the text...", duration=2.0)
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
# Footer with Personal Information
# ---------------------------
st.markdown("---")
st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
st.markdown("""
[GitHub](https://github.com/gdc0000) | 
[ORCID](https://orcid.org/0000-0002-1439-5790) | 
[LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
""")
