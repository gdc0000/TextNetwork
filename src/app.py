"""
Streamlit App for Text Network and Topic Modeling Analysis.

Run with:
    streamlit run src/app.py
"""

import streamlit as st

from src.data.loader import load_file
from src.data.preprocessing import preprocess_documents
from src.network.analysis import compute_graph_centralities
from src.network.builder import build_text_network
from src.network.export import get_edge_list_csv, get_gephi_gexf
from src.topics.export import get_topic_modeling_excel
from src.topics.modeling import perform_topic_modeling
from src.utils.progress import simulate_progress


def main() -> None:
    # Session state
    for key in ("df", "documents", "G", "document_topic_df", "word_topic_df"):
        if key not in st.session_state:
            st.session_state[key] = None

    # Sidebar
    st.sidebar.header("Configuration Panel")
    st.sidebar.markdown(
        "Upload a data file (Excel, CSV, SPSS), specify the text column, and configure "
        "network / topic modeling parameters below."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload a data file", type=["xlsx", "xls", "csv", "sav"]
    )
    text_column = st.sidebar.text_input("Text column name", value="text")

    st.sidebar.header("Text Network Analysis Options")
    cv_min_df = st.sidebar.number_input("min_df", value=1, step=1)
    cv_max_df = st.sidebar.number_input("max_df", value=1.0, step=0.1)
    cv_ngram_lower = st.sidebar.number_input("ngram_range lower", value=1, step=1)
    cv_ngram_upper = st.sidebar.number_input("ngram_range upper", value=1, step=1)

    st.sidebar.header("Topic Modeling Options")
    tfidf_ngram_lower = st.sidebar.number_input(
        "TFIDF ngram_range lower", value=1, step=1
    )
    tfidf_ngram_upper = st.sidebar.number_input(
        "TFIDF ngram_range upper", value=3, step=1
    )
    tfidf_max_features = st.sidebar.number_input("max_features", value=1000, step=100)
    tfidf_min_df = st.sidebar.number_input("min_df", value=5, step=1)
    tfidf_max_df = st.sidebar.number_input("max_df", value=0.9, step=0.1)

    n_topics = st.sidebar.number_input("Number of Topics", value=10, step=1)
    nmf_beta_loss = st.sidebar.selectbox(
        "beta_loss", options=["kullback-leibler", "frobenius"], index=0
    )
    nmf_solver = st.sidebar.selectbox("solver", options=["mu", "cd"], index=0)
    nmf_max_iter = st.sidebar.number_input("max_iter", value=1000, step=100)
    nmf_alpha_H = st.sidebar.number_input("alpha_H", value=0.1, step=0.1, format="%.2f")
    nmf_l1_ratio = st.sidebar.number_input(
        "l1_ratio", value=0.5, step=0.1, format="%.2f"
    )

    # Main layout
    st.title("Hashtag Network and Topic Modeling Analysis")
    st.markdown(
        "1. **Preprocessing** — extracts hashtags from your text.\n"
        "2. **Text Network Analysis** — builds a co-occurrence network and computes "
        "centralities.\n"
        "3. **Topic Modeling** — applies TF-IDF and NMF to uncover latent topics."
    )

    # Step 1: Upload
    st.header("1. Data Upload")
    if uploaded_file is not None:
        st.session_state.df = load_file(uploaded_file)
        if st.session_state.df is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head())
    else:
        st.info("Please upload a data file from the sidebar.")

    # Step 2: Preprocessing
    st.header("2. Preprocessing (Hashtag Extraction)")
    if st.session_state.df is not None:
        if st.button("Preprocess Data"):
            simulate_progress(
                "Preprocessing data: Extracting hashtags from the text...", duration=1.0
            )
            st.session_state.documents = preprocess_documents(
                st.session_state.df, text_column
            )
            if st.session_state.documents is not None:
                st.success("Preprocessing completed!")
                st.subheader("Sample Processed Text")
                st.write(st.session_state.documents.head())
    else:
        st.info("Upload data to enable preprocessing.")

    # Step 3: Network Analysis
    st.header("3. Text Network Analysis")
    if st.session_state.documents is not None:
        cv_params = {
            "min_df": cv_min_df,
            "max_df": cv_max_df,
            "ngram_range": (cv_ngram_lower, cv_ngram_upper),
        }
        if st.button("Build Text Network"):
            simulate_progress(
                "Building text network: Converting text to a network based on "
                "co-occurrence...",
                duration=1.5,
            )
            st.session_state.G = build_text_network(
                st.session_state.documents, cv_params=cv_params
            )
            compute_graph_centralities(st.session_state.G)
            st.success("Text network built and centralities computed!")
            st.write(f"**Number of nodes:** {st.session_state.G.number_of_nodes()}")
            st.write(f"**Number of edges:** {st.session_state.G.number_of_edges()}")

            edge_csv = get_edge_list_csv(st.session_state.G)
            gephi_gexf = get_gephi_gexf(st.session_state.G)
            st.download_button(
                "Download Edge List CSV",
                data=edge_csv,
                file_name="Text_Network_Edge_List.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download Gephi GEXF",
                data=gephi_gexf,
                file_name="Text_Network_Gephi_Results.gexf",
                mime="application/gexf+xml",
            )
    else:
        st.info("Preprocess your data to enable network analysis.")

    # Step 4: Topic Modeling
    st.header("4. Topic Modeling")
    if st.session_state.documents is not None:
        tfidf_params = {
            "ngram_range": (tfidf_ngram_lower, tfidf_ngram_upper),
            "max_features": tfidf_max_features,
            "min_df": tfidf_min_df,
            "max_df": tfidf_max_df,
        }
        nmf_params = {
            "n_components": n_topics,
            "beta_loss": nmf_beta_loss,
            "solver": nmf_solver,
            "max_iter": nmf_max_iter,
            "alpha_H": nmf_alpha_H,
            "l1_ratio": nmf_l1_ratio,
        }
        if st.button("Perform Topic Modeling"):
            simulate_progress(
                "Performing topic modeling: Extracting latent topics from the text...",
                duration=2.0,
            )
            (
                st.session_state.document_topic_df,
                st.session_state.word_topic_df,
                _,
                _,
            ) = perform_topic_modeling(
                st.session_state.documents,
                n_topics=n_topics,
                tfidf_params=tfidf_params,
                nmf_params=nmf_params,
            )
            st.success("Topic modeling completed!")
            st.subheader("Document-Topic Assignments (Sample)")
            st.dataframe(st.session_state.document_topic_df.head())

            topic_excel = get_topic_modeling_excel(
                st.session_state.df,
                st.session_state.document_topic_df,
                st.session_state.word_topic_df,
            )
            st.download_button(
                "Download Topic Modeling Results (Excel)",
                data=topic_excel,
                file_name="NMF_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("Preprocess your data to enable topic modeling.")


