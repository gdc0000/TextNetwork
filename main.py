import streamlit as st
import pandas as pd
import networkx as nx
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
import io

st.title("Text Network Analysis and Topic Modeling")
st.markdown(
    """
    This app lets you upload an Excel or CSV file containing text data.
    It extracts hashtags from the specified text column, builds a text network,
    computes centrality measures, and performs topic modeling via NMF.
    
    You can then download:
    - An **Edge List CSV** file,
    - A **Gephi GEXF** file, and
    - An **Excel** file with topic modeling results.
    """
)

# File uploader for Excel or CSV
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])
# User input for which column contains the text to analyze
text_column = st.text_input("Enter the column name containing text", value="text")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")

    # Sidebar parameters for TF-IDF and NMF
    st.sidebar.header("Tfidf Vectorizer Parameters")
    ngram_min = st.sidebar.number_input("Min ngram", min_value=1, value=1)
    ngram_max = st.sidebar.number_input("Max ngram", min_value=1, value=3)
    max_features = st.sidebar.number_input("Max features", min_value=100, value=1000)
    min_df = st.sidebar.number_input("Min document frequency", min_value=1, value=5)
    max_df = st.sidebar.slider("Max document frequency ratio", 0.0, 1.0, 0.9)

    st.sidebar.header("NMF Parameters")
    n_topics = st.sidebar.number_input("Number of topics", min_value=2, value=10)
    max_iter = st.sidebar.number_input("Max iterations", min_value=100, value=1000)
    alpha_H = st.sidebar.number_input("Alpha_H", min_value=0.0, value=0.1, format="%.2f")
    l1_ratio = st.sidebar.number_input("L1 ratio", min_value=0.0, max_value=1.0, value=0.5, format="%.2f")

    # --- Text Preprocessing & Hashtag Extraction ---
    def extract_hashtags(text):
        """Extract hashtags from text and remove the '#' symbol."""
        if pd.isnull(text):
            return ""
        text = str(text)
        hashtags = re.findall(r'#\w+', text)
        return " ".join(tag.lstrip("#") for tag in hashtags)

    # Apply the hashtag extraction function
    documents = df[text_column].apply(extract_hashtags)
    st.subheader("Extracted Hashtags")
    st.write(documents.head())

    # --- Text Network Analysis ---
    # Create a document-term matrix based on hashtags
    cv = CountVectorizer()
    X = cv.fit_transform(documents)
    words = cv.get_feature_names_out()

    # Build the adjacency (co-occurrence) matrix and graph
    Adj = pd.DataFrame((X.T * X).toarray(), columns=words, index=words)
    G = nx.from_pandas_adjacency(Adj)

    # Calculate centrality measures
    degree_dict = dict(G.degree())
    try:
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception as e:
        st.warning("Eigenvector centrality did not converge. Try adjusting parameters.")
        eigenvector_dict = {node: 0 for node in G.nodes()}
    closeness_dict = nx.closeness_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G)

    # Set the calculated measures as node attributes
    nx.set_node_attributes(G, degree_dict, "degree")
    nx.set_node_attributes(G, eigenvector_dict, "eigenvector")
    nx.set_node_attributes(G, closeness_dict, "closeness")
    nx.set_node_attributes(G, betweenness_dict, "betweenness")

    # Create an edge list (excluding self-loops)
    edge_list = [(u, v) for u, v in G.edges() if u != v]
    edge_df = pd.DataFrame(edge_list, columns=["Source", "Target"])
    st.subheader("Edge List Preview")
    st.dataframe(edge_df.head())

    # --- Prepare Downloadable Outputs ---
    # Output A: Edge List CSV
    csv_buffer = io.StringIO()
    edge_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Output B: Gephi GEXF file
    gexf_buffer = io.StringIO()
    nx.write_gexf(G, gexf_buffer)
    gexf_data = gexf_buffer.getvalue().encode("utf-8")  # convert to bytes for download

    # --- Topic Modeling ---
    st.subheader("Topic Modeling with NMF")
    tfidf = TfidfVectorizer(
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    V = tfidf.fit_transform(documents)
    words_tfidf = tfidf.get_feature_names_out()
    st.write("TF-IDF matrix shape:", V.shape)

    nmf_model = NMF(
        n_components=n_topics,
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=max_iter,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        random_state=42,
    )
    W = nmf_model.fit_transform(V)
    H = nmf_model.components_

    # Create dataframes for topic modeling results
    word_topic_df = pd.DataFrame(H.T, index=words_tfidf)
    document_topic_df = pd.DataFrame(W)
    document_topic_df["Topic"] = document_topic_df.idxmax(axis=1)
    sheet_1 = pd.concat([df, document_topic_df], axis=1)
    sheet_2 = word_topic_df

    # Output C: Excel file with two sheets (documents and words)
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        sheet_1.to_excel(writer, sheet_name="documents", index=False)
        sheet_2.to_excel(writer, sheet_name="words")
    excel_data = excel_buffer.getvalue()

    # --- Download Buttons ---
    st.download_button(
        label="Download Edge List CSV",
        data=csv_data,
        file_name="Text_Network_Edge_List.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Gephi GEXF File",
        data=gexf_data,
        file_name="Text_Network_Gephi_Results.gexf",
        mime="text/xml",
    )

    st.download_button(
        label="Download Topic Modeling Excel File",
        data=excel_data,
        file_name="NMF_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
