import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import networkx as nx
import io
import plotly.graph_objects as go

# ------------------------------
# Helper Functions
# ------------------------------

def extract_hashtags(text):
    """Extracts hashtags from a text string using regex.
    
    Converts the input to a string to handle non-string values.
    """
    if pd.isnull(text):
        return []
    text = str(text)  # Convert non-string values to string
    return re.findall(r'#\w+', text)

def sanitize_graph(G):
    """Converts node and edge attributes to native Python types.
    
    In particular, converts any bytes values to strings and numpy scalars
    to Python scalars. This prevents errors during XML serialization.
    """
    # Sanitize node attributes
    for n, attr in G.nodes(data=True):
        for key, value in attr.items():
            if isinstance(value, bytes):
                G.nodes[n][key] = value.decode('utf-8')
            elif isinstance(value, (np.int64, np.float64)):
                G.nodes[n][key] = value.item()
    # Sanitize edge attributes
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            if isinstance(value, bytes):
                data[key] = value.decode('utf-8')
            elif isinstance(value, (np.int64, np.float64)):
                data[key] = value.item()

def add_footer():
    """
    Adds a footer with personal information and social links.
    """
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

# ------------------------------
# Main App
# ------------------------------

def main():
    st.title("Hashtag Co-occurrence Network & Topic Modeling Analysis")

    # Initialize a step counter in session_state.
    if "step" not in st.session_state:
        st.session_state.step = 1

    # ---------- Step 1: File Upload ----------
    if st.session_state.step == 1:
        st.header("Step 1: Upload File")
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.write("### Data Preview")
            st.write(df.head())
            if st.button("Next: Select Column"):
                st.session_state.step = 2

    # ---------- Step 2: Select Column ----------
    if st.session_state.step >= 2:
        st.header("Step 2: Select Text Column")
        df = st.session_state.df
        text_column = st.selectbox("Select the textual column to analyze for hashtags", df.columns)
        st.session_state.text_column = text_column
        if st.button("Next: Extract Hashtags"):
            st.session_state.step = 3

    # ---------- Step 3: Extract Hashtags ----------
    if st.session_state.step >= 3:
        st.header("Step 3: Extract Hashtags")
        df = st.session_state.df
        text_column = st.session_state.text_column
        df['hashtags'] = df[text_column].apply(extract_hashtags)
        st.write("### Data with Extracted Hashtags")
        st.write(df.head())
        # Create a string version of the hashtags for vectorization.
        df['hashtags_str'] = df['hashtags'].apply(lambda x: " ".join(x))
        st.session_state.df = df  # update dataframe in session_state
        if st.button("Next: Configure CountVectorizer"):
            st.session_state.step = 4

    # ---------- Step 4: CountVectorizer & Co-occurrence Matrix ----------
    if st.session_state.step >= 4:
        st.header("Step 4: Configure CountVectorizer & Compute Co-occurrence Matrix")
        st.sidebar.header("CountVectorizer Options")
        token_pattern = st.sidebar.text_input("Token Pattern", value=r'#\w+')
        min_df = st.sidebar.number_input("Min Document Frequency", min_value=1, value=1, step=1)
        max_df = st.sidebar.slider("Max Document Frequency (proportion)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        ngram_min = st.sidebar.number_input("N-gram Range (min)", min_value=1, value=1, step=1)
        ngram_max = st.sidebar.number_input("N-gram Range (max)", min_value=ngram_min, value=ngram_min, step=1)

        if st.button("Next: Compute Co-occurrence"):
            vectorizer = CountVectorizer(token_pattern=token_pattern,
                                         min_df=min_df,
                                         max_df=max_df,
                                         ngram_range=(ngram_min, ngram_max))
            X = vectorizer.fit_transform(st.session_state.df['hashtags_str'])
            st.session_state.vectorizer = vectorizer
            st.session_state.X = X
            hashtag_labels = vectorizer.get_feature_names_out()
            st.session_state.hashtag_labels = hashtag_labels
            # Compute co-occurrence matrix: X.T * X (diagonals are term frequencies)
            cooccurrence_matrix = (X.T * X).toarray()
            st.session_state.cooccurrence_matrix = cooccurrence_matrix
            cooccurrence_df = pd.DataFrame(cooccurrence_matrix, index=hashtag_labels, columns=hashtag_labels)
            st.write("### Hashtag Co-occurrence Matrix")
            st.write(cooccurrence_df)
            st.session_state.step = 5

    # ---------- Step 5: Build Network Graph ----------
    if st.session_state.step >= 5:
        st.header("Step 5: Build Network Graph")
        cooccurrence_matrix = st.session_state.cooccurrence_matrix
        hashtag_labels = st.session_state.hashtag_labels
        # Build graph from co-occurrence matrix (ignoring self-loops)
        G = nx.Graph()
        for i in range(len(hashtag_labels)):
            for j in range(i + 1, len(hashtag_labels)):
                weight = cooccurrence_matrix[i][j]
                if weight > 0:
                    G.add_edge(hashtag_labels[i], hashtag_labels[j], weight=weight)
        # Compute centrality measures
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        for node in G.nodes():
            G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
            G.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
            G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
        st.session_state.G = G
        if st.button("Next: Network Downloads & Frequency Chart"):
            st.session_state.step = 6

    # ---------- Step 6: Network Summary & Downloads ----------
    if st.session_state.step >= 6:
        st.header("Step 6: Network Summary & Download Options")
        G = st.session_state.G
        # Manually show graph info instead of using nx.info(G)
        st.write(f"**Number of nodes:** {G.number_of_nodes()}")
        st.write(f"**Number of edges:** {G.number_of_edges()}")
        # Sanitize graph attributes to avoid bytes issues
        sanitize_graph(G)
        # Prepare download for Gephi (GEXF format)
        gexf_buffer = io.StringIO()
        nx.write_gexf(G, gexf_buffer)
        gexf_data = gexf_buffer.getvalue().encode('utf-8')
        st.download_button(
            label="Download Gephi File (GEXF)",
            data=gexf_data,
            file_name="network.gexf",
            mime="text/xml"
        )
        # Prepare download for edge list CSV.
        edge_data = []
        for u, v, data in G.edges(data=True):
            edge_data.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1)
            })
        edge_df = pd.DataFrame(edge_data)
        csv_buffer = io.StringIO()
        edge_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')
        st.download_button(
            label="Download Edge List CSV",
            data=csv_data,
            file_name="edge_list.csv",
            mime="text/csv"
        )
        if st.button("Next: Frequency Bar Chart"):
            st.session_state.step = 7

    # ---------- Step 7: Frequency Bar Chart ----------
    if st.session_state.step >= 7:
        st.header("Step 7: Interactive Frequency Bar Chart")
        # Use session_state to track number of hashtags to display.
        if "num_hashtags" not in st.session_state:
            st.session_state.num_hashtags = 10
        col_freq = st.columns(2)
        if col_freq[0].button("+ Hashtags", key="freq_plus"):
            st.session_state.num_hashtags += 1
        if col_freq[1].button("- Hashtags", key="freq_minus"):
            if st.session_state.num_hashtags > 1:
                st.session_state.num_hashtags -= 1

        # Calculate term frequencies from the CountVectorizer output.
        X = st.session_state.X
        hashtag_labels = st.session_state.hashtag_labels
        term_frequencies = np.array(X.sum(axis=0)).flatten()
        freq_df = pd.DataFrame({
            "hashtag": hashtag_labels,
            "frequency": term_frequencies
        }).sort_values(by="frequency", ascending=False)
        top_freq_df = freq_df.head(st.session_state.num_hashtags)
        fig_freq = go.Figure(data=go.Bar(x=top_freq_df["hashtag"], y=top_freq_df["frequency"]))
        fig_freq.update_layout(title="Top Frequent Hashtags", xaxis_title="Hashtag", yaxis_title="Frequency")
        st.plotly_chart(fig_freq)
        if st.button("Next: Optional Topic Modeling"):
            st.session_state.step = 8

    # ---------- Step 8: Optional NNMF Topic Modeling ----------
    if st.session_state.step >= 8:
        st.header("Step 8: NNMF Topic Modeling (Optional)")
        perform_topic_modeling = st.sidebar.checkbox("Perform NNMF Topic Modeling", value=False)
        if perform_topic_modeling:
            n_topics = st.sidebar.number_input("Number of Topics", min_value=1, value=3, step=1)
            if "num_hashtags_topic" not in st.session_state:
                st.session_state.num_hashtags_topic = 10
            col_topic = st.columns(2)
            if col_topic[0].button("+ Hashtags per Topic", key="topic_plus"):
                st.session_state.num_hashtags_topic += 1
            if col_topic[1].button("- Hashtags per Topic", key="topic_minus"):
                if st.session_state.num_hashtags_topic > 1:
                    st.session_state.num_hashtags_topic -= 1

            # Run NNMF on the CountVectorizer matrix.
            X = st.session_state.X
            nmf_model = NMF(n_components=n_topics, init='nndsvd', random_state=42)
            W = nmf_model.fit_transform(X)
            H = nmf_model.components_
            feature_names = st.session_state.hashtag_labels

            st.write("#### NNMF Topic Modeling Results")
            for topic_idx, topic in enumerate(H):
                top_indices = topic.argsort()[::-1][:st.session_state.num_hashtags_topic]
                top_hashtags = [feature_names[i] for i in top_indices]
                top_weights = topic[top_indices]
                fig_topic = go.Figure(data=go.Bar(x=top_hashtags, y=top_weights))
                fig_topic.update_layout(
                    title=f"Topic {topic_idx+1}",
                    xaxis_title="Hashtag",
                    yaxis_title="Weight"
                )
                st.plotly_chart(fig_topic)
        else:
            st.write("NNMF Topic Modeling not selected.")

    # ---------- Footer ----------
    add_footer()

if __name__ == "__main__":
    main()
