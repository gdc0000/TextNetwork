import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from stop_words import get_stop_words
import tempfile
import os
import numpy as np
import sys

def add_footer():
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

def load_data(uploaded_file, file_type, header):
    try:
        if file_type == "CSV":
            df = pd.read_csv(uploaded_file, header=0 if header else None, sep=None, engine='python')
        elif file_type == "TSV":
            df = pd.read_csv(uploaded_file, header=0 if header else None, sep='\t')
        elif file_type == "XLSX":
            df = pd.read_excel(uploaded_file, header=0 if header else None)
        elif file_type == "TXT":
            # Assuming one document per line
            content = uploaded_file.read().decode('utf-8')
            if header:
                # If headers are present, read as DataFrame
                df = pd.DataFrame([line.split('\t') for line in content.splitlines()])
                df.columns = df.iloc[0]  # Set first row as header
                df = df[1:]
            else:
                # No headers, assume single column named 'text'
                df = pd.DataFrame({'text': content.splitlines()})
        else:
            st.error("Unsupported file type.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_stop_words_list(selected_languages):
    stop_words = set()
    for lang in selected_languages:
        try:
            stop_words.update(get_stop_words(lang))
        except Exception as e:
            st.warning(f"Could not load stop words for language '{lang}': {e}")
    return list(stop_words)

def create_cooccurrence_matrix(documents, min_df, max_df, stop_words):
    cv = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words)
    X = cv.fit_transform(documents)
    words = cv.get_feature_names_out()
    Adj = pd.DataFrame((X.T * X).toarray(), columns=words, index=words)
    np.fill_diagonal(Adj.values, 0)  # Remove self-loops
    return Adj, words

def build_graph(Adj):
    G = nx.from_pandas_adjacency(Adj)
    return G

def compute_centrality_measures(G):
    degree_dict = dict(G.degree(G.nodes()))
    try:
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        st.warning("Eigenvector centrality did not converge. Setting all eigenvector centrality values to 0.")
        eigenvector_dict = {node: 0 for node in G.nodes()}
    closeness_dict = nx.closeness_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G)
    return degree_dict, eigenvector_dict, closeness_dict, betweenness_dict

def assign_attributes(G, degree, betweenness, eigenvector, closeness):
    nx.set_node_attributes(G, degree, 'degree')
    nx.set_node_attributes(G, betweenness, 'betweenness')
    nx.set_node_attributes(G, eigenvector, 'eigenvector')
    nx.set_node_attributes(G, closeness, 'closeness')

def generate_network_html(G):
    net = Network(height='600px', width='100%', notebook=False, directed=False)
    # Add nodes with centrality attributes
    for node, data in G.nodes(data=True):
        size = data.get('degree', 1) * 2  # Adjust size based on degree
        title = (f"Word: {node}<br>"
                 f"Degree: {data.get('degree', 0)}<br>"
                 f"Betweenness: {data.get('betweenness', 0):.4f}<br>"
                 f"Eigenvector: {data.get('eigenvector', 0):.4f}<br>"
                 f"Closeness: {data.get('closeness', 0):.4f}")
        net.add_node(node, label=node, size=size, title=title)
    # Add edges
    for edge in G.edges(data=True):
        word1, word2, data = edge
        weight = data.get('weight', 1)
        net.add_edge(word1, word2, value=weight)
    net.force_atlas_2based()
    try:
        # Attempt to use to_html()
        html_content = net.to_html(full_html=False)
        return html_content
    except AttributeError:
        # Fallback method using generate_html (if available)
        st.error("PyVis 'to_html' method is not available. Please ensure you have pyvis version 0.6.1 or higher.")
        return ""

def main():
    # **1. Set Page Configuration FIRST**
    st.set_page_config(page_title="📚 Educational Text Network Analysis App", layout="wide")

    # **2. Optional: Display PyVis version for debugging**
    try:
        import pyvis
        st.sidebar.markdown(f"**PyVis Version:** {pyvis.__version__}")
    except ImportError:
        st.sidebar.markdown("**PyVis Version:** Not installed")

    # **3. Title and Overview**
    st.title("📚 Educational Text Network Analysis App")

    st.markdown("""
    ### **Overview**
    This application analyzes the co-occurrence of words in your text data and visualizes the relationships as a network graph. It provides various centrality metrics to help identify the most important or influential words based on different criteria. Customize the analysis by adjusting parameters and excluding stop words in multiple languages.
    """)

    # **4. Sidebar: Data Upload**
    st.sidebar.header("📁 Upload Data")
    file_type = st.sidebar.selectbox("Select File Type", ["CSV", "TSV", "XLSX", "TXT"])
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv", "tsv", "xlsx", "txt"])
    header = st.sidebar.checkbox("Label in the first row?", value=True, help="Check if your file contains headers in the first row.")

    if uploaded_file is not None:
        df = load_data(uploaded_file, file_type, header)
        if df is not None:
            if 'text' not in df.columns:
                if header:
                    # If headers are present but 'text' column is missing, prompt user
                    st.error("The uploaded file does not contain a 'text' column.")
                    st.stop()
                else:
                    # If no headers, assume the first column is text
                    if file_type != "TXT":
                        df.columns = ['text']
            documents = df['text'].astype(str).tolist()
            st.success("✅ File successfully uploaded and loaded.")
    else:
        st.info("📄 Awaiting file upload.")
        st.stop()

    # **5. Sidebar: Configuration Settings**
    st.sidebar.header("🔧 Configuration Settings")

    min_df = st.sidebar.slider(
        "Minimum Document Frequency (min_df)",
        min_value=1,
        max_value=100,
        value=22,
        step=1,
        help="Ignore terms that appear in fewer than the specified number of documents."
    )

    max_df = st.sidebar.slider(
        "Maximum Document Frequency (max_df) (%)",
        min_value=50,
        max_value=100,
        value=100,
        step=5,
        help="Ignore terms that appear in more than the specified percentage of documents."
    )
    max_df_fraction = max_df / 100  # Convert to fraction

    top_n = st.sidebar.number_input(
        "Top N Words to Display",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of top words to include based on frequency."
    )

    # **6. Sidebar: Stop Words Exclusion**
    st.sidebar.header("🛑 Stop Words Exclusion")
    languages = st.sidebar.multiselect(
        "Select Languages for Stop Words Exclusion",
        options=["english", "french", "spanish", "italian"],
        default=["english"],
        help="Select the languages for which you want to exclude stop words."
    )

    if not languages:
        st.sidebar.warning("⚠️ At least one language should be selected for stop words exclusion.")

    # **7. Run Analysis Button**
    run_analysis = st.sidebar.button("🔄 Run Analysis")

    if run_analysis:
        if not languages:
            st.error("❌ Please select at least one language for stop words exclusion.")
            st.stop()
        with st.spinner("🔍 Processing..."):
            stop_words = get_stop_words_list(languages)
            Adj, words = create_cooccurrence_matrix(documents, min_df, max_df_fraction, stop_words)

            # Optionally limit to top N words
            word_counts = Adj.sum(axis=1).sort_values(ascending=False)
            selected_words = word_counts.head(top_n).index.tolist()
            Adj = Adj.loc[selected_words, selected_words]
            words = selected_words

            G = build_graph(Adj)
            degree_dict, eigenvector_dict, closeness_dict, betweenness_dict = compute_centrality_measures(G)
            assign_attributes(G, degree_dict, betweenness_dict, eigenvector_dict, closeness_dict)

            centrality_df = pd.DataFrame({
                'Word': list(G.nodes()),
                'Degree': list(degree_dict.values()),
                'Betweenness': list(betweenness_dict.values()),
                'Eigenvector': list(eigenvector_dict.values()),
                'Closeness': list(closeness_dict.values())
            }).sort_values(by='Degree', ascending=False)

            # Generate network HTML
            network_html = generate_network_html(G)

        st.success("✅ **Analysis Complete!**")

        # **8. Display Centrality Metrics**
        st.header("🔍 Centrality Metrics")
        st.write("""
        The table below displays various centrality measures for each word in the network. Centrality metrics help identify the most important or influential words based on different criteria.
        """)
        st.dataframe(centrality_df)

        # **9. Display Word Frequency**
        st.header("📈 Word Frequency")
        word_freq = pd.DataFrame({
            'word': list(word_counts.head(top_n).index),
            'count': list(word_counts.head(top_n).values)
        })
        st.bar_chart(word_freq.set_index('word')['count'])

        # **10. Display Word Co-occurrence Network**
        st.header("🕸️ Word Co-occurrence Network")
        st.write("""
        The interactive network graph below visualizes the relationships between words based on their co-occurrence in the documents. Nodes represent words, and edges represent co-occurrences. Hover over a node to see its centrality metrics.
        """)
        try:
            if network_html:
                st.components.v1.html(network_html, height=600, scrolling=True)
            else:
                st.error("⚠️ Failed to generate network visualization.")
        except Exception as e:
            st.error(f"⚠️ Error displaying network graph: {e}")

        # **11. Export Graph**
        st.header("📂 Export Graph")
        st.write("You can export the network graph in GEXF format for further analysis in tools like Gephi.")
        if st.button("⬇️ Download GEXF"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.gexf') as tmp_gexf:
                    nx.write_gexf(G, tmp_gexf.name)
                    tmp_gexf.close()
                    with open(tmp_gexf.name, 'rb') as f:
                        gexf_data = f.read()
                    st.download_button(
                        label="Download GEXF",
                        data=gexf_data,
                        file_name="Text_Network_Gephi_Results.gexf",
                        mime="application/gexf+xml"
                    )
                os.remove(tmp_gexf.name)
            except Exception as e:
                st.error(f"⚠️ Error exporting GEXF: {e}")

    add_footer()

if __name__ == "__main__":
    main()
