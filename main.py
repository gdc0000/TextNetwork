import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def add_footer():
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

def main():
    st.set_page_config(page_title="Text Network Analysis", layout="wide")
    st.title("📚 Educational Text Network Analysis App")

    st.sidebar.header("Configuration Settings")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file containing a 'text' column", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if 'text' not in data.columns:
                st.error("The uploaded CSV must contain a 'text' column.")
                return
            documents = data['text'].astype(str).tolist()
            st.success("File successfully uploaded and loaded.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
    else:
        st.info("Awaiting CSV file upload.")
        st.stop()

    st.markdown("""
    ### **How It Works**
    This application analyzes the co-occurrence of words in your text data and visualizes the relationships as a network graph. You can customize various parameters to explore different aspects of your data.
    """)

    # Configuration parameters
    min_df = st.sidebar.slider(
        "Minimum Document Frequency (min_df)",
        min_value=1,
        max_value=100,
        value=22,
        step=1,
        help="Ignore terms that appear in fewer than the specified number of documents."
    )

    max_df = st.sidebar.slider(
        "Maximum Document Frequency (max_df)",
        min_value=0.5,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Ignore terms that appear in more than the specified proportion of documents."
    )

    top_n = st.sidebar.number_input(
        "Top N Words to Display",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of top words to include based on frequency."
    )

    # Button to trigger analysis
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Processing..."):
            # 1. Create Bag-of-Words Representation
            cv = CountVectorizer(min_df=min_df, max_df=max_df)
            X = cv.fit_transform(documents)
            words = cv.get_feature_names_out()

            # Optionally limit to top N words
            word_counts = X.sum(axis=0).A1
            word_freq = pd.DataFrame({'word': words, 'count': word_counts})
            word_freq = word_freq.sort_values(by='count', ascending=False).head(top_n)
            selected_words = word_freq['word'].tolist()

            cv = CountVectorizer(vocabulary=selected_words, min_df=min_df, max_df=max_df)
            X = cv.fit_transform(documents)
            words = cv.get_feature_names_out()

            # 2. Construct Adjacency Matrix
            Adj = pd.DataFrame((X.T * X).toarray(), columns=words, index=words)
            Adj.values[[np.diag_indices_from(Adj)]] = 0  # Remove self-loops

            # 3. Build Graph
            G = nx.from_pandas_adjacency(Adj)

            # 4. Compute Centrality Metrics
            degree_dict = dict(G.degree(G.nodes()))
            eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000)
            closeness_dict = nx.closeness_centrality(G)
            betweenness_dict = nx.betweenness_centrality(G)

            # 5. Assign Attributes to Nodes
            nx.set_node_attributes(G, degree_dict, 'degree')
            nx.set_node_attributes(G, betweenness_dict, 'betweenness')
            nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
            nx.set_node_attributes(G, closeness_dict, 'closeness')

            # 6. Prepare Data for Display
            centrality_df = pd.DataFrame({
                'Word': list(G.nodes()),
                'Degree': list(degree_dict.values()),
                'Betweenness': list(betweenness_dict.values()),
                'Eigenvector': list(eigenvector_dict.values()),
                'Closeness': list(closeness_dict.values())
            }).sort_values(by='Degree', ascending=False)

        st.success("Analysis Complete!")

        st.header("🔍 Centrality Metrics")
        st.write("""
        The table below displays various centrality measures for each word in the network. Centrality metrics help identify the most important or influential words based on different criteria.
        """)
        st.dataframe(centrality_df)

        st.header("📈 Word Frequency")
        st.bar_chart(word_freq.set_index('word')['count'])

        st.header("🕸️ Word Co-occurrence Network")

        # Visualize using PyVis
        net = Network(height='600px', width='100%', notebook=False, directed=False)

        # Add nodes with centrality attributes
        for node in G.nodes(data=True):
            word = node[0]
            size = node[1]['degree'] * 2  # Adjust size based on degree
            title = (f"Word: {word}<br>"
                     f"Degree: {node[1]['degree']}<br>"
                     f"Betweenness: {node[1]['betweenness']:.4f}<br>"
                     f"Eigenvector: {node[1]['eigenvector']:.4f}<br>"
                     f"Closeness: {node[1]['closeness']:.4f}")
            net.add_node(word, label=word, size=size, title=title)

        # Add edges
        for edge in G.edges(data=True):
            word1, word2, data = edge
            weight = data.get('weight', 1)
            net.add_edge(word1, word2, value=weight)

        # Generate and display the network
        net.force_atlas_2based()
        net.show("network.html")
        st.components.v1.html(open("network.html", 'r', encoding='utf-8').read(), height=600, scrolling=True)

        st.header("📂 Export Graph")
        st.write("You can export the network graph in GEXF format for further analysis in tools like Gephi.")
        if st.button("Download GEXF"):
            gexf_data = nx.generate_gexf(G)
            gexf_str = ''.join(gexf_data)
            st.download_button(
                label="Download GEXF",
                data=gexf_str,
                file_name="Text_Network_Gephi_Results.gexf",
                mime="application/gexf+xml"
            )

    add_footer()

if __name__ == "__main__":
    import numpy as np
    main()
