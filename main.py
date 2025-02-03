import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import networkx as nx
import io
import plotly.graph_objects as go

def extract_hashtags(text):
    """Extract hashtags from a text string using regex."""
    if pd.isnull(text):
        return []
    return re.findall(r'#\w+', text)

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

def main():
    st.title("Hashtag Co-occurrence Network & Topic Modeling Analysis")
    
    # 1. File Upload: Accept CSV or Excel files.
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        # Read file based on extension.
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())
        
        # 2. Ask the user which column contains text to analyze.
        text_column = st.selectbox("Select the textual column to analyze for hashtags", df.columns)
        
        # 3. Extract hashtags from the selected column and store them in a new column.
        df['hashtags'] = df[text_column].apply(extract_hashtags)
        st.write("### Data with Extracted Hashtags")
        st.write(df.head())
        
        # Create a string version of hashtags (space-separated).
        df['hashtags_str'] = df['hashtags'].apply(lambda x: " ".join(x))
        
        # -------------------------------
        # Customizable CountVectorizer options.
        # -------------------------------
        st.sidebar.header("CountVectorizer Options")
        token_pattern = st.sidebar.text_input("Token Pattern", value=r'#\w+')
        min_df = st.sidebar.number_input("Min Document Frequency", min_value=1, value=1, step=1)
        max_df = st.sidebar.slider("Max Document Frequency (proportion)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        ngram_min = st.sidebar.number_input("N-gram Range (min)", min_value=1, value=1, step=1)
        ngram_max = st.sidebar.number_input("N-gram Range (max)", min_value=ngram_min, value=ngram_min, step=1)
        
        # 4. Use CountVectorizer to compute the hashtag co-occurrence matrix.
        vectorizer = CountVectorizer(token_pattern=token_pattern, 
                                     min_df=min_df, 
                                     max_df=max_df, 
                                     ngram_range=(ngram_min, ngram_max))
        X = vectorizer.fit_transform(df['hashtags_str'])
        # Co-occurrence matrix computed as X.T * X (diagonals are term frequencies).
        cooccurrence_matrix = (X.T * X).toarray()
        hashtag_labels = vectorizer.get_feature_names_out()
        
        st.write("### Hashtag Co-occurrence Matrix")
        cooccurrence_df = pd.DataFrame(cooccurrence_matrix, index=hashtag_labels, columns=hashtag_labels)
        st.write(cooccurrence_df)
        
        # 5. Build a network graph from the co-occurrence matrix.
        G = nx.Graph()
        for i in range(len(hashtag_labels)):
            for j in range(i + 1, len(hashtag_labels)):
                weight = cooccurrence_matrix[i][j]
                if weight > 0:
                    G.add_edge(hashtag_labels[i], hashtag_labels[j], weight=weight)
        
        # 6. Compute centrality measures and assign them as node attributes.
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        for node in G.nodes():
            G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
            G.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
            G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
        
        st.write("### Network Graph Summary")
        st.write(nx.info(G))
        
        # 7. Download options for network data.
        # 7a. Download Gephi file (GEXF format).
        gexf_buffer = io.StringIO()
        nx.write_gexf(G, gexf_buffer)
        gexf_data = gexf_buffer.getvalue().encode('utf-8')
        st.download_button(
            label="Download Gephi File (GEXF)",
            data=gexf_data,
            file_name="network.gexf",
            mime="text/xml"
        )
        # 7b. Download edge list as CSV.
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
        
        # ----------------------------------------------------
        # 8. Plotly interactive bar visualization: Most Frequent Hashtags.
        # Use session_state to keep track of the number to show.
        if "num_hashtags" not in st.session_state:
            st.session_state.num_hashtags = 10
        
        st.write("### Most Frequent Hashtags")
        col_freq = st.columns(2)
        if col_freq[0].button("+ Hashtags", key="freq_plus"):
            st.session_state.num_hashtags += 1
        if col_freq[1].button("- Hashtags", key="freq_minus"):
            if st.session_state.num_hashtags > 1:
                st.session_state.num_hashtags -= 1
        
        # Calculate term frequencies from the CountVectorizer output.
        term_frequencies = np.array(X.sum(axis=0)).flatten()
        freq_df = pd.DataFrame({
            "hashtag": hashtag_labels,
            "frequency": term_frequencies
        }).sort_values(by="frequency", ascending=False)
        top_freq_df = freq_df.head(st.session_state.num_hashtags)
        
        fig_freq = go.Figure(data=go.Bar(x=top_freq_df["hashtag"], y=top_freq_df["frequency"]))
        fig_freq.update_layout(title="Top Frequent Hashtags", xaxis_title="Hashtag", yaxis_title="Frequency")
        st.plotly_chart(fig_freq)
        
        # ----------------------------------------------------
        # 9. NNMF Topic Modeling (optional).
        perform_topic_modeling = st.sidebar.checkbox("Perform NNMF Topic Modeling", value=False)
        if perform_topic_modeling:
            st.write("### NNMF Topic Modeling")
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
            nmf_model = NMF(n_components=n_topics, init='nndsvd', random_state=42)
            W = nmf_model.fit_transform(X)
            H = nmf_model.components_
            feature_names = hashtag_labels
            
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
                
    # Add footer at the end of the app.
    add_footer()

if __name__ == "__main__":
    main()
