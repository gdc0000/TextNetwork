import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from stop_words import get_stop_words
import tempfile
import os
import numpy as np

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

def main():
    # **1. Set Page Configuration FIRST**
    st.set_page_config(page_title="📚 Educational Text Network Analysis App", layout="wide")

    # **2. Title and Overview**
    st.title("📚 Educational Text Network Analysis App")

    st.markdown("""
    ### **Overview**
    This application analyzes the co-occurrence of words in your text data and provides various centrality metrics to help identify the most important or influential words based on different criteria. Customize the analysis by adjusting parameters and excluding stop words in multiple languages.
    """)

    # **3. Sidebar: Data Upload**
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

    # **4. Sidebar: Configuration Settings**
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

    # **5. Sidebar: Stop Words Exclusion**
    st.sidebar.header("🛑 Stop Words Exclusion")
    languages = st.sidebar.multiselect(
        "Select Languages for Stop Words Exclusion",
        options=["english", "french", "spanish", "italian"],
        default=["english"],
        help="Select the languages for which you want to exclude stop words."
    )

    if not languages:
        st.sidebar.warning("⚠️ At least one language should be selected for stop words exclusion.")

    # **6. Run Analysis Button**
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

            # Generate Edgelist
            edgelist = pd.DataFrame(
                [(u, v, d['weight']) for u, v, d in G.edges(data=True)],
                columns=['Source', 'Target', 'Weight']
            )

            # Generate Adjacency Matrix
            adjacency_matrix = Adj.copy()

            # Generate Gephi file (GEXF)
            gexf_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.gexf')
            nx.write_gexf(G, gexf_temp.name)
            gexf_temp.close()

        st.success("✅ **Analysis Complete!**")

        # **7. Display Centrality Metrics**
        st.header("🔍 Centrality Metrics")
        st.write("""
        The table below displays various centrality measures for each word in the network. Centrality metrics help identify the most important or influential words based on different criteria.
        """)
        st.dataframe(centrality_df)

        # **8. Display Word Frequency**
        st.header("📈 Word Frequency")
        word_freq = pd.DataFrame({
            'Word': list(word_counts.head(top_n).index),
            'Count': list(word_counts.head(top_n).values)
        })
        st.bar_chart(word_freq.set_index('Word')['Count'])

        # **9. Downloadable Files**
        st.header("📂 Downloadable Files")
        st.write("""
        You can download the following files for further analysis:
        - **Adjacency Matrix**: Represents the co-occurrence of words.
        - **Edgelist**: Represents the connections between words.
        - **Gephi File (GEXF)**: For advanced network analysis in Gephi.
        """)

        # **Adjacency Matrix Download**
        st.subheader("Adjacency Matrix")
        csv_adj = adjacency_matrix.to_csv(index=True)
        st.download_button(
            label="⬇️ Download Adjacency Matrix (CSV)",
            data=csv_adj,
            file_name="Adjacency_Matrix.csv",
            mime="text/csv"
        )

        # **Edgelist Download - CSV**
        st.subheader("Edgelist (CSV)")
        csv_edgelist = edgelist.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Edgelist (CSV)",
            data=csv_edgelist,
            file_name="Edgelist.csv",
            mime="text/csv"
        )

        # **Edgelist Download - Excel**
        excel_edgelist = edgelist.to_excel(index=False, engine='openpyxl')
        st.download_button(
            label="⬇️ Download Edgelist (Excel)",
            data=excel_edgelist,
            file_name="Edgelist.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # **Gephi File Download**
        st.subheader("Gephi File (GEXF)")
        with open(gexf_temp.name, 'rb') as f:
            gephi_data = f.read()
        st.download_button(
            label="⬇️ Download Gephi File (GEXF)",
            data=gephi_data,
            file_name="Text_Network_Gephi_Results.gexf",
            mime="application/gexf+xml"
        )
        # Clean up temporary Gephi file
        os.remove(gexf_temp.name)

    add_footer()

if __name__ == "__main__":
    main()
