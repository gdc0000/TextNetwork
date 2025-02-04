# Text Network and Topic Modeling Analysis App

This repository contains a Streamlit application that performs advanced text analysis on data contained in an Excel file. The app is designed for social scientists and data analysts who want to explore text through network analysis and topic modeling. It offers extensive configuration options, detailed scientific explanations, and progress bars for each processing step.

## Features

- **Data Upload:**  
  Upload an Excel file containing your text data and preview its contents.

- **Preprocessing (Hashtag Extraction):**  
  Extracts hashtags from a specified text column to focus on the most meaningful keywords.  
  _Scientific Explanation:_ Hashtags often capture key topics and sentiments, allowing the analysis to reduce noise and emphasize relevant features.

- **Text Network Analysis:**  
  Constructs a co-occurrence network of terms using customizable CountVectorizer settings. Computes centrality measures (degree, eigenvector, closeness, and betweenness) to identify the most influential terms.  
  _Scientific Explanation:_ The network is built by converting text to a document-term matrix and then creating a co-occurrence (adjacency) matrix. Centrality measures help in understanding the role and influence of terms within the network.

- **Topic Modeling:**  
  Performs topic modeling using TF-IDF vectorization and Non-negative Matrix Factorization (NMF). Users can adjust parameters such as n-gram range, max features, and regularization settings.  
  _Scientific Explanation:_ TF-IDF weighs words by their importance relative to the corpus, while NMF decomposes the TF-IDF matrix into latent topics, revealing the underlying thematic structure of the text data.

- **Custom Configuration:**  
  The sidebar provides extensive options for configuring preprocessing, network analysis, and topic modeling parameters. Detailed scientific explanations accompany each setting to help users understand their impact.

- **Progress Bars:**  
  Visual feedback (progress bars) is provided during each processing step to enhance user experience.

- **Downloadable Outputs:**  
  Download results as:
  - **CSV:** Edge list for network analysis.
  - **GEXF:** Network file for visualization in Gephi.
  - **Excel:** Detailed topic modeling results.

- **Footer:**  
  Displays personal details and social links for **Gabriele Di Cicco, PhD in Social Psychology**.

## Installation

Ensure you have Python 3.7 or higher installed. It is recommended to use a virtual environment.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/<your-username>/<your-repository>.git
   cd <your-repository>
   ```

2. **Install the Dependencies:**

   You can install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes:
   ```
   streamlit
   pandas
   openpyxl
   xlsxwriter
   scikit-learn
   networkx
   ```

## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

### Step-by-Step Workflow

1. **Data Upload:**  
   - Use the sidebar to upload an Excel file containing your text data.
   - Specify the name of the text column that will be used for analysis.
   - The main window will display a preview of the uploaded data.

2. **Preprocessing (Hashtag Extraction):**  
   - Click the **"Preprocess Data"** button.
   - A progress bar will indicate the status of hashtag extraction.
   - The app extracts hashtags from the specified column, and a sample of the processed text is shown.

3. **Text Network Analysis:**  
   - Configure CountVectorizer parameters (min_df, max_df, n-gram range) in the sidebar.
   - Click **"Build Text Network"** to convert the processed text into a co-occurrence network.
   - A progress bar will simulate the processing steps.
   - The network summary (number of nodes and edges) is displayed, and download buttons are provided for the edge list CSV and Gephi GEXF file.

4. **Topic Modeling:**  
   - Adjust TF-IDF and NMF parameters via the sidebar.
   - Click **"Perform Topic Modeling"** to extract latent topics.
   - A progress bar will indicate the progress.
   - The app displays a sample of document-topic assignments and offers a downloadable Excel file with detailed results.


## Customization

The app is designed to be modular and easily customizable:
- **CountVectorizer Settings:** Adjust `min_df`, `max_df`, and `ngram_range` to control which terms are included in the network.
- **TF-IDF and NMF Parameters:** Fine-tune the topic modeling process by changing parameters like the n-gram range, maximum features, and regularization settings.
- **Scientific Explanations:** Each configurable option is accompanied by a scientific rationale to help users understand its impact on the analysis.

## Contributing

Contributions are welcome! If you have suggestions for improvements, encounter issues, or would like to extend the functionality, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

**Gabriele Di Cicco, PhD in Social Psychology**  
- [GitHub](https://github.com/gdc0000)  
- [ORCID](https://orcid.org/0000-0002-1439-5790)  
- [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
