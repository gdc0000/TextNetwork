from typing import Any, Dict, Optional

import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def build_text_network(
    documents: pd.Series,
    cv_params: Optional[Dict[str, Any]] = None,
) -> nx.Graph:
    if cv_params is None:
        cv_params = {}
    if documents.empty:
        return nx.Graph()
    cv = CountVectorizer(**cv_params)
    try:
        X = cv.fit_transform(documents)
    except ValueError:
        return nx.Graph()
    words = cv.get_feature_names_out()
    adj_matrix = (X.T * X).toarray()
    adj_df = pd.DataFrame(adj_matrix, index=words, columns=words)
    G = nx.from_pandas_adjacency(adj_df)
    return G
