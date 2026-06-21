from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_TFIDF_PARAMS: Dict[str, Any] = {
    "ngram_range": (1, 3),
    "max_features": 1000,
    "min_df": 5,
    "max_df": 0.9,
}

DEFAULT_NMF_PARAMS: Dict[str, Any] = {
    "beta_loss": "kullback-leibler",
    "solver": "mu",
    "max_iter": 1000,
    "alpha_H": 0.1,
    "l1_ratio": 0.5,
}


def perform_topic_modeling(
    documents: pd.Series,
    n_topics: int = 10,
    tfidf_params: Optional[Dict[str, Any]] = None,
    nmf_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    if tfidf_params is None:
        tfidf_params = {**DEFAULT_TFIDF_PARAMS}
    if nmf_params is None:
        nmf_params = {**DEFAULT_NMF_PARAMS, "n_components": n_topics}

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
