from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AnalysisConfig:
    # CountVectorizer defaults
    cv_min_df: int = 1
    cv_max_df: float = 1.0
    cv_ngram_range: Tuple[int, int] = (1, 1)

    # TF-IDF defaults
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    tfidf_max_features: int = 1000
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.9

    # NMF defaults
    n_topics: int = 10
    nmf_beta_loss: str = "kullback-leibler"
    nmf_solver: str = "mu"
    nmf_max_iter: int = 1000
    nmf_alpha_H: float = 0.1
    nmf_l1_ratio: float = 0.5

    # UI
    app_title: str = "Hashtag Network and Topic Modeling Analysis"
    sidebar_header: str = "Configuration Panel"
    text_column_default: str = "text"
