from io import BytesIO

import pandas as pd

from src.topics.export import get_topic_modeling_excel
from src.topics.modeling import perform_topic_modeling


class TestPerformTopicModeling:
    def test_basic_topics(self) -> None:
        docs = pd.Series(
            [
                "#machine #learning is great",
                "#deep #learning is powerful",
                "#natural #language processing",
                "#computer #vision is cool",
                "#reinforcement #learning rocks",
            ]
        )
        tfidf_params = {
            "ngram_range": (1, 1),
            "max_features": 100,
            "min_df": 1,
            "max_df": 1.0,
        }
        nmf_params = {
            "n_components": 2,
            "beta_loss": "frobenius",
            "solver": "cd",
            "max_iter": 100,
        }
        doc_topic, word_topic, W, H = perform_topic_modeling(
            docs, n_topics=2, tfidf_params=tfidf_params, nmf_params=nmf_params
        )
        assert "Topic" in doc_topic.columns
        assert doc_topic.shape[0] == 5
        assert word_topic.shape[0] > 0

    def test_custom_params(self) -> None:
        docs = pd.Series([
            "#data #science", "#machine #learning",
            "#deep #nlp", "#text #mining", "#ai #ml",
        ])
        tfidf_params = {
            "ngram_range": (1, 1),
            "max_features": 10,
            "min_df": 1,
            "max_df": 1.0,
        }
        nmf_params = {
            "n_components": 2,
            "beta_loss": "frobenius",
            "solver": "cd",
            "max_iter": 100,
        }
        doc_topic, word_topic, W, H = perform_topic_modeling(
            docs, n_topics=2, tfidf_params=tfidf_params, nmf_params=nmf_params
        )
        assert doc_topic.shape[0] == 5


class TestGetTopicModelingExcel:
    def test_excel_has_two_sheets(self) -> None:
        original = pd.DataFrame({"text": ["#hello world"]})
        doc_topic = pd.DataFrame({"Topic": [0]})
        word_topic = pd.DataFrame({0: [0.5]}, index=["hello"])
        excel_bytes = get_topic_modeling_excel(original, doc_topic, word_topic)
        sheets = pd.ExcelFile(BytesIO(excel_bytes)).sheet_names
        assert "documents" in sheets
        assert "words" in sheets
