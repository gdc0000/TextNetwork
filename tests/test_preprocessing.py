import pandas as pd

from src.data.preprocessing import extract_hashtags, preprocess_documents


class TestExtractHashtags:
    def test_extracts_hashtags(self) -> None:
        result = extract_hashtags("Hello #world and #python")
        assert result == "world python"

    def test_no_hashtags(self) -> None:
        result = extract_hashtags("Hello world")
        assert result == ""

    def test_nan_input(self) -> None:
        result = extract_hashtags(float("nan"))
        assert result == ""

    def test_none_input(self) -> None:
        result = extract_hashtags(None)
        assert result == ""

    def test_non_string_input(self) -> None:
        result = extract_hashtags(42)
        assert result == ""

    def test_duplicate_hashtags(self) -> None:
        result = extract_hashtags("#test #test")
        assert result == "test test"


class TestPreprocessDocuments:
    def test_valid_column(self) -> None:
        df = pd.DataFrame({"text": ["#hello #world", "#python"]})
        result = preprocess_documents(df, "text")
        assert result is not None
        assert list(result) == ["hello world", "python"]

    def test_missing_column(self) -> None:
        df = pd.DataFrame({"other": ["data"]})
        result = preprocess_documents(df, "text")
        assert result is None

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"text": []})
        result = preprocess_documents(df, "text")
        assert result is not None
        assert len(result) == 0
