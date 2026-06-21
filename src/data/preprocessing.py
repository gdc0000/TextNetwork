import math
import re
from typing import Any, Optional

import pandas as pd
import streamlit as st

HASHTAG_PATTERN = re.compile(r"#(\w+)")


def extract_hashtags(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    hashtags = HASHTAG_PATTERN.findall(str(text))
    return " ".join(hashtags)


def preprocess_documents(df: pd.DataFrame, text_column: str) -> Optional[pd.Series]:
    if text_column not in df.columns:
        st.error(f"Column '{text_column}' not found in the DataFrame.")
        return None
    documents = df[text_column].apply(extract_hashtags)
    return documents
