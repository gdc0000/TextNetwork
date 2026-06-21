from io import BytesIO
from typing import Optional

import pandas as pd
import streamlit as st

SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}


def load_excel_file(file: BytesIO) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(file, engine="openpyxl")
        if df.empty:
            st.warning("The uploaded file contains no data.")
            return None
        return df
    except ValueError as e:
        st.error(f"Invalid file format: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None
