from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".sav"}


def load_file(file: BytesIO) -> Optional[pd.DataFrame]:
    name = getattr(file, "name", None)
    ext = Path(name).suffix.lower() if name else None

    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(file, engine="openpyxl")
        elif ext == ".csv":
            df = pd.read_csv(file)
        elif ext == ".sav":
            df = pd.read_spss(file)
        else:
            st.error(f"Unsupported file format: {ext}")
            return None

        if df.empty:
            st.warning("The uploaded file contains no data.")
            return None
        return df
    except ValueError as e:
        st.error(f"Invalid file format: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
