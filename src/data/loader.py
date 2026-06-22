from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional

import pandas as pd
import streamlit as st

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".sav"}


def _read_excel(file: BytesIO) -> pd.DataFrame:
    return pd.read_excel(file, engine="openpyxl")


def _read_sav(file: BytesIO) -> pd.DataFrame:
    tmp_path: Optional[str] = None
    try:
        file.seek(0)
        with NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        return pd.read_spss(tmp_path)
    except ImportError:
        st.error(
            "Reading SPSS (.sav) files requires the 'pyreadstat' package. "
            "Install it with: pip install pyreadstat"
        )
        raise
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


_READERS: dict[str, Callable[[BytesIO], pd.DataFrame]] = {
    ".xlsx": _read_excel,
    ".xls": _read_excel,
    ".csv": pd.read_csv,
    ".sav": _read_sav,
}


def load_file(file: BytesIO) -> Optional[pd.DataFrame]:
    name = getattr(file, "name", None)
    ext = Path(name).suffix.lower() if name else None

    reader = _READERS.get(ext) if ext else None
    if reader is None:
        st.error(f"Unsupported file format: {ext}")
        return None

    try:
        df = reader(file)
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
