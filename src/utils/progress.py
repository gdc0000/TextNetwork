import time

import streamlit as st

STEPS = 100


def simulate_progress(step_description: str, duration: float = 1.0) -> None:
    st.info(step_description)
    progress_bar = st.progress(0)
    for i in range(STEPS):
        time.sleep(duration / STEPS)
        progress_bar.progress(i + 1)
