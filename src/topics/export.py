from io import BytesIO

import pandas as pd


def get_topic_modeling_excel(
    original_df: pd.DataFrame,
    document_topic_df: pd.DataFrame,
    word_topic_df: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        combined = pd.concat([original_df, document_topic_df], axis=1)
        combined.to_excel(writer, sheet_name="documents", index=False)
        word_topic_df.to_excel(writer, sheet_name="words")
    return output.getvalue()
