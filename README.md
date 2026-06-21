# Text Network and Topic Modeling Analysis

[![CI](https://github.com/gdc0000/TextNetwork/actions/workflows/ci.yml/badge.svg)](https://github.com/gdc0000/TextNetwork/actions/workflows/ci.yml)

A Streamlit application for text network analysis and topic modeling from Excel data.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Project Structure

```
TextNetwork/
├── src/
│   ├── app.py              # Streamlit UI (entry point)
│   ├── data/
│   │   ├── loader.py       # Excel file loading
│   │   └── preprocessing.py # Hashtag extraction
│   ├── network/
│   │   ├── builder.py      # Co-occurrence network construction
│   │   ├── analysis.py     # Centrality computation
│   │   └── export.py       # CSV / GEXF export
│   ├── topics/
│   │   ├── modeling.py     # TF-IDF + NMF topic modeling
│   │   └── export.py       # Excel topic results export
│   └── utils/
│       ├── constants.py    # Default configuration
│       └── progress.py     # Progress bar helper
├── tests/                  # Pytest suite
├── main.py                 # Backward-compatible shim
├── Dockerfile              # Container image
└── .github/workflows/      # CI pipeline
```

## Features

- **Hashtag extraction** — isolates meaningful keywords from text
- **Co-occurrence network** — builds a term graph with centrality measures (degree, eigenvector, closeness, betweenness)
- **Topic modeling** — TF-IDF + NMF with configurable parameters
- **Downloads** — edge list CSV, Gephi GEXF, topic modeling Excel
- **Docker** — ready-to-deploy container image on `ghcr.io`

## Development

```bash
pip install -r requirements-dev.txt
pytest --cov=src/
ruff check src/ tests/
mypy src/ tests/
```

## License

MIT
