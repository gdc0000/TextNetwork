from io import BytesIO

import networkx as nx
import pandas as pd


def get_edge_list_csv(G: nx.Graph) -> bytes:
    edge_list = [(u, v) for u, v in G.edges() if u != v]
    edge_df = pd.DataFrame(edge_list, columns=["Source", "Target"])
    return edge_df.to_csv(index=False).encode("utf-8")


def get_gephi_gexf(G: nx.Graph) -> bytes:
    buffer = BytesIO()
    nx.write_gexf(G, buffer)
    return buffer.getvalue()
