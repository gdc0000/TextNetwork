import networkx as nx
import pandas as pd

from src.network.analysis import compute_graph_centralities
from src.network.builder import build_text_network
from src.network.export import get_edge_list_csv, get_gephi_gexf


class TestBuildTextNetwork:
    def test_basic_network(self) -> None:
        docs = pd.Series(["#hello #world", "#hello #python", "#world #python"])
        G = build_text_network(docs)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_empty_documents(self) -> None:
        docs = pd.Series([], dtype=str)
        G = build_text_network(docs)
        assert G.number_of_nodes() == 0

    def test_cv_params(self) -> None:
        docs = pd.Series(["#hello #world", "#hello #python"])
        G = build_text_network(docs, cv_params={"min_df": 1})
        assert G.number_of_nodes() >= 2


class TestComputeGraphCentralities:
    def test_centralities_added(self) -> None:
        G: nx.Graph = nx.Graph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        compute_graph_centralities(G)
        for node in G.nodes():
            for attr in ("degree", "eigenvector", "closeness", "betweenness"):
                assert attr in G.nodes[node]

    def test_isolated_node(self) -> None:
        G: nx.Graph = nx.Graph()
        G.add_node("alone")
        compute_graph_centralities(G)
        assert G.nodes["alone"]["degree"] == 0


class TestExport:
    def test_edge_list_csv_excludes_self_loops(self) -> None:
        G: nx.Graph = nx.Graph()
        G.add_edge("a", "b")
        G.add_edge("a", "a")
        csv_bytes = get_edge_list_csv(G)
        assert b"Source,Target" in csv_bytes
        assert csv_bytes.count(b"\n") == 2

    def test_gephi_gexf_valid_xml(self) -> None:
        G: nx.Graph = nx.Graph()
        G.add_edge("x", "y")
        gexf = get_gephi_gexf(G)
        assert gexf.startswith(b"<?xml")
