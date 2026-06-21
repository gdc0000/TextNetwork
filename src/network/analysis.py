import logging
from typing import Dict

import networkx as nx

logger = logging.getLogger(__name__)


def compute_graph_centralities(G: nx.Graph) -> None:
    degree_dict: Dict[str, int] = dict(G.degree())
    try:
        eigenvector_dict: Dict[str, float] = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality did not converge; using zeros.")
        eigenvector_dict = {node: 0.0 for node in G.nodes()}
    closeness_dict: Dict[str, float] = nx.closeness_centrality(G)
    betweenness_dict: Dict[str, float] = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, degree_dict, "degree")
    nx.set_node_attributes(G, eigenvector_dict, "eigenvector")
    nx.set_node_attributes(G, closeness_dict, "closeness")
    nx.set_node_attributes(G, betweenness_dict, "betweenness")
