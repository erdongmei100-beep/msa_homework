"""Sioux Falls TAP 求解示例。"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from tntp_io import load_sioux_falls_network, load_sioux_falls_od
from msa_core import solve_ue_msa


def build_graph(tail: np.ndarray, head: np.ndarray, weights: np.ndarray, n_nodes: int) -> csr_matrix:
    """根据链路列表构建有向稀疏图。"""
    return csr_matrix((weights, (tail, head)), shape=(n_nodes, n_nodes))


def main():
    project_root = Path(__file__).resolve().parent.parent
    net_path = project_root / "data" / "SiouxFalls_net.tntp"
    od_path = project_root / "data" / "SiouxFalls_od.csv"

    print("Loading Sioux Falls network and OD data ...")
    tail, head, capacity, free_flow_time, alpha, beta, n_nodes = load_sioux_falls_network(str(net_path))
    origins, dests, demand_matrix = load_sioux_falls_od(str(od_path))

    graph = build_graph(tail, head, free_flow_time.copy(), n_nodes)

    print("Start solving with MSWA (d=2.0)...")
    start = time.time()
    flows, gap, iters = solve_ue_msa(
        graph=graph,
        tail=tail,
        head=head,
        free_flow_time=free_flow_time,
        capacity=capacity,
        alpha=alpha,
        beta=beta,
        origins=origins,
        dest_indices=dests,
        demand_matrix=demand_matrix,
        method="mswa",
        d=2.0,
        max_iter=200,
    )
    elapsed = time.time() - start

    print("Finished.")
    print(f"Iterations: {iters}")
    print(f"Relative gap: {gap:.6e}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Total flow (sum): {flows.sum():.2f}")


if __name__ == "__main__":
    main()
