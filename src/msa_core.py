"""MSA/MSWA 求解 Sioux Falls 用户均衡问题的核心逻辑。"""
from __future__ import annotations

import time

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


@njit
def load_aon_numba(predecessors, origins, dest_indices, demand_matrix, edge_index, num_links):
    """Numba 加速的全有全无分配。

    通过最短路前驱矩阵对每个 OD 对的需求沿最短路径分配到链路。
    """
    link_flows = np.zeros(num_links)

    for i in range(len(origins)):
        origin_node = origins[i]
        preds = predecessors[i]
        for j in range(len(dest_indices)):
            demand = demand_matrix[i, j]
            if demand <= 0.0:
                continue
            dest_node = dest_indices[j]
            curr = dest_node
            while curr != origin_node:
                pred_node = preds[curr]
                if pred_node < 0:
                    break
                eid = edge_index[pred_node, curr]
                if eid >= 0:
                    link_flows[eid] += demand
                curr = pred_node

    return link_flows


def solve_ue_msa(
    graph: csr_matrix,
    tail: np.ndarray,
    head: np.ndarray,
    free_flow_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    origins: np.ndarray,
    dest_indices: np.ndarray,
    demand_matrix: np.ndarray,
    max_iter: int = 500,
    method: str = "msa",
    d: float = 2.0,
    tol: float = 1e-4,
):
    """使用 MSA/MSWA 求解交通分配的用户均衡。

    参数
    ----
    graph : csr_matrix
        稀疏图的邻接矩阵，data 存储当前链路阻抗。
    tail, head : np.ndarray
        链路起讫节点索引（0-based）。
    free_flow_time, capacity, alpha, beta : np.ndarray
        BPR 阻抗函数参数。
    origins : np.ndarray
        作为多源最短路起点的节点索引。
    dest_indices : np.ndarray
        OD 需求的终点节点索引。
    demand_matrix : np.ndarray
        需求矩阵，行与 ``origins`` 对齐，列与 ``dest_indices`` 对齐。
    max_iter : int
        最大迭代次数。
    method : str
        "msa" 使用标准 1/k 步长，"mswa" 使用加权步长。
    d : float
        MSWA 幂次，步长权重为 k**d。
    tol : float
        相对间隙收敛阈值。

    返回
    ----
    tuple
        (x, rel_gap, iter_count)
    """
    n_nodes = int(max(tail.max(), head.max()) + 1)
    edge_index = -np.ones((n_nodes, n_nodes), dtype=np.int64)
    for eid in range(len(tail)):
        edge_index[tail[eid], head[eid]] = eid

    num_links = len(tail)
    x = np.zeros(num_links, dtype=float)
    s_k = 0.0
    start_time = time.time()

    for k in range(1, max_iter + 1):
        tt = free_flow_time * (1.0 + alpha * (x / capacity) ** beta)
        graph.data = tt

        dist, preds = shortest_path(
            graph,
            indices=origins,
            return_predecessors=True,
            directed=True,
        )

        y = load_aon_numba(
            preds,
            origins.astype(np.int64),
            dest_indices.astype(np.int64),
            demand_matrix,
            edge_index,
            num_links,
        )

        if method == "msa":
            lam = 1.0 / k
        elif method == "mswa":
            w_k = k ** d
            s_k += w_k
            lam = w_k / s_k
        else:
            raise ValueError("method must be 'msa' or 'mswa'")

        x = x + lam * (y - x)

        total_tt = float((tt * x).sum())
        gap_num = 0.0
        for i in range(len(origins)):
            for j in range(len(dest_indices)):
                q = demand_matrix[i, j]
                if q <= 0.0:
                    continue
                dj = dest_indices[j]
                c_min = dist[i, dj]
                gap_num += float(q * c_min)

        rel_gap = (total_tt - gap_num) / max(total_tt, 1e-9)
        print(f"Iter {k:4d}: lambda={lam:.4f}, gap={rel_gap:.6e}")

        if rel_gap < tol:
            elapsed = time.time() - start_time
            print(f"Converged at iter {k}, gap={rel_gap:.3e}, time={elapsed:.2f}s")
            break

    else:
        k = max_iter

    return x, rel_gap, k


__all__ = ["solve_ue_msa", "load_aon_numba"]
