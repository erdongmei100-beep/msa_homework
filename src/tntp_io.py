"""数据读取模块：加载 Sioux Falls 网络和 OD 需求。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_sioux_falls_network(net_path: str):
    """加载 Sioux Falls 网络 TNTP 文件。

    参数
    ----
    net_path: str
        网络文件路径（SiouxFalls_net.tntp）。

    返回
    ----
    tuple
        (tail, head, capacity, free_flow_time, alpha, beta, n_nodes)
    """
    df = pd.read_csv(
        net_path,
        sep="\t",
        lineterminator="\n",
        comment="~",
        skiprows=8,
        usecols=np.arange(1, 11),
        engine="python",
    )
    df.columns = [
        "init_node",
        "term_node",
        "capacity",
        "length",
        "free_flow_time",
        "B",
        "power",
        "speed_limit",
        "toll",
        "link_type",
    ]

    tail = df["init_node"].to_numpy(dtype=int) - 1
    head = df["term_node"].to_numpy(dtype=int) - 1

    capacity = df["capacity"].to_numpy(float)
    free_flow_time = df["free_flow_time"].to_numpy(float)
    alpha = df["B"].to_numpy(float)
    beta = df["power"].to_numpy(float)

    n_nodes = int(max(tail.max(), head.max()) + 1)

    return tail, head, capacity, free_flow_time, alpha, beta, n_nodes


def load_sioux_falls_od(od_csv_path: str):
    """加载 Sioux Falls OD CSV，返回稠密需求矩阵。"""
    df = pd.read_csv(od_csv_path)
    origins = np.sort(df["O"].unique())
    dests = np.sort(df["D"].unique())

    origins_0 = origins - 1
    dests_0 = dests - 1

    demand_matrix = np.zeros((len(origins), len(dests)), dtype=float)
    for _, row in df.iterrows():
        o = int(row["O"])
        d = int(row["D"])
        ton = float(row["Ton"])
        oi = np.where(origins == o)[0][0]
        di = np.where(dests == d)[0][0]
        demand_matrix[oi, di] = ton

    return origins_0, dests_0, demand_matrix


__all__ = ["load_sioux_falls_network", "load_sioux_falls_od"]
