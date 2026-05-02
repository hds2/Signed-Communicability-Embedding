# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import expm, eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS


def get_adj_matrix(obj):
    if isinstance(obj, nx.Graph):
        return nx.to_numpy_array(obj)
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        raise ValueError("输入必须是 NetworkX 图或 NumPy 数组")


def calc_comm_metrics(obj, output="all"):

    A = get_adj_matrix(obj)
    N = len(A)

    S = expm(A)  
    K = np.diag(S)

    if output in ["all", "comm_coordinates"]:
        Lamb, U = eigh(A)
        X = np.diag(np.exp(Lamb / 2)) @ U.T

    if output in ["all", "distance"]:
        xi = np.sqrt(np.outer(K, np.ones(N)) + np.outer(np.ones(N), K) - 2 * S)

    if output in ["all", "angle"]:
        cos_theta = S / np.sqrt(np.outer(K, np.ones(N)) * np.outer(np.ones(N), K))
        cos_theta = np.clip(cos_theta, -1, 1) 
        theta = np.arccos(cos_theta)

    if output == "all":
        return S, xi, theta, X
    elif output == "comm":
        return S
    elif output == "distance":
        return xi
    elif output == "angle":
        return theta
    elif output == "comm_coordinates":
        return X
    else:
        raise ValueError("无效的输出类型")
    

def calculate_factions(G):

    N = G.order()
    distance = calc_comm_metrics(G, output="angle")
    distance = (distance + distance.T) / 2  
    MDS_dimension_v = [2, 3, 5, max(int(N / 10), 7)]
    n_factions_v = range(2, min(10, N))

    best_labels = None
    max_score = -np.inf

    for MDS_dimension in MDS_dimension_v:
        low_dim_embedding = MDS(n_components=MDS_dimension, dissimilarity="precomputed").fit_transform(distance)
        for n_factions in n_factions_v:
            labels = KMeans(n_clusters=n_factions).fit(low_dim_embedding).labels_

            if len(set(labels)) > 1 and len(set(labels)) < len(labels):  
                score = silhouette_score(distance, labels, metric="precomputed")
            else:
                score = 0  

            if score > max_score + 1e-4:  
                max_score = score
                best_labels = labels

    return best_labels


def embed_mds_coords(G, embedding_dimension=2):
    distance = calc_comm_metrics(G, output = 'angle')
    assert np.allclose(distance, distance.T, atol=1e-3), "距离矩阵不对称"
    distance = (distance + distance.T) / 2
    embedding = MDS(n_components=embedding_dimension, dissimilarity="precomputed")
    coords = embedding.fit_transform(distance)
    pos = {}
    for k, node in enumerate(G.nodes()):
        pos[node] = tuple(coords[k, :])

    return pos


def extract_base_layout(G):
    G_abs = nx.Graph()
    for u, v, w in G.edges(data="weight"):
        G_abs.add_edge(u, v)
    pos = nx.kamada_kawai_layout(G_abs)
    return pos


def visualize_signed_network(
    obj,
    ax=None,
    pos=None,
    labels=None,
    node_size=500,
    node_color="white",
    nodeedge_color="k",
    cmap_nodes=None,
    label_fontsize=12,
    pos_edge_width=2,
    neg_edge_width=2,
    pos_ls="-",
    neg_ls="--",
    with_labels=False,
    spines=False,
    differenciate_groups=False,
    group_assignments=None,
    group_markers=None,
    group_colors=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if isinstance(obj, nx.Graph):
        G = obj
    elif isinstance(obj, np.ndarray):
        G = nx.from_numpy_array(obj)
    else:
        raise ValueError("输入必须是 NetworkX 图或 NumPy 数组")

    if pos is None:
        pos = extract_base_layout(G)

    for u, v, w in G.edges(data="weight"):
        if w is None:
            G[u][v]["weight"] = 1

    edge_color = []
    ls = []
    width = []
    for u, v, w in G.edges(data=True):
        if w["weight"] > 0:
            edge_color.append("darkgreen")
            ls.append(pos_ls)
            width.append(pos_edge_width)
        else:
            edge_color.append("red")
            ls.append(neg_ls)
            width.append(neg_edge_width)

    if differenciate_groups:
        if group_assignments is None:
            raise ValueError(
                "当 differenciate_groups 为 True 时，必须提供 group_assignments"
            )
        groups = {}
        for node, group in group_assignments.items():
            groups.setdefault(group, []).append(node)
        if group_colors is None:
            default_colors = plt.cm.tab10.colors
            group_colors = {
                grp: default_colors[i % len(default_colors)]
                for i, grp in enumerate(sorted(groups.keys()))
            }
        if group_markers is None:
            default_markers = ["o", "s", "^", "D", "v", "p", "*", "X", "8"]
            group_markers = {
                grp: default_markers[i % len(default_markers)]
                for i, grp in enumerate(sorted(groups.keys()))
            }
        for grp, nodes in groups.items():
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                ax=ax,
                nodelist=nodes,
                node_size=node_size,
                node_color=[group_colors[grp]],
                node_shape=group_markers[grp],
                edgecolors=nodeedge_color,
            )
    else:
        if cmap_nodes is not None:
            nodes = nx.draw_networkx_nodes(
                G,
                pos=pos,
                ax=ax,
                node_size=node_size,
                node_color=node_color,
                cmap=cmap_nodes,
            )
        else:
            nodes = nx.draw_networkx_nodes(
                G, pos=pos, ax=ax, node_size=node_size, node_color=node_color
            )
        nodes.set_edgecolor(nodeedge_color)

    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color=edge_color, style=ls, width=width
    )
    if with_labels:
        nx.draw_networkx_labels(
            G, pos=pos, ax=ax, labels=labels, font_size=label_fontsize
        )
    if not spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    return ax

def compute_sce_representation(A, c=0.5, epsilon=1e-9):
    evals, evecs = eigh(A)
    rho_m = np.max(np.abs(evals))
    beta = c / (rho_m + epsilon)
    exp_beta_evals = np.exp(beta * evals)
    W_beta = evecs @ np.diag(exp_beta_evals) @ evecs.T
    W_diag = np.diag(W_beta)
    M_beta = W_diag[:, None] + W_diag[None, :] - 2 * W_beta
    return W_beta, M_beta


def calculate_sce_dissimilarity(W1, M1, W2, M2, epsilon=1e-9):
    N = W1.shape[0]

    iu = np.triu_indices(N, k=1)
    

    W1_diag = np.diag(W1)
    W2_diag = np.diag(W2)

    omega_denom = W1_diag[iu[0]] + W1_diag[iu[1]] + W2_diag[iu[0]] + W2_diag[iu[1]] + epsilon
    omega = 1.0 / omega_denom
    diff_sq = (M1[iu] - M2[iu]) ** 2
    

    numerator = np.sum(omega * diff_sq)
    denominator = np.sum(omega)
    

    D_sce = np.sqrt(numerator / denominator)
    
    return float(D_sce)