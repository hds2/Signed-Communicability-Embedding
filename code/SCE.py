# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import csv
import gc
import time
import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Iterable, Set

import numpy as np
import pandas as pd
import networkx as nx

try:
    from utils import compute_sce_representation, calculate_sce_dissimilarity
except ImportError:
    raise ImportError("错误: 当前目录下找不到 'utils.py' 或其中没有 'compute_sce_representation' 等新函数。")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)

def _edge_sign(weight: float) -> int:
    if weight > 0:
        return +1
    if weight < 0:
        return -1
    return 0

def list_edges_by_sign(G: nx.Graph) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    pos_edges: List[Tuple[int, int]] = []
    neg_edges: List[Tuple[int, int]] = []
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        s = _edge_sign(w)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if s > 0:
            pos_edges.append((a, b))
        elif s < 0:
            neg_edges.append((a, b))
    return pos_edges, neg_edges

def dataset_name_from_path(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]

def load_signed_graph_from_csv(
    path: str,
    *,
    make_undirected: bool = True,
    aggregate: Literal["sum_sign", "last"] = "sum_sign"
) -> nx.Graph:
    """
    读取 CSV：至少包含列 source,target,sign（sign ∈ {+1,-1,0} 或任意正负整数）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")

    edge_acc: Dict[Tuple[int, int], int] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [c.strip() for c in reader.fieldnames]

        for row in reader:
            try:
                u = int(row["source"])
                v = int(row["target"])
                s = int(row["sign"])
            except Exception:
                continue

            if s == 0 or u == v:
                continue

            s = 1 if s > 0 else -1

            if make_undirected:
                a, b = (u, v) if u < v else (v, u)
            else:
                a, b = u, v

            key = (a, b)
            if aggregate == "last":
                edge_acc[key] = s
            else:
                edge_acc[key] = edge_acc.get(key, 0) + s

    G = nx.Graph()
    nodes = set()
    for (u, v) in edge_acc.keys():
        nodes.add(u)
        nodes.add(v)
    G.add_nodes_from(sorted(nodes))

    for (u, v), ssum in edge_acc.items():
        if aggregate == "sum_sign":
            if ssum == 0:
                continue
            w = 1 if ssum > 0 else -1
        else:
            w = 1 if ssum > 0 else -1
        G.add_edge(u, v, weight=float(w))

    return G

def cache_path_for_step(dataset_cache_dir: str, ptype: str, ratio: float, trial: int) -> str:
    rtag = f"r{int(round(ratio * 10)):02d}"
    ttag = f"t{trial:03d}"
    return os.path.join(dataset_cache_dir, ptype, rtag, f"{ttag}.npz")

def save_matrices_npz(file_path: str, nodes: np.ndarray, A: np.ndarray, W: np.ndarray, M: np.ndarray) -> None:
    ensure_dir(os.path.dirname(file_path))
    np.savez_compressed(
        file_path,
        nodes=nodes,
        A=A.astype(np.float32),
        W=W.astype(np.float32),
        M=M.astype(np.float32)
    )

def load_matrices_npz(file_path: str) -> Dict[str, np.ndarray]:
    data = np.load(file_path, allow_pickle=False)
    return {k: data[k] for k in data.files}

def compute_and_cache_sce_metrics(A: np.ndarray, nodes: np.ndarray, file_path: str, c: float = 0.5) -> Dict[str, np.ndarray]:
    if os.path.exists(file_path):
        return load_matrices_npz(file_path)
    W_beta, M_beta = compute_sce_representation(A, c=c)

    save_matrices_npz(file_path, nodes, A, W_beta, M_beta)
    
    return {
        "W": W_beta.astype(np.float32),
        "M": M_beta.astype(np.float32),
    }



def delete_edges(G: nx.Graph, ratio: float, which: str, seed: int) -> nx.Graph:
    Gp = G.copy()
    rng = make_rng(seed)
    pos, neg = list_edges_by_sign(Gp)
    if which == "pos":
        pool = pos
    elif which == "neg":
        pool = neg
    elif which == "all":
        pool = pos + neg
    else:
        return Gp

    m = len(pool)
    k = int(round(ratio * m))
    if m > 0 and k > 0:
        idx = rng.choice(m, size=min(k, m), replace=False)
        for i in idx:
            u, v = pool[i]
            if Gp.has_edge(u, v):
                Gp.remove_edge(u, v)
    return Gp

def _sample_non_edge(G: nx.Graph, nodes: List[int], rng: np.random.Generator, max_tries: int = 2000):
    n = len(nodes)
    for _ in range(max_tries):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        u, v = nodes[i], nodes[j]
        if not G.has_edge(u, v):
            return (u, v)
    return None

def add_edges(
    G: nx.Graph,
    ratio: float,
    which: str,
    ref_counts: Tuple[int, int, int],
    seed: int,
    mode: Literal["match_original", "balanced"] = "match_original",
) -> nx.Graph:
    m_pos, m_neg, m_all = ref_counts
    Gp = G.copy()
    rng = make_rng(seed)
    nodes = list(G.nodes())

    if which == "pos":
        k = int(round(ratio * m_pos))
        w_fix = 1.0
    elif which == "neg":
        k = int(round(ratio * m_neg))
        w_fix = -1.0
    elif which == "rand":
        k = int(round(ratio * m_all))
        w_fix = None
    else:
        return Gp

    if k <= 0:
        return Gp

    if w_fix is None:
        if mode == "balanced":
            p_pos = 0.5
        else:
            p_pos = (m_pos / m_all) if m_all > 0 else 0.5
    else:
        p_pos = 1.0 if w_fix > 0 else 0.0

    added = 0
    max_total_tries = k * 50
    tries = 0

    while added < k and tries < max_total_tries:
        tries += 1
        uv = _sample_non_edge(Gp, nodes, rng)
        if uv is None:
            break
        u, v = uv

        if w_fix is not None:
            w = w_fix
        else:
            w = 1.0 if rng.random() < p_pos else -1.0

        Gp.add_edge(u, v, weight=w)
        added += 1

    return Gp



@dataclass
class ExperimentConfig:
    ratios: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    n_trials: int = 1
    rand_add_sign_mode: Literal["match_original", "balanced"] = "match_original"

def get_completed_tasks(csv_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return done
            for row in reader:
                p = row.get("perturbation")
                r = row.get("ratio")
                t = row.get("trial")
                if p and r and t:
                    r_fmt = f"{float(r):.1f}"
                    done.add(f"{p}_{r_fmt}_{t}")
    except Exception:
        pass
    return done

def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))

def is_dataset_done(dataset_dir: str, out_detail_path: str, cfg: ExperimentConfig) -> bool:
    done_marker = os.path.join(dataset_dir, "DONE.marker")
    if os.path.exists(done_marker):
        return True

    perturbations = ["del_pos", "add_pos", "del_neg", "add_neg", "del_rand", "add_rand"]
    total_tasks = len(perturbations) * len(list(cfg.ratios)) * cfg.n_trials
    completed = len(get_completed_tasks(out_detail_path))
    return completed >= total_tasks

def write_done_marker(dataset_dir: str) -> None:
    ensure_dir(dataset_dir)
    marker = os.path.join(dataset_dir, "DONE.marker")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"done_at={datetime.datetime.now().isoformat()}\n")

def run_experiment_incremental(
    G0: nx.Graph,
    cfg: ExperimentConfig,
    dataset_name: str,
    base_cache_dir: str,
    out_detail_path: str,
    seed: int
) -> None:
    rng = make_rng(seed)
    nodelist = sorted(G0.nodes())
    nodes_arr = np.asarray(nodelist, dtype=np.int64)
    n_nodes = len(nodelist)

    file_exists = os.path.exists(out_detail_path)
    csv_file = open(out_detail_path, "a", newline="", encoding="utf-8")
    fieldnames = ["perturbation", "ratio", "trial", "distance", "cache_file"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    completed_keys = get_completed_tasks(out_detail_path)
    dataset_dir = os.path.join(base_cache_dir, dataset_name)
    ensure_dir(dataset_dir)

    print(f"🚀 [1/2] 正在计算基准 (dataset={dataset_name}, N={n_nodes})...")
    t0 = time.time()


    A0 = nx.to_numpy_array(G0, nodelist=nodelist, weight="weight", dtype=np.float64)
    

    base_metrics = compute_and_cache_sce_metrics(A0, nodes_arr, base_cache_file, c=0.5)

    print(f"    -> 基准计算完成，耗时: {format_time(time.time() - t0)}")

    del A0
    gc.collect()

    pos0, neg0 = list_edges_by_sign(G0)
    ref_counts = (len(pos0), len(neg0), len(pos0) + len(neg0))
    perturbations = ["del_pos", "add_pos", "del_neg", "add_neg", "del_rand", "add_rand"]

    total_tasks = len(perturbations) * len(list(cfg.ratios)) * cfg.n_trials
    processed_count = len(completed_keys)

    print(f"🚀 [2/2] 开始扰动实验 (总任务: {total_tasks}, 已完成/跳过: {processed_count})")

    start_t = time.time()
    run_count = 0

    for ptype in perturbations:
        for ratio in cfg.ratios:
            for trial in range(cfg.n_trials):
                task_key = f"{ptype}_{ratio:.1f}_{trial}"
                if task_key in completed_keys:
                    continue

                run_count += 1
                processed_count += 1

                if run_count > 1:
                    avg_t = (time.time() - start_t) / (run_count - 1)
                    eta = format_time(avg_t * (total_tasks - processed_count + 1))
                else:
                    eta = "计算中..."

                print(f"⏳ [{processed_count}/{total_tasks}] {ptype} (r={ratio}) | ETA: {eta}", end="\r", flush=True)

                t_step = time.time()
                sub_seed = int(rng.integers(0, 2**31 - 1))

                if ptype == "del_pos":
                    Gp = delete_edges(G0, ratio, "pos", sub_seed)
                elif ptype == "del_neg":
                    Gp = delete_edges(G0, ratio, "neg", sub_seed)
                elif ptype == "del_rand":
                    Gp = delete_edges(G0, ratio, "all", sub_seed)
                elif ptype == "add_pos":
                    Gp = add_edges(G0, ratio, "pos", ref_counts, sub_seed, cfg.rand_add_sign_mode)
                elif ptype == "add_neg":
                    Gp = add_edges(G0, ratio, "neg", ref_counts, sub_seed, cfg.rand_add_sign_mode)
                elif ptype == "add_rand":
                    Gp = add_edges(G0, ratio, "rand", ref_counts, sub_seed, cfg.rand_add_sign_mode)
                else:
                    Gp = G0.copy()

                Ap = nx.to_numpy_array(Gp, nodelist=nodelist, weight="weight", dtype=np.float64)
                del Gp

                fpath = cache_path_for_step(dataset_dir, ptype, float(ratio), trial)
                pert_metrics = compute_and_cache_sce_metrics(Ap, nodes_arr, fpath, c=0.5)
                del Ap

              
                d = calculate_sce_dissimilarity(
                    W1=base_metrics["W"], 
                    M1=base_metrics["M"], 
                    W2=pert_metrics["W"], 
                    M2=pert_metrics["M"]
                )
                
                del pert_metrics
                gc.collect()

                writer.writerow({
                    "perturbation": ptype,
                    "ratio": ratio,
                    "trial": trial,
                    "distance": d,
                    "cache_file": fpath,
                })
                csv_file.flush()

                print(f"✅ [{processed_count}/{total_tasks}] {ptype} (r={ratio}) | Dist: {d:.4f} | 耗时: {time.time()-t_step:.1f}s")

    csv_file.close()



def summarize_detail_to_summary(out_detail: str, out_summary: str) -> None:
    if not os.path.exists(out_detail):
        return
    df = pd.read_csv(out_detail)
    if df.empty:
        return

    summary = df.groupby(["perturbation", "ratio"])["distance"].agg(["mean", "std", "count"]).reset_index()
    max_dist = float(summary["mean"].max())

    if max_dist > 1e-12:
        summary["mean"] = summary["mean"] / max_dist
        summary["std"] = summary["std"] / max_dist
        print(f"   -> 全局最大距离: {max_dist:.6f}，已执行 0-1 映射。")
    else:
        summary["mean"] = 0.0
        summary["std"] = 0.0

    summary.rename(columns={"count": "n"}, inplace=True)
    summary.to_csv(out_summary, index=False)

def process_one_dataset_file(
    file_path: str,
    cfg: ExperimentConfig,
    cache_dir: str,
    out_dir: str,
    seed: int
) -> None:
    dataset_name = dataset_name_from_path(file_path)
    dataset_cache_dir = os.path.join(cache_dir, dataset_name)
    ensure_dir(dataset_cache_dir)

    out_detail = os.path.join(out_dir, f"results_detail_{dataset_name}.csv")
    out_summary = os.path.join(out_dir, f"results_summary_{dataset_name}.csv")

    if is_dataset_done(dataset_cache_dir, out_detail, cfg):
        print(f"⏭️  跳过已完成数据集: {dataset_name}")
        return

    print(f"\n{'='*70}")
    print(f"PROCESSING: {dataset_name}")
    print(f"FILE: {file_path}")
    print(f"{'='*70}")

    try:
        G0 = load_signed_graph_from_csv(file_path, make_undirected=True, aggregate="sum_sign")
    except Exception as e:
        print(f"❌ 加载失败: {dataset_name} | {e}")
        return

    n = G0.number_of_nodes()
    m = G0.number_of_edges()
    print(f"📊 实际读取：节点数 N={n}, 边数 M={m}")

    try:
        run_experiment_incremental(
            G0,
            cfg,
            dataset_name=dataset_name,
            base_cache_dir=cache_dir,
            out_detail_path=out_detail,
            seed=seed
        )

        print(f"📊 正在生成汇总表: {out_summary}")
        summarize_detail_to_summary(out_detail, out_summary)

        write_done_marker(dataset_cache_dir)
        print(f"🎉 数据集完成: {dataset_name}")

    except KeyboardInterrupt:
        print("\n⛔ 用户手动停止。")
        raise
    except Exception as e:
        print(f"\n❌ 数据集出错: {dataset_name} | {e}")
        import traceback
        traceback.print_exc()


def main():
    DATA_FILE = "data/slashdot.csv" 
    CACHE_DIR = "cache_matrices"
    OUT_DIR = "results_csv"
    TRIALS = 1

    if not os.path.isfile(DATA_FILE):
        print(f"❌ 找不到指定的数据文件: {DATA_FILE}")
        return

    ensure_dir(CACHE_DIR)
    ensure_dir(OUT_DIR)

    cfg = ExperimentConfig(
        ratios=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        n_trials=TRIALS,
        rand_add_sign_mode="match_original",  
    )

    print(f"🎯 数据文件: {DATA_FILE}")
    print(f"📂 缓存目录: {CACHE_DIR}")
    print(f"🧾 输出目录: {OUT_DIR}")

    # 直接调用处理单个文件
    process_one_dataset_file(
        DATA_FILE,
        cfg=cfg,
        cache_dir=CACHE_DIR,
        out_dir=OUT_DIR,
        seed=2025
    )
    
    print("\n✅ 数据处理完毕。")


if __name__ == "__main__":
    main()