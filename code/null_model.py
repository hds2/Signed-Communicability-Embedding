"""
基于真实世界符号网络构建的5种零模型 (Null Models)：
    1. Positive rewire (正边重连零模型)
    2. Negative rewire (负边重连零模型)
    3. Signed rewire (符号分离重连零模型)
    4. Full rewire (全局重连零模型)
    5. Sign shuffle (符号洗牌零模型)
全局声明：
    +: sign 1
    -: sign -1
"""

import networkx as nx
import random
import copy
import pandas as pd
import time
import os


def positive_rewire(G0, f, max_tries_multiplier=10):
    """
    :description: Positive rewire (正边重连零模型)
    仅重连正边，负边保持不变。随机选择两条正边并交换其端点。
    严格保留正边总数和正子图的度序列。
    """
    G = copy.deepcopy(G0)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign") == 1]
    m_pos = len(pos_edges)
    nswap = int(f * m_pos)  
    
    if nswap == 0:
        return G

    max_tries = nswap * max_tries_multiplier
    tries = 0
    swaps = 0

    while swaps < nswap and tries < max_tries:
        tries += 1
    
        i, j = random.sample(range(m_pos), 2)
        u, v = pos_edges[i]
        x, y = pos_edges[j]


        if len({u, v, x, y}) < 4:
            continue


        if random.random() < 0.5:
            if not G.has_edge(u, y) and not G.has_edge(x, v):
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, y, sign=1)
                G.add_edge(x, v, sign=1)
                pos_edges[i] = (u, y)
                pos_edges[j] = (x, v)
                swaps += 1
        else:
            if not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, x, sign=1)
                G.add_edge(v, y, sign=1)
                pos_edges[i] = (u, x)
                pos_edges[j] = (v, y)
                swaps += 1

    print(f'[Positive rewire] 目标: {nswap}, 成功重连: {swaps}, 尝试次数: {tries}')
    return G


def negative_rewire(G0, f, max_tries_multiplier=10):
    """
    :description: Negative rewire (负边重连零模型)
    仅重连负边，正边保持不变。随机选择两条负边并交换其端点。
    严格保留负边总数和负子图的度序列。
    """
    G = copy.deepcopy(G0)

    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign") == -1]
    m_neg = len(neg_edges)
    nswap = int(f * m_neg)
    
    if nswap == 0:
        return G

    max_tries = nswap * max_tries_multiplier
    tries = 0
    swaps = 0

    while swaps < nswap and tries < max_tries:
        tries += 1
        i, j = random.sample(range(m_neg), 2)
        u, v = neg_edges[i]
        x, y = neg_edges[j]

        if len({u, v, x, y}) < 4:
            continue

        if random.random() < 0.5:
            if not G.has_edge(u, y) and not G.has_edge(x, v):
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, y, sign=-1)
                G.add_edge(x, v, sign=-1)
                neg_edges[i] = (u, y)
                neg_edges[j] = (x, v)
                swaps += 1
        else:
            if not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, x, sign=-1)
                G.add_edge(v, y, sign=-1)
                neg_edges[i] = (u, x)
                neg_edges[j] = (v, y)
                swaps += 1

    print(f'[Negative rewire] 目标: {nswap}, 成功重连: {swaps}, 尝试次数: {tries}')
    return G


def signed_rewire(G0, f, max_tries_multiplier=10):
    """
    :description: Signed rewire (符号分离重连零模型)
    分别按比例 f 重连正边和负边。
    为避免正负边重连时产生重复边冲突，在同一张图上先后分别执行正负重连是符合逻辑的最优做法。
    """
    print(f"[Signed rewire] 开始分离重连...")
    G_p = positive_rewire(G0, f, max_tries_multiplier)
    G_res = negative_rewire(G_p, f, max_tries_multiplier)
    return G_res


def full_rewire(G0, f, max_tries_multiplier=10):
    """
    :description: Full rewire (全局重连零模型)
    不区分符号，随机挑选两条边交换端点，但保留边原本的符号。
    """
    G = copy.deepcopy(G0)
    # 提取所有边及其符号
    all_edges = [(u, v, d.get("sign")) for u, v, d in G.edges(data=True)]
    m_total = len(all_edges)
    nswap = int(f * m_total)
    
    if nswap == 0:
        return G

    max_tries = nswap * max_tries_multiplier
    tries = 0
    swaps = 0

    while swaps < nswap and tries < max_tries:
        tries += 1
        i, j = random.sample(range(m_total), 2)
        u, v, s1 = all_edges[i]
        x, y, s2 = all_edges[j]

        if len({u, v, x, y}) < 4:
            continue

        if random.random() < 0.5:
            if not G.has_edge(u, y) and not G.has_edge(x, v):
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, y, sign=s1)
                G.add_edge(x, v, sign=s2)
                all_edges[i] = (u, y, s1)
                all_edges[j] = (x, v, s2)
                swaps += 1
        else:
            if not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, x, sign=s1)
                G.add_edge(v, y, sign=s2)
                all_edges[i] = (u, x, s1)
                all_edges[j] = (v, y, s2)
                swaps += 1

    print(f'[Full rewire] 目标: {nswap}, 成功重连: {swaps}, 尝试次数: {tries}')
    return G


def sign_shuffle(G0, f, max_tries_multiplier=10):
    """
    :description: Sign shuffle (符号洗牌零模型)
    保持拓扑结构不变，仅交换异号边的符号。
    目标交换次数根据整体边数的比例 f 计算。
    """
    G = copy.deepcopy(G0)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign") == 1]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign") == -1]
    m_total = len(G.edges())
    nswap = int(f * m_total)
    
    if nswap == 0:
        return G

    max_tries = nswap * max_tries_multiplier
    tries = 0
    swaps = 0

    while swaps < nswap and tries < max_tries:
        tries += 1
        if not pos_edges or not neg_edges:
            break
            
        i = random.randrange(len(pos_edges))
        j = random.randrange(len(neg_edges))
        
        u, v = pos_edges[i]
        x, y = neg_edges[j]

        G[u][v]["sign"] = -1
        G[x][y]["sign"] = 1
        

        pos_edges[i] = (x, y)
        neg_edges[j] = (u, v)
        
        swaps += 1

    print(f'[Sign shuffle] 目标: {nswap}, 成功洗牌: {swaps}, 尝试次数: {tries}')
    return G


def convert_graph_to_dataframe(G):
    """
    将带 sign 属性的无向图转为 DataFrame：列为 [source, target, sign]
    """
    edge_list = []
    for u, v, data in G.edges(data=True):
        sign = data.get("sign", None)
        edge_list.append([u, v, sign])
    df = pd.DataFrame(edge_list, columns=['source', 'target', 'sign'])
    return df


if __name__ == '__main__':

    DATA_PATH = r"data/slashdot.csv"
    OUT_DIR = r"./null_model_output"
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到文件：{DATA_PATH}")
        exit()

    os.makedirs(OUT_DIR, exist_ok=True)
    
    dataset_name = os.path.splitext(os.path.basename(DATA_PATH))[0]

    start = time.time()
    print(f"正在读取数据：{DATA_PATH} ...")
    data = pd.read_csv(DATA_PATH, encoding="UTF-8")
    G0 = nx.from_pandas_edgelist(
        data, "source", "target", "sign", create_using=nx.Graph()
    )
    print(f'成功读取网络：{dataset_name}')
    print(f"节点数 N = {G0.number_of_nodes()}, 边数 M = {G0.number_of_edges()}")

    # 设置比例参数 f：从 0.1 到 0.9
    f_ratios = [i / 10 for i in range(1, 10)]

    for f in f_ratios:
        print("\n" + "="*40)
        print(f"当前重连/洗牌比例 f = {f}")
        print("="*40)

       
        G_pos = positive_rewire(G0, f)
        G_neg = negative_rewire(G0, f)
        G_signed = signed_rewire(G0, f)
        G_full = full_rewire(G0, f)
        G_shuffle = sign_shuffle(G0, f)

        suffix = f"_f{int(f * 100)}"

        
        convert_graph_to_dataframe(G_pos).to_csv(os.path.join(OUT_DIR, f"{dataset_name}_positive_rewire{suffix}.csv"), index=False)
        convert_graph_to_dataframe(G_neg).to_csv(os.path.join(OUT_DIR, f"{dataset_name}_negative_rewire{suffix}.csv"), index=False)
        convert_graph_to_dataframe(G_signed).to_csv(os.path.join(OUT_DIR, f"{dataset_name}_signed_rewire{suffix}.csv"), index=False)
        convert_graph_to_dataframe(G_full).to_csv(os.path.join(OUT_DIR, f"{dataset_name}_full_rewire{suffix}.csv"), index=False)
        convert_graph_to_dataframe(G_shuffle).to_csv(os.path.join(OUT_DIR, f"{dataset_name}_sign_shuffle{suffix}.csv"), index=False)

        print(f"✔ 比例 f = {f} 的五种零模型已成功保存。")

    end = time.time()
    print(f'\n程序总运行时间：{end - start:.2f} 秒')