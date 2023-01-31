from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from numpy import genfromtxt
import collections
import numpy as np
import csv
import time
import math
from itertools import permutations
from tqdm import tqdm


def bayesian_score(D, parents, node_values):
    M = collections.defaultdict(int)
    Alpha_sum = collections.defaultdict(int)
    M_sum = collections.defaultdict(int)

    num_data, num_node = D.shape

    for data_i in range(num_data):
        for node_j in range(num_node):
            val = D[data_i, node_j]
            parent = parents[node_j]
            parent_vals = D[data_i, parent]
            M[node_j, tuple(parent_vals), val] += 1
            M_sum[node_j, tuple(parent_vals)] += 1

    for key in M_sum.keys():
        node = key[0]
        Alpha_sum[key] = len(node_values[node])

    score = 0
    for key in M.keys():
        score = score + math.lgamma(1 + M[key])

    for key in M_sum.keys():
        score = (
            score
            + math.lgamma(Alpha_sum[key])
            - math.lgamma(Alpha_sum[key] + M_sum[key])
        )

    return score


def k2_search(D, topologicalOrder):
    num_data, num_node = D.shape
    node_values = collections.defaultdict(list)
    for i in range(num_node):
        node_values[i] = list(np.unique(D[:, i]))

    Graph = np.zeros([num_node, num_node])
    parents = collections.defaultdict(list)

    best_score = bayesian_score(D, parents, node_values)

    # adding edges
    for topo_idx in range(len(topologicalOrder)):
        node = topologicalOrder[topo_idx]
        for parent_idx in range(topo_idx):
            parent = topologicalOrder[parent_idx]
            if parent not in parents[node]:
                Graph[parent, node] = 1
                parents[node].append(parent)
                current_score = bayesian_score(D, parents, node_values)
                if current_score > best_score:
                    best_score = current_score
                else:
                    parents[node].pop()
                    Graph[parent, node] = 0

    return (Graph, parents, best_score)


if __name__ == "__main__":
    f = open("./data/small.csv", "r")
    reader = csv.reader(f)
    headers = next(reader)
    f.close()
    D = genfromtxt("./data/small.csv", delimiter=",", skip_header=1)
    start_time = time.time()
    score_arr = []
    score_cache = {}

    l = list(permutations(range(D.shape[1])))
    print(len(l))
    futures_list = []
    results = []
    best_score = None
    with ProcessPoolExecutor(max_workers=16) as executor:
        for topologicalOrder in l:
            topologicalOrder = np.array(topologicalOrder)
            futures = executor.submit(k2_search, D, topologicalOrder)
            futures_list.append(futures)

        for future in tqdm(futures_list):
            try:
                result = future.result()
                results.append(result)
                Graph, parents, current_score = result
                if best_score is None or current_score > best_score:
                    best_score = current_score
                    text_file = open(f"./small_results/{best_score}_small_dev.gph", "w")
                    for (key, values) in parents.items():
                        for val in values:
                            text_file.write(headers[val] + "," + headers[key] + "\n")
                    text_file.close()
                    print(f"current best score: {best_score}")

            except Exception:
                results.append(None)    
