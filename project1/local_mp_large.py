from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures

from numpy import genfromtxt
import collections
import numpy as np
import csv
import time
import math
from itertools import permutations
from tqdm import tqdm
import random
import networkx as nx

def bayesian_score(D, G, node_values):

    parents = collections.defaultdict(list)

    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] == 1:
                parents[j].append(i)
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

    return score, parents

def random_directed_graph(num_node, p=0.2):
    # generate arbitrary ordering of nodes
    Graph = np.zeros([num_node, num_node])

    for parent in range(num_node):
        for node in range(parent+1, num_node):
            if np.random.uniform() < p:
                Graph[parent, node] = 1
    return Graph

def rand_graph_neighbor(Graph):
    i = np.random.randint(Graph.shape[0])
    j = (i + np.random.randint(1, Graph.shape[0])) % Graph.shape[0]
    Graph_ = Graph.copy()

    if Graph_[i, j] == 1:
        Graph_[i, j] = 0
    else:
        Graph_[i, j] = 1
    return Graph_


def dfs(Graph, i, visited):
    #print(i, visited)

    if i in visited:
        return True
    else:
        visited += [i]
    for j in range(Graph.shape[0]):
        if j == i:
            continue
        if Graph[i][j] > 0:
            return dfs(Graph, j, visited)
    return False


# def has_circle(Graph):
#     for i in range(Graph.shape[0]):
#         if dfs(Graph, i, []):
#             return True
#     return False

def has_circle(Graph):
    dag = nx.from_numpy_matrix(Graph, create_using=nx.DiGraph)
    try:
        nx.find_cycle(dag, orientation='original')
    except nx.exception.NetworkXNoCycle:
        return False
    return True

# Graph = np.zeros([5, 5])
# Graph = np.ones([5, 5])

# Graph = np.array([[0, 1, 0, 0],
#          [0, 0, 1, 0],
#          [0, 0, 0, 1],
#          [0, 0, 0, 0]])
# print(has_circle(Graph))

def local_search(D, node_values, data_size, headers):

    G = random_directed_graph(D.shape[1])
    while has_circle(G):
        G = random_directed_graph(D.shape[1])
    best_score, parents = bayesian_score(D, G, node_values)
    for k in range(9999999999):
        G_= rand_graph_neighbor(G)
        if has_circle(G_):
            continue
        else:
            current_score, parents = bayesian_score(D, G_, node_values)

            if current_score > best_score:
                best_score = current_score
                text_file = open(f"{data_size}_results/{best_score}_{data_size}_dev.gph", "w")
                for (key, values) in parents.items():
                    for val in values:
                        text_file.write(headers[val] + "," + headers[key] + "\n")
                text_file.close()
                print(f"current best score: {best_score}")
                G = G_.copy()

    return G

if __name__ == "__main__":
    data_size="large"
    raw_data=f"./data/{data_size}.csv"
    f = open(raw_data, "r")
    reader = csv.reader(f)
    headers = next(reader)
    f.close()
    D = genfromtxt(raw_data, delimiter=",", skip_header=1)

    start_time = time.time()
    score_arr = []
    score_cache = {}

    node_values = collections.defaultdict(list)
    for i in range(D.shape[1]):
        node_values[i] = list(np.unique(D[:, i]))

    futures_list = []
    results = []
    best_score = None
    l = []
    for i in range(100):
        l.append(random_directed_graph(D.shape[1]))
    print(len(l))

    total = len(l)
    consumed = 0

    #local_search(D,node_values, data_size,headers)
    with ProcessPoolExecutor(max_workers=96) as executor:
        for G in tqdm(l):
            futures = executor.submit(local_search, D,node_values, data_size,headers)
            futures_list.append(futures)

        for future in tqdm(futures_list):
            result = future.result()
            results.append(result)
            print(results)
                

    print(f"It took about: {(time.time() - start_time)} seconds")
