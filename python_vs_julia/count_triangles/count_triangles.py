
# https://www.geeksforgeeks.org/number-of-triangles-in-directed-and-undirected-graphs/
import time
import networkx 
import numpy as np
import pandas as pd

def countTriangle(g, isDirected): 
    nodes = len(g) 
    count_Triangle = 0 #Initialize result 
    # Consider every possible triplet of edges in graph 
    for i in range(nodes): 
        for j in range(nodes): 
            for k in range(nodes): 
                # check the triplet if it satisfies the condition 
                if( i!=j and i !=k and j !=k and 
                        g[i][j] and g[j][k] and g[k][i]): 
                    count_Triangle += 1
    # if graph is directed , division is done by 3 
    # else division by 6 is done 
    return count_Triangle/3 if isDirected else count_Triangle/6
  
# Create adjacency matrix of an undirected graph 
graph = [[0, 1, 1, 0], 
         [1, 0, 1, 1], 
         [1, 1, 0, 1], 
         [0, 1, 1, 0]] 

  
n_nodes = 1000
n_edges = 10

print("The Number of nodes {}, number of expected edges per node: {} ".format(n_nodes, n_edges)) 

G = networkx.barabasi_albert_graph(n_nodes, n_edges, seed=1) 
adj_list = [(n, nbrdict) for n, nbrdict in G.adjacency()]

def create_adjacency_matrix(adj_list):
    n_nodes = len(adj_list)
    graph = [[0]*n_nodes for node in range(n_nodes)]
    for i,conections in adj_list:  
        positions_with_ones = list(conections.keys())
        for k in positions_with_ones:
            graph[i][k] = np.int8(1)
    return graph

graph = create_adjacency_matrix(adj_list)
g_df = pd.DataFrame(graph)
g_df.to_csv("graph.csv")

t0 = time.time()
print("Start computing")
n_triangles = countTriangle(graph, False)
print("The Number of triangles in undirected graph: ", n_triangles ) 
t = time.time()-t0
print("Total time: {} seconds".format(t))
