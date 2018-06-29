### generate graph and save it 

import networkx
import json


var = input("\n\nPlease enter Number of nodes in the random graph: ")

node_number   = var
initial_nodes = 2
G = networkx.barabasi_albert_graph(node_number, initial_nodes, seed=1) 

adj_list = G.adjacency_list() 
n_nodes  = len(adj_list)
adj_dict = {i:adjacent_i for i,adjacent_i in zip(range(n_nodes), adj_list)}


with open('graph.json', 'w') as fp:
    json.dump(adj_dict, fp)

print("Graph:   `graph.json`    generated")
print("\n")