### generate graph and save it 

import networkx
import json
import argparse

print("networkx.__version__:", networkx.__version__)

parser = argparse.ArgumentParser(description="script generate_graph, it uses a Albert Barabasi random graph generator")
parser.add_argument('-n', '--n_nodes', type=int, required=False, help='Number of nodes to generate')
parser.add_argument('-e', '--n_edges', type=int, required=False, help='Number of edges to attach from a new node to existing nodes')
args = parser.parse_args()

#n_nodes = int(input("\n\nPlease enter Number of nodes in the random graph: "))

if __name__ == "__main__":


    ##### Argparsing ###############################
    if args.n_nodes :
        n_nodes = args.n_nodes 
        print("\n\tUser introduced {} nodes".format(n_nodes))
    else:
        n_nodes = 1000
        print("\n\tUser did not introduced n_nodes, using default value {}".format(n_nodes))

    if args.n_edges:
        n_edges = args.n_edges 
        print("\n\tUser introduced {edges} nodes".format(n_edges))
    else:
        n_edges = 10
        print("\tUser did not introduced n_edges, using default value {}".format(n_edges))


    ################################################

    G = networkx.barabasi_albert_graph(n_nodes, n_edges, seed=1) 

    #adj_list = G.adjacency_list() 
    adj_list = list(G.adjacency()) 
    n_nodes  = len(adj_list)
    adj_dict = {i:adjacent_i for i,adjacent_i in zip(range(n_nodes), adj_list)}

    with open('graph.json', 'w') as fp:
        json.dump(adj_dict, fp)

    print("\tGraph:   `graph.json`    generated")
    print("\n")