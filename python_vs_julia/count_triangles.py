
import networkx

#### function for counting undirected cycles
def generate_triangles(nodes):
    visited_ids = set() # mark visited node
    for node_a_id in nodes:
        temp_visited = set() # to get undirected triangles
        for node_b_id in nodes[node_a_id]:
            if node_b_id == node_a_id:
                # to prevent self-loops, if your graph allows self-loops then you don't need this condition
                raise ValueError 
            if node_b_id in visited_ids:
                continue
            for node_c_id in nodes[node_b_id]:
                if node_c_id in visited_ids:
                    continue    
                if node_c_id in temp_visited:
                    continue
                if node_a_id in nodes[node_c_id]:
                    yield(node_a_id, node_b_id, node_c_id)
                    #visited_ids.add((node_a_id, node_b_id, node_c_id))
                else:
                    continue
            temp_visited.add(node_b_id)
        visited_ids.add(node_a_id)


node_number   = 100
initial_nodes = 2
G = networkx.barabasi_albert_graph(node_number, initial_nodes, seed=1) 

adj_list = G.adjacency_list() 
n_nodes  = len(adj_list)
adj_dict = {i:adjacent_i for i,adjacent_i in zip(range(n_nodes), adj_list)}

cycles = generate_triangles(adj_dict)
cycles = list(cycles)
print(cycles)
