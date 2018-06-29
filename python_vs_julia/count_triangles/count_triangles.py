
import json 

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


#### function for counting undirected cycles
def generate_triangles_2(nodes):
    result = []
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
                    result.append((node_a_id, node_b_id, node_c_id))
                else:
                    continue
            temp_visited.add(node_b_id)
        visited_ids.add(node_a_id)
        
    return result

with open('graph.json', 'r') as fp:
    adj_dict = json.load( fp)

adj_dict_ = {}
for key in adj_dict:
    adj_dict_[int(key)] =  adj_dict[key]
del adj_dict


import time

t0 = time.time()
print("Start Computing")
for i in range(1000):
    cycles = generate_triangles_2(adj_dict_)
cycles = list(cycles)
print("Number of triangles : {}".format(len(cycles)))
print( "Total time: {} seconds".format(abs(t0- time.time()))) 


