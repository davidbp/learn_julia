
import json 


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
    cycles = generate_triangles(adj_dict_)

import pdb;pdb.set_trace()
cycles = list(cycles)
print("Number of triangles : {}".format(len(cycles)))
print( "Total time: {} seconds".format(abs(t0- time.time()))) 


