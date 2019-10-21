import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import random
import utils as utils


nverts = 8
clayers = 5

nodesnumlist = np.random.randint(int(nverts/2), nverts, clayers)
nodeslist = [random.sample(range(nverts),nodesnumlist[index]) for index in range(clayers)]
notnodes = [[x for x in range(8) if x not in nodeslist[ind]] for ind in range(clayers)]

bg_network = [nx.binomial_graph(nverts,0.5) for index in range(clayers)]
bg_network = [utils.import_graph(x) for x in bg_network]
bg_network = np.array(bg_network)
# remove the verts associated with notnodes in each layer (set to 0)
for layer in range(clayers):
    bg_network[layer,notnodes[layer],:] = 0
    bg_network[layer,:,notnodes[layer]] = 0

# take subset of nodes
tempnodes = random.sample(range(nverts),4)
tem_graph = bg_network[:,tempnodes,:]
tem_graph = tem_graph[:,:,tempnodes]
# random perm mat
p_matrix = np.identity(4)
p_matrix = np.random.permutation(p_matrix)
print(p_matrix)

# test MFAQ on the subset
# find error
