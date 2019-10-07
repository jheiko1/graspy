import warnings
from collections import Iterable
from functools import reduce
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.utils import check_array

def import_seeds(seeds, nverts = 0):
    """
    A function for reading a vector of seeds and returning an array

    Parameters
    ----------
    seeds: object

    nverts: int
        Number of total vertices
    """

def graph_padding(graph, pverts, scheme = 'centered'):
    """
    A function for taking a graph and applying a selected padding approach to it.

    Parameters
    ----------
    graphs: object
        Either array-like, shape (n_vertices, n_vertices) numpy array, or an object of type networkx.Graph.

    pverts: int
        The number of vertices there will be in the graph after padding

    scheme: str, (default='centered')
        When scheme = 'naive', A' = A (direct_sum) np.zeros(pverts-nverts)
        When scheme = 'centered', A' = (2A-np.ones(n,n)) (direct_sum) np.zeros(pverts-nverts)
        ADD FORMATTING AFTER LEARNING NOTATION IN SPHINX

    Returns
    ----------
    out: array-like, shape(pverts,pverts)
        The padded graph.

    """

    graph = import_graph(graph)
    nverts = np.shape(graph)[0]
    xverts = pverts - nverts

    if (scheme=='naive'):
        graph = np.pad(graph,((0,xverts),(0,xverts)),'constant', constant_values = 0)

    if (scheme=='centered'):
        graph = (2*graph) - np.ones((nverts,nverts))
        graph = np.pad(graph,((0,xverts),(0,xverts)),'constant', constant_values = 0)
