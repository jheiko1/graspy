import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize_scalar
from sinkhorn_knopp import sinkhorn_knopp as skp

def mfaq(graph1, graph2, p_init = 'random', weights = 1,
         scheme = 'naive', tolerance = 1.0):

    """
    A function for returning the permutation matrix P associated with the
    template network graph1 and the "best fitting" subgraph in the larger
    background network graph2.

    Parameters
    ----------
    graph1: array-like
        Shape (c_layers, m_vertices, m_vertices) numpy array. Adjacency matrix
        for the multiplex template network G.

    graph2: array-like
        Shape (c_layers, n_vertices, n_vertices) numpy array. Adjacency matrix
        for the multiplex background network H, where n_vertices >> m_vertices.

    p_init: array-like, optional (default = 'random')
        Shape (n_vertices, n_vertices) numpy array. Initialization for the
        permutation matrix P.
        If unspecified, will call on random_DSM to initialize as a random
        permutation matrix

    weights: array-like, optional (default = 1)
        A vector of length (c_layers), where the ith entry corresponds to the
        weight given to the ith layer.
        If unspecified, each layer will be given equal weight.

    scheme: str, optional (default = 'naive')
        Padding scheme for the template multiplex network.

        'naive' (default)
            Pads the template network with 0 to match the size of the background
            network

        'centered'
            Modifies the template network according to the centered padding scheme
            of graph_pad, then pads with 0 to match the size of the background
            network.

    tolerance: float, optional (default = 1.0)
        The value of the Frobenius norm between each iteration of P that MFAQ will
        converge to before terminating.

    Returns
    -------
    pfinal: array-like, shape (n_vertices, n_vertices)
        A permutation matrix

    See Also
    --------
    numpy.array, sinkhorn_knopp.SinkhornKnopp
    """

    graph1 = import_graph(graph1)
    graph2 = import_graph(graph2)
    shape1 = graph1.shape
    shape2 = graph2.shape

    if (len(shape1) != 3) or (len(shape2) != 3):
        msg = "Both input networks must have 3 dimensions."
        raise ValueError(msg)
    if shape1[0] != shape2[0]:
        msg = "Input networks must have matching numbers of layers"
        raise ValueError(msg)
    if shape1[1] > shape2[1]:
        msg: "Template network must have fewer vertices than background tensor"
        raise ValueError(msg)

    nlayers = shape1[0]
    pverts = np.shape(graph2)[1]
    if weights == 1:
        weights = np.ones(pverts)

    # write better padding code
    graph1 = graph_pad(graph1, pverts, scheme)
    nverts = pverts

    perm0 = np.identity(nverts)

    if p_init == 'random':
        perm1 = random_DSM(nverts, permutation = True)
    else:
        perm1 = p_init

    while np.linalg.norm((perm1-perm0), ord = 'fro') > tolerance:

        def perm_grad_sum(ind):
            graph_a = graph1[ind,:,:]
            graph_aT = np.transpose(graph_a)
            graph_b = graph2[ind,:,:]
            graph_bT = np.transpose(graph_b)

            out = (graph_aT@perm1@graph_b) + (graph_a@perm1@graph_bT)
            out = weights[ind] * out
            return out

        perm_grad = np.array(
        [perm_grad_sum(index) for index in range(nlayers)]).sum(axis=0)

        qrows, qcols = linear_sum_assignment(-perm_grad)
        q_perm = np.zeros((nverts,nverts))
        q_perm[qrows, qcols] = 1
        q_perm = q_perm.transpose

        def alpha_max(alpha):
            def alpha_sum(ind):
                graph_a = graph1[ind,:,:]
                graph_b = graph2[ind,:,:]
                q_alpha = (alpha*perm1) + (1-alpha)*q_perm
                q_alpha_T = np.transpose(q_alpha)

                out = graph_a@q_alpha@graph_b@q_alpha_T
                out = np.trace(out)
                out = weights[i]*out
                return out

            out = np.array([alpha_sum(index) for index in range(nlayers)]).sum()
            return out

        alpha = minimize_scalar(alpha_max, bounds = (0,1), method = 'bounded')

        perm0 = perm1
        perm1 = (alpha*perm1) + (1 - alpha)*q_perm

    perm_T = np.transpose(perm1)
    prows, pcols = linear_sum_assignment(-perm_T)
    pfinal = np.zeros((nverts,nverts))
    pfinal[prows, pcols] = 1
    pfinal = pfinal.transpose

    return pfinal


def random_DSM(nverts, permutation = 'False'):
    sk = skp.SinkhornKnopp()
    random_mat = np.random.rand(nverts,nverts)

    if permutation = True:
        out = np.identity(nverts)
        out = np.random.permutation(out)
        return (out)

    for i in range(10):
        random_mat = sk.fit(random_mat)
    double_stoch_bary = np.ones((nverts,nverts))/float(nverts)
    out = (random_mat + double_stoch_bary)/2
    return out
