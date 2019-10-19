import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize_scalar
for sinkhorn_knopp import sinkhorn_knopp as skp

def mfaq(
    graph1,
    graph2,
    p_init = 'random',
    weights = 1,
    scheme = 'naive',
    tolerance = 1
    ):

    # let's just assume everything will be a 3d array with the third dimension
    # being layers, and that import_graph can handle 3d

    nlayers = np.shape(graph1)[0]
    pverts = np.shape(graph2)[1]
    if weights == 1:
        weights = np.ones(pverts)

    # write better padding code
    graph1 = graph_pad(graph1, pverts, scheme)
    nverts = pverts

    perm0 = np.identity(nverts)
    perm1 = p_init
    if p_init == 'random':
        perm1 = random_DSM(nverts)

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

    # end while
    perm_T = np.transpose(perm1)
    prows, pcols = linear_sum_assignment(-perm_T)
    pfinal = np.zeros((nverts,nverts))
    pfinal[prows, pcols] = 1

    return pfinal


def random_DSM(nverts):
    sk = skp.SinkhornKnopp()
    random_mat = np.random.rand(nverts,nverts)
    for i in range(10):
        random_mat = sk.fit(random_mat)
    double_stoch_bary = np.ones((nverts,nverts))/float(nverts)
    out = (random_mat + double_stoch_bary)/2
    return out
