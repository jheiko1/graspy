def mfaq(
    graph1, graph2, p_init, weights = 1, scheme = 'naive', tolerance = 1)

    # let's just assume everything will be a 3d array with the third dimension
    # being layers, and that import_graph can handle 3d

    nlayers = np.shape(graph1)[0]
    pverts = np.shape(graph2)[1]
    if weights == 1:
        weights = np.ones(pverts)

    graph1 = graph_pad(graph1, pverts, scheme)

    perm0 = np.zeros((pverts,pverts))
    perm1 = p_init

    while np.linalg.norm((perm1-perm0), ord = 'fro') > tolerance:
        temp_grad = np.empty([nlayers,pverts,pverts])

        for i in range(nlayers):
            graph_a = graph1[i,:,:]
            graph_aT = np.transpose(graph_a)
            graph_b = graph2[i,:,:]
            graph_bT = np.transpose(graph_b)

            dummy = (
            np.matmul(np.matmul(graph_aT, perm1), graph_b) +
            np.matmul(np.matmul(graph_a, perm1), graph_bT)
            )
            temp_grad[i,:,:] = np.dot(weights[i], dummy)

        grad_perm = np.sum(temp_grad, axis=0)

        # pretty sure this next step is finding the doubly stochastic matrix Q
        # that maximizes the trace of grad.T * Q but Idk how
        # maybe hungarian algo? scipy.optimize.linear_sum_assignment
        # ask Ali

        perm0 = perm1
        perm1 = (alpha*perm1) + (1 - alpha)*q_perm

    # find a permutation matrix P that maximizes the trace of P.T * Pf where
    # Pf is the final perm1 from the while loop
    # outputs P
