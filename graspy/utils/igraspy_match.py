def graph_match_fw(
    graph1, graph2, seeds = None, start = 'convex', max_iter = 20,
    similarity = None):

    graph1 = import_graph(graph1)
    graph2 = import_graph(graph2)

    nverts1 = np.shape(graph1)[1]
    nverts2 = np.shape(graph2)[2]

    if nverts1 > nverts2:
        graph2 = graph_pad(graph2, pverts=nverts1, scheme='naive')
    elif nverts1 < nverts2:
        graph1 = graph_pad(graph1, pverts=nverts2, scheme='naive')

    nverts = max(nverts1,nverts2)

    if seeds is None:
        seeds = np.full((nverts, 1), False)
        seed_err = False
        nseeds = 0
    else:
        nseeds = len(seeds)
        # create a vector of zeros length nverts, insert 1 where dicated by seeds
        # binary vector for disagreements in seedA and seedB
        #

    'May be better to work with seeded-FAQ instead of this'
