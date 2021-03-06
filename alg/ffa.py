# modified from NetworkX package

# Copyright (C) 2010 Loïc Séguin-C. <loicseguin@gmail.com>
# All rights reserved.
# BSD license.


from lib import *


# Global Variables
BackwardFlow = 'back_cap'
Capacity = 'weight'
FFA_FlowName = 'ffa'


# ============================================================


def ford_fulkerson_impl(G, s, t, path_decomposition=False, cutoff=0.0001):
    # TODO: implement cutoff?
    auxiliary = _create_auxiliary_digraph(G)
    flow_value = 0   # Initial feasible flow.
    path_ls = []

    # As long as there is an (s, t)-path in the auxiliary digraph, find
    # the shortest (with respect to the number of arcs) such path and
    # augment the flow on this path.
    while True:
        try:
            path_capacity, path_nodes = widest_path(auxiliary, s, t)
        except nx.NetworkXNoPath:
            break
        if not path_nodes:
            break

        # Get the list of edges in the shortest path.
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

        # pathway decomposition
        path_ls.append([path_nodes, path_capacity])

        flow_value += path_capacity
        print_more('path: {}'.format(path_nodes))
        print_more('  flow: {:.3f}, total flow: {:.3f}'.format(path_capacity, flow_value))

        # Augment the flow along the path.
        for u, v in path_edges:
            edge_attr = auxiliary[u][v]
            # decrease the capacities along the path
            edge_attr[Capacity] -= path_capacity

            # for backward penalty
            # which disapprove to cancel the existing flow
            edge_attr[BackwardFlow] -= path_capacity
            if edge_attr[BackwardFlow] <= 0:
                edge_attr[BackwardFlow] = 0

            # delete edge if the edge has no capacity
            if edge_attr[Capacity] == 0:
                auxiliary.remove_edge(u, v)

            # increase the capacities of backward edges
            if auxiliary.has_edge(v, u):
                edge_attr = auxiliary[v][u]
                edge_attr[Capacity] += path_capacity
                # backward penalty
                edge_attr[BackwardFlow] += path_capacity

            else:
                auxiliary.add_edge(v, u)
                edge_attr = auxiliary[v][u]
                edge_attr[Capacity] = path_capacity
                edge_attr[BackwardFlow] = path_capacity

    # print path
    if path_decomposition:
        path_ls = sorted(path_ls, key=itemgetter(1), reverse=True)
        for p, c in path_ls:
            print(format(c / flow_value * 100, '.0f'), end='%: ')
            for i, v in enumerate(p):
                if i != 0:
                    print(' -> ', end='')
                nodes = v.split(' ')
                for n in nodes:
                    if not n.isdigit():
                        break
                else:
                    nodes = sorted(map(int, nodes))
                for n in nodes:
                    if n != nodes[-1]:
                        print(n, end=',')
                    else:
                        print(n, end='')
            print()

    # remove all redundant edges
    for u, v, w in tuple(auxiliary.edges(data=Capacity)):
        if w == 0:
            auxiliary.remove_edge(u, v)

    return flow_value, auxiliary


def _create_auxiliary_digraph(G):

    auxiliary = nx.DiGraph()
    auxiliary.add_nodes_from(G)
    for edge in G.edges(data=True):
        auxiliary.add_edge(edge[0], edge[1])
        edge_attr = auxiliary[edge[0]][edge[1]]
        edge_attr[Capacity] = edge[2][Capacity]
        edge_attr[BackwardFlow] = 0

    return auxiliary


def _create_flow_dict(G, H):
    flow = dict([(u, {}) for u in G])

    for u, v in G.edges():
        if H.has_edge(u, v):
            if Capacity in G[u][v]:
                flow[u][v] = max(0, G[u][v][Capacity] - H[u][v][Capacity])

            else:
                flow[u][v] = max(0, H[v].get(u, {}).get(Capacity, 0) -
                                 G[v].get(u, {}).get(Capacity, 0))
        else:
            flow[u][v] = G[u][v][Capacity]

    return flow


def ford_fulkerson(G, s, t, path_decomposition=False):

    flow_value, R = ford_fulkerson_impl(G, s, t, path_decomposition=path_decomposition)

    non_reachable = set(dict(nx.shortest_path_length(R, target=t)))
    partition = (set(G) - non_reachable, non_reachable)

    if path_decomposition:
        flow_dict = _create_flow_dict(G, R)
        for s_itr, t_dict in flow_dict.items():
            for t_itr, flow in t_dict.items():
                G[s_itr][t_itr][FFA_FlowName] = flow

    return flow_value, partition


# ============================================================

# find maximum bottleneck (widest) path
# modified Dijkstra's algorithm in networkx

#    Copyright (C) 2004-2018 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#
# Authors:  Aric Hagberg <hagberg@lanl.gov>
#           Loïc Séguin-C. <loicseguin@gmail.com>
#           Dan Schult <dschult@colgate.edu>
#           Niels van Adrichem <n.l.m.vanadrichem@tudelft.nl>


def widest_path(G, source, target):
    from networkx.algorithms.shortest_paths.weighted import _weight_function
    from heapq import heappush, heappop

    weight = _weight_function(G, Capacity)
    paths = {source: [source]}  # dictionary of paths

    G_succ = G._succ

    width_to = {}  # dictionary of final distances
    seen = {}

    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    seen[source] = float('inf')  # seen maximum width
    heappush(fringe, (-float('inf'), next(c), source))  # use min-heap as max-heap

    while fringe:
        (w, _, v) = heappop(fringe)
        if v in width_to:
            continue  # already searched this node.
        width_to[v] = -w
        if v == target:
            break
        for u, e in G_succ[v].items():
            width = weight(v, u, e)
            if not width:
                continue
            vu_width = min((width, width_to[v]))
            if u not in seen or vu_width > seen[u]:
                seen[u] = vu_width
                heappush(fringe, (-vu_width, next(c), u))
                paths[u] = paths[v] + [u]

    try:
        return width_to[target], paths[target]
    except KeyError:
        raise nx.NetworkXNoPath("No path to {}.".format(target))


# ============================================================
# David Ye in YCC Lab @ NTU
# DeWeiYe@ntu.edu.tw


# option -F
def flow_analysis(system, cluster=None, source=None, target=None):
    """
    FFA flow decomposition:

    The legendary Ford-Fulkerson algorithm (FFA), using the widest path as augment path

    :param system: reference system
    :param cluster: should be
                    1. tuple: rate matrix, cluster energies, (plot name)
                    2. ClusterMap object
    :param source: str, source of FFA
    :param target: str, target of FFA

    :return: flow matrix, flow graph with flow in attr: ['ffa']
    """
    node_ls = system.ExcitonName

    if source not in node_ls:
        print_normal('source {} not found, used default source {}'.format(source, node_ls[-1]))
        source = node_ls[-1]
    if target not in node_ls:
        print_normal('target {} not found, used default target {}'.format(target, node_ls[0]))
        target = node_ls[0]

    # find the source/target cluster
    from obj.cluster_map import ClusterMap
    if cluster is not None:
        if isinstance(cluster, ClusterMap):
            cluster = system.get_cluster_rate_and_energies(cluster)

        network = get_cluster_graph(cluster)
        if source not in network.nodes():
            for c in network.nodes():
                if source in c:
                    source = c
                    break

        if target not in network.nodes():
            for c in network.nodes():
                if target in c:
                    target = c
                    break
        order = list(cluster[0].keys())
    else:
        network = system.get_graph()
        order = system.ExcitonName

    # do FFA
    print('start calculating FFA flow,\nsource:', source, 'target:', target)
    max_flow, _ = ford_fulkerson(network, source, target, path_decomposition=True)
    print('flow: ', format(max_flow, '.2f'))

    print_normal('flow matrix:')
    flow_matrix = nx.linalg.attrmatrix.attr_matrix(network, edge_attr=FFA_FlowName, rc_order=order).T
    print_normal(flow_matrix)

    return flow_matrix, network
