# modified by David Ye for min-cut clustering output

"""
*****
Pydot
*****

Import and export NetworkX graphs in Graphviz dot format using pydot.

Either this module or nx_agraph can be used to interface with graphviz.

See Also
--------
pydot:         https://github.com/erocarrera/pydot
Graphviz:      http://www.research.att.com/sw/tools/graphviz/
DOT Language:  http://www.graphviz.org/doc/info/lang.html
"""
# Author: Aric Hagberg (aric.hagberg@gmail.com)

#    Copyright (C) 2004-2017 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    Cecil Curry <leycec@gmail.com>
#    All rights reserved.
#    BSD license.

from networkx.utils import open_file, make_str
import networkx as nx


@open_file(1, mode='w')
def write_dot(G, path):
    """Write NetworkX graph G to Graphviz dot format on path.

    Path can be a string or a file handle.
    """
    P = to_pydot(G)
    path.write(P.to_string())
    return


def to_pydot(N):
    """Return a pydot graph from a NetworkX graph N.

    Parameters
    ----------
    N : NetworkX graph
      A graph created with NetworkX

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> P = nx.nx_aux.to_pydot(K5)

    Notes
    -----

    """
    pydot = _import_pydot()

    # set Graphviz graph type
    if N.is_directed():
        graph_type = 'digraph'
    else:
        graph_type = 'graph'
    strict = nx.number_of_selfloops(N) == 0 and not N.is_multigraph()

    name = N.name
    graph_defaults = N.graph
    # graph_defaults = N.graph.get('graph', {})
    if name is '':
        P = pydot.Dot('', graph_type=graph_type, strict=strict,
                      **graph_defaults)
    else:
        P = pydot.Dot('"%s"' % name, graph_type=graph_type, strict=strict,
                      **graph_defaults)
    try:
        P.set_node_defaults(**N.graph['node'])
    except KeyError:
        pass
    try:
        P.set_edge_defaults(**N.graph['edge'])
    except KeyError:
        pass

    for n, nodedata in N.nodes(data=True):
        str_nodedata = dict((k, make_str(v)) for k, v in nodedata.items())
        p = pydot.Node(make_str(n), **str_nodedata)
        P.add_node(p)

    if N.is_multigraph():
        for u, v, key, edgedata in N.edges(data=True, keys=True):
            str_edgedata = dict((k, make_str(v)) for k, v in edgedata.items() if k != 'key')
            edge = pydot.Edge(make_str(u), make_str(v),
                              key=make_str(key), **str_edgedata)
            P.add_edge(edge)

    else:
        for u, v, edgedata in N.edges(data=True):
            str_edgedata = dict((k, make_str(v)) for k, v in edgedata.items())
            edge = pydot.Edge(make_str(u), make_str(v), **str_edgedata)
            P.add_edge(edge)
    return P


def _import_pydot():
    # Minimum required version of pydot, which broke backwards API compatibility in
    # non-trivial ways and is thus a hard NetworkX requirement. Note that, although
    # pydot 1.2.0 was the first to do so, pydot 1.2.3 resolves a critical long-
    # standing Python 2.x issue required for sane NetworkX operation. See also:
    #     https://github.com/erocarrera/pydot/blob/master/ChangeLog
    import pydot
    return pydot
