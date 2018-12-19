from . import nx_pydot


def nx_graph_draw(ref_graph, system, plot_name='', label='weight', e_name='energy', rc_order=None):
    # prevent circular import
    from aux import pass_int, pass_float, nx, wraps, os
    from plot import colormap
    from alg import flow_kmeans, FFAName
    from matplotlib.colors import rgb2hex

    setting = system.back_ptr.setting

    file_name = plot_name + '_' + str(setting.Setting['cutoff']).replace('.', '')
    file_format = setting.Setting['format'].lower()
    dot_path = system.back_ptr.config.get_graphaviz_dot_path()
    dot_file = file_name + '.dot'
    image_file = file_name + '.' + file_format

    # wrap
    graph = ref_graph.copy()
    graph.graph['ranksep'] = .7
    graph.graph['dpi'] = pass_int(setting.Setting['dpi'])

    # node color
    if rc_order is not None:
        colors = colormap(len(rc_order), bright=True)
        colors.reverse()
        color_order = rc_order
    else:
        colors = colormap(len(ref_graph), bright=True)
        color_order, energies = zip(*sorted(graph.nodes(data=e_name)))

    for n, c in zip(color_order, colors):
        graph.nodes[n]['style'] = 'filled'
        graph.nodes[n]['color'] = rgb2hex(c)

    # wrap edges
    max_flow_ratio = FFAName == label
    decimal = pass_int(setting.Setting['decimal'])
    cutoff = pass_float(setting.Setting['cutoff'])

    labels = []
    for s, t, cap in ref_graph.edges(data=label):
        # flux/flow
        if not cap:
            graph.remove_edge(s, t)
            continue

        # provide a special rule for LHCII monomer
        if 'LHC8' in setting.KeyWords:
            LHC_sp_rule = (s == '8' and cap > 0.1 ** decimal / 2)
        else:
            LHC_sp_rule = False

        if cap < cutoff and not LHC_sp_rule:
            graph.remove_edge(s, t)
            continue

        if max_flow_ratio:
            # label = 0.12/0.38
            if ref_graph[s][t]['weight'] == cap:
                graph[s][t]['fontcolor'] = 'Red'
            graph[s][t]['label'] = '{}/{}'.format(format(cap, '.{}f'.format(decimal)),
                                                  format(ref_graph[s][t]['weight'], '.{}f'.format(decimal)))
        else:
            graph[s][t]['label'] = format(cap, '.{}f'.format(decimal))
        labels.append(cap)

    # mapping name
    mapping = {}
    for n, r in zip(ref_graph.nodes(), wraps(ref_graph.nodes())):
        mapping[n] = r
        graph.nodes[n]['fontname'] = 'Arial bold'
        if len(n.split()) > 2:
            if '\n' not in r:
                graph.nodes[n]['fontsize'] = 24
            else:
                graph.nodes[n]['fontsize'] = 20
        else:
            graph.nodes[n]['fontsize'] = 24
    graph = nx.relabel_nodes(graph, mapping, copy=False)
    rc_order = [mapping[n] for n in rc_order]

    # next step: width of flow:
    if labels:
        flow_dict = flow_kmeans(labels, lifetime=label == 'lifetime')

        for s, t, cap in graph.edges(data=label):
            dic = graph[s][t]
            dic['fontname'] = 'Arial bold'
            dic['fontsize'] = 17 if max_flow_ratio else 22
            dic['penwidth'], dic['color'] = flow_dict[cap]
            dic['weight'] = dic['penwidth']

            # ranking by rc_order/energies
            if 'norankdown' not in setting.KeyWords:
                if rc_order is not None:
                    change = rc_order.index(s) < rc_order.index(t)
                else:
                    change = graph.nodes[s][e_name] < graph.nodes[t][e_name]
                if change:
                    graph[s][t]['constraint'] = 'false'
                else:
                    graph[s][t]['constraint'] = 'true'

            # external label
            # the mechanism need to be optimized
            if 'xlabel' in setting.KeyWords:
                lab_dict = {}
                for s, t, lab in [(s, t, lab) for s, t, lab in graph.edges(data='label') if lab]:
                    graph[s][t]['taillabel'] = lab
                    if s in lab_dict:
                        graph[s][t]['labeldistance'] = 1 + lab_dict[s]
                        lab_dict[s] += 1.5
                    else:
                        graph[s][t]['labeldistance'] = 1
                        lab_dict[s] = 1.5
                    del graph[s][t]['label']

    # use pydot and system cmd instead of pygraphviz (which is out-of-date)
    with open(dot_file, 'w') as f:
        nx_pydot.write_dot(graph, f)

    os.system(dot_path + " -T" + file_format + " " + dot_file + " -o " + image_file)

    # save to:
    print('plot graph:', image_file)
    print('write dot file:', dot_file)
