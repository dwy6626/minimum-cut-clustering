from aux import *


# ============================================================


def input_map_clustering(system):
    # format: 1,2|3,4|5,6|7
    cluster_map = system.get_new_map('inp')
    map_ls = system.back_ptr.setting.Setting['map'].split('|')

    clx_ls = []
    for clx in map_ls:
        s = set()
        for n in set(clx.split(',')):
            if "-" in n:
                a, b = n.split("-")
                s.update([str(i) for i in range(int(a), int(b)+1)])
            else:
                s.add(n)
        clx_ls.append(s)

    # self check
    for a, b in combinations(clx_ls, 2):
        if a & b:
            print('wrong cluster map')
            raise SyntaxError
    else:
        for clx in clx_ls:
            cluster_map.group_up(clx)

    cluster_map.save()


def k_clustering(system):
    """
    k-clustering with Kruskal's algorithm
    """
    cluster_map = system.get_new_map('k-clustering', one_group=False)
    k = 3

    # to undirected graph by energies:
    edge_list = system.to_undirected(4)
    for u, v, cap in edge_list:
        if cluster_map[u] != cluster_map[v]:
            cluster_map.merge(u, v)
            k += 1
            cluster_map.save()
            if k > len(system):
                break


def cut_off_method(system, option, pass_map=False):
    return_object = {}

    edge_list = system.to_undirected(option)
    edge_list.reverse()  # cutoff from small to large
    if option == 5:
        cluster_map = system.get_new_map('DCcp', site_map=True)
    else:
        cluster_map = system.get_new_map(['DCmax', 'DCgeo', 'DCrms', 'DC'][option-1])

    # record change in cgm and output
    subgroups = [{}]

    # the max cut-off cut all nodes => no meaning
    # and the min cut-off merge all nodes into 1 cluster
    # edge_list[2:-1]
    for i in range(2, len(edge_list)-1):
        *_, cutoff = edge_list[i]

        if cutoff == edge_list[i+1][-1]:
            continue
        cluster_map.update_cutoff(cutoff)

        network = nx.Graph()
        network.add_nodes_from(cluster_map.keys())
        network.add_weighted_edges_from(edge_list[i+1:])
        new_subgroups = tuple(nx.connected_components(network))
        del network

        # if new_subgroup > subgroup than cut the new one
        if len(new_subgroups) > len(subgroups):
            for s in new_subgroups:
                if s not in subgroups:
                    cluster_map.group_up(s)
            subgroups = new_subgroups

            # if change: plot and save for k_means
            if option == 5:
                cluster_map.print_all()
            elif pass_map:
                return_object[len(cluster_map)] = cluster_map.copy()
            else:
                cluster_map.save()

    if pass_map:
        return return_object
