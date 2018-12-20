from lib import *


# ============================================================


def simple_cut(system, option=0):
    """
    option:
        0: SC simple cut
        1: SR simple ratio cut
    """
    tree = system.get_tree()
    for suffix, norm in tree.run(['SC', 'SR'][option]):
        cluster_map = system.get_new_map(suffix, one_group=False)

        # from max to min: merge the nodes
        # -1 for note that no meaning for plot 1 cluster model
        for s, t, f in tree.collect_flow(norm=norm, st=True, sort=True, ratio=option)[:0:-1]:
            # Merge the clusters with maximum flow
            cluster_map.update_cutoff(f)

            to_merge = list(s | t)
            if set(to_merge) > set(cluster_map[to_merge[0]]):
                cluster_map.group_up(to_merge)
                if len(cluster_map) == 1:
                    return
                cluster_map.save()


# find all flow, begin from min to max, look the 2 subgroups of the flow
# compare the max intra-flow inside the subgroup
# choose the subgroup with larger inter-flow => cut it as a new cluster
# the other subgroup (remain) stay in the original group
def ascending_cut(system, option=1):
    """
    :param option:
            0: cut source cluster if equal
            1: cut target cluster if equal
    """
    tree_ref = system.get_tree()

    for suffix, norm in tree_ref.run(['TDC2', 'TDC'][option]):
        # a big group contain all node
        # cluster_map will be cut and tell network to merge nodes into CGM
        cluster_map = system.get_new_map(suffix, one_group=True)

        # a tree to be modified into forest of subtrees
        tree = tree_ref.ascending_cut_init(norm)

        # from N nodes to N-2 clusters
        # => no 1-cluster CGM and N-cluster CGM
        for p, f in tree.collect_flow(norm=norm, p=True, sort=True)[:-1]:
            # cut the cluster with larger intra-flow
            if p.sflow > p.tflow:
                cut = p.left
            elif p.sflow < p.tflow:
                cut = p.right
            else:
                cut = p.right if option else p.left

            # adjust intra-flow
            if cut is p.right:
                if p.is_root():
                    pass
                elif p.p.left is p:
                    p.p.sflow = p.sflow
                else:
                    p.p.tflow = p.sflow
            else:
                if p.is_root():
                    pass
                elif p.p.left is p:
                    p.p.sflow = p.tflow
                else:
                    p.p.tflow = p.tflow

            cluster_map.group_cut(cut.val)
            cluster_map.update_cutoff(f)
            cluster_map.save()


def bottom_up_clx(system):
    tree_ref = system.get_tree()
    for suffix, norm in tree_ref.run('BUC'):
        cluster_map = system.get_new_map(suffix, one_group=False)
        tree = tree_ref.copy()

        while True:
            # break if only 1 flow remain
            if tree.root.left.is_leaf() and tree.root.right.is_leaf():
                break

            s = [tree.root]
            leaves = []
            while s:
                cur = s.pop()
                if cur.is_leaf():
                    continue
                elif cur.left.is_leaf() and cur.right.is_leaf():
                    leaves.append((cur, cur.norm_flow if norm else cur.max_flow))
                else:
                    s.extend([cur.left, cur.right])

            cut, flow = max(leaves, key=itemgetter(1))  # min?
            cluster_map.update_cutoff(flow)
            cluster_map.group_up(cut.val)

            # remove cut group in tree
            cut.left = cut.right = None
            cluster_map.save()
