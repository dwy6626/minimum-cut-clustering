import alg
from lib import *
import plot

NormTemplate = '"true" or "false" or "both"'


# ============================================================


class MinCutTree:
    def __init__(self, system, ref_tree=None):

        # global source and target
        self.tree_source = system.back_ptr.setting.get('s', 'not a node')
        self.tree_target = system.back_ptr.setting.get('t', 'not a node')

        self.back_ptr = system

        if isinstance(ref_tree, MinCutTree):
            self.norm = ref_tree.norm
            self.root = ref_tree.root.copy()
            return

        self.root = TreeNode(set(system.ExcitonName))

        # 0: normalized
        # 1: not normalized
        self.norm = (False, False)
        while True:
            self.set_normalized(system.back_ptr.setting['norm'])
            if any(self.norm):
                break
            system.back_ptr.setting['norm'] = input(
                'if the maximum flow need to be normalized?\n'
                'please enter one of the following:\n'
                '{}\n'.format(NormTemplate)
            )

        self.build_init()
        print_1_line_stars()

    def build_init(self):
        print('Start generating min-cut tree using Ford-Fulkerson algorithms.')
        print('Augment path: maximum bottleneck path')
        self.__build(self.back_ptr.get_graph(), self.root)

    def __build(self, graph, root):
        """
        :param graph: reference only, don't modify this graph
        """
        if len(root.val) == 1:
            return

        source = self.get_source_or_target(graph, root.val, target=False)
        target = self.get_source_or_target(graph, root.val, target=True)

        print('source: {}, target: {}'.format(source, target))

        # main part: do the minimum cut by FFA!
        root.max_flow, cuts = alg.ford_fulkerson(graph.subgraph(root.val), source, target)

        # add results into tree
        root.norm_flow = root.max_flow / len(cuts[0]) / len(cuts[1])
        root.left = TreeNode(cuts[0])
        root.right = TreeNode(cuts[1])
        root.left.p = root.right.p = root

        # To know where is the process
        print('source subgraph: {}\ntarget subgraph: {}\n'.format(*cuts))

        # keep doing FFA until all nodes are isolated:
        self.__build(graph, root.left)
        self.__build(graph, root.right)

    def get_source_or_target(self, graph, subset, target=False):
        energies = {n: e for n, e in graph.nodes(data='energy') if n in subset}

        if target:
            r = self.tree_target
            _func = min
        else:
            r = self.tree_source
            _func = max
        if r not in subset:
            # disordered networks:
            # use the state name to determinate the source:
            if self.back_ptr.get_index() != 0 or 'labelorder' in self.back_ptr.back_ptr.setting:
                try:
                    return str(_func(map(int, subset)))
                except:
                    pass

        # otherwise: the exciton state with highest/lowest energy
        return _func(energies, key=lambda x: energies[x])

    def set_normalized(self, norm):
        s = norm in ('true', 'both'), norm in ('both', 'false')
        if not any(s):
            Warning(NormTemplate)
        else:
            self.norm = s
            if self.norm[1]:
                print('Normalize min-cut tree')

    def copy(self):
        return MinCutTree(self.back_ptr, ref_tree=self)

    def ascending_cut_init(self, norm):
        tree = self.copy()
        s = [tree.root]
        while s:
            cur = s.pop()
            if cur.is_leaf():
                continue

            if cur.left.is_leaf():
                cur.sflow = float("inf")
            else:
                cur.sflow = cur.left.norm_flow if norm else cur.left.max_flow
            if cur.right.is_leaf():
                cur.tflow = float("inf")
            else:
                cur.tflow = cur.right.norm_flow if norm else cur.right.max_flow

            s.extend([cur.left, cur.right])
        return tree

    def run(self, prefix):
        """
        :return: 1. suffix for job name
                 2. normalized?
        """
        for i, b in enumerate(self.norm):
            if b:
                print(
                    ['maximum flow normalized',
                     'maximum flow not normalized (_uN)'][i]
                )
                yield prefix + '_uN' if i == 1 else prefix, i == 0

    def draw(self):
        file_name = self.back_ptr.get_output_name('Tree')
        file_format = self.back_ptr.back_ptr.setting['format'].lower()
        dot_path = self.back_ptr.back_ptr.config.get_graphaviz_dot_path()

        for run_name, norm in self.run(file_name):
            dot_file = run_name + '.dot'
            image_file = run_name + '.' + file_format
            print('Plot the min-cut binary tree into ' + image_file)
            self.__to_dot(dot_file, norm)
            os.system(dot_path + " -T" + file_format + " " + dot_file + " -o " + image_file)
            print('write dot file:', dot_file)

    def collect_flow(self, norm=True, st=False, p=False, ratio=False, sort=False):
        # do not assert st and p simultaneously
        r = []
        s = [self.root]
        while s:
            cur = s.pop()
            if not cur.is_leaf():
                s.extend([cur.left, cur.right])
                flow = cur.norm_flow if norm else cur.max_flow

                # for ratio cut
                # child flow / parent flow
                if ratio:
                    if cur.p:
                        p_flow = cur.p.norm_flow if norm else cur.p.max_flow
                        flow = flow / p_flow
                    else:
                        flow = 0

                if st:
                    flow = copy(cur.left.val), copy(cur.right.val), flow
                elif p:
                    flow = cur, flow
                r.append(flow)
        if sort:
            return sorted(r, key=itemgetter(2*st + p))
        return r

    def __to_dot(self, file_name, norm):
        setting = self.back_ptr.back_ptr.setting
        color_dict = plot.node_color_energy(self.back_ptr.ExcitonName,
                                            self.back_ptr.get_original().ExcitonEnergies)
        flow_dict = alg.flow_kmeans(self.collect_flow(norm))

        dpi = pass_int(setting['dpi'])

        with open(file_name, 'w') as f:
            f.write('strict digraph  {{\nranksep=0.1;\ndpi={};\n'.format(dpi))
            self.root.write_data(f, color_dict, flow_dict,
                                 setting['decimal'], norm
                                 )
            f.write('}')


# ============================================================


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.p = None
        self.left = None
        self.right = None
        self.max_flow = None
        self.norm_flow = None
        self.__wrapped = None

        # for ascending cut
        self.sflow = None
        self.tflow = None

    def copy(self):
        r = TreeNode(self.val)
        r.max_flow, r.norm_flow = self.max_flow, self.norm_flow
        if not self.is_leaf():
            r.left = self.left.copy()
            r.right = self.right.copy()
            r.left.p = r.right.p = r
        return r

    def is_leaf(self):
        return not any([self.left, self.right])

    def is_root(self):
        return not self.p

    def get_wrapped_name(self):
        if not self.__wrapped:
            # use All as abbreviation when all node are included
            if self.is_root():
                self.__wrapped = 'All'
            else:
                self.__wrapped = wrap_str(self.val, width=12)
        return self.__wrapped

    def write_data(self, f, color_dict, flow_dict, decimal=2, norm=True):
        if self.is_leaf():
            color = color_dict[next((x for x in self.val))]
            font_size = 24
        elif self.is_root():
            color = '#969696'
            font_size = 18
        else:
            color = '#d1d1d1'
            if '\n' in self.get_wrapped_name():
                font_size = 13
            else:
                font_size = 16

        # node
        f.write('"{}" [style=filled, fontname="Arial bold", fontsize={}, color="{}"];\n'.format(
            self.get_wrapped_name(),
            font_size,
            color
        ))

        link = '"{}" -> "{}" [dir=none, style=dashed];\n'
        if not self.is_leaf():
            # link
            f.write(link.format(self.get_wrapped_name(), self.left.get_wrapped_name()))
            f.write(link.format(self.get_wrapped_name(), self.right.get_wrapped_name()))

            # flow
            flow = self.norm_flow if norm else self.max_flow
            f.write(
                '"{{}}" -> "{{}}"'
                ' [fontname="Arial bold", fontsize=17, '
                'penwidth="{{}}", color="{{}}", label="{{:.{}f}}", constraint=False];\n'
                ''.format(decimal).format(
                    self.left.get_wrapped_name(),
                    self.right.get_wrapped_name(),
                    *flow_dict[flow], flow
            ))

            # recursively write
            self.left.write_data(f, color_dict, flow_dict, decimal, norm)
            self.right.write_data(f, color_dict, flow_dict, decimal, norm)
