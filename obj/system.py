# import local modules
from .map import ClusterMap
from .tree import MinCutTree
from lib import *
import alg
import plot


# ============================================================


class System:
    def __init__(self, reference, back_ptr, index=0, is_rate_matrix=False, additional_hamiltonian_string=''):
        self.__NumberSite = 0

        # should align with Hamiltonian
        self.SiteName = []
        self.Hamiltonian = None
        self.EigenVectors = None

        # index for disorders
        # default: 0
        self.__index = index

        # ensure the excitons are sorted by energies
        self.ExcitonName = []
        self.ExcitonEnergies = []

        # align with exciton name:
        self.RateConstantMatrix = None

        # a back pointer for easier operations:
        self.back_ptr = back_ptr

        # minimum-cut tree
        self.__Tree = None

        # full (all-states) population dynamics
        # 3-tuple: dynamics(t), time, grid
        self.__pop_tuple = None

        # if the population difference is calculated?
        # (to a selected size: 15 by default)
        self.__cost = False

        # copy and add disorders
        if isinstance(reference, System):
            self.SiteName = reference.SiteName
            self.__NumberSite = len(reference)
            self.generate_disordered_hamiltonian(reference)
            return

        elif isinstance(reference, str):
            if is_rate_matrix:
                self.load_key(reference)
                # load H of the same system:
                self.load_h(additional_hamiltonian_string)
            else:
                self.load_h(reference)
                self.ExcitonName = list(map(str, range(1, len(self) + 1)))
                self.RateConstantMatrix = pd.DataFrame(
                    alg.modified_Redfield_theory_calculation(self.Hamiltonian, *self.back_ptr.setting.get_mrt_params()),
                    columns=self.ExcitonName, index=self.ExcitonName
                )
        else:
            raise KeyError('should receive string or a reference system')

    def get_original(self):
        return self.back_ptr.get_reference_system()

    def generate_disordered_hamiltonian(self, reference):
        # standard deviation
        sd = pass_float(self.back_ptr.setting.get('sd', 100))
        while True:
            if len(self.back_ptr.disorders) > self.__index:
                disorders = self.back_ptr.disorders[self.__index]
            else:
                disorders = np.array([np.random.normal(scale=sd) for _ in range(len(self))])

            hamiltonian = copy(reference.Hamiltonian)
            for j in range(len(self)):
                hamiltonian[j, j] = hamiltonian[j, j] + disorders[j]

            print('static disorder (cm-1):')
            print(disorders)

            # find corresponding node names: maximum overlap
            w, eigv = np.linalg.eigh(hamiltonian)
            w0, v0 = np.linalg.eigh(reference.Hamiltonian)
            self.back_ptr.print_log('eigenvector 0\n', v0.T, '\n')
            self.back_ptr.print_log('eigenvector\n', eigv.T, '\n')

            overlaps = v0.T.dot(eigv) ** 2

            # maximum in original basis
            from_v0_max = np.argmax(overlaps, axis=0)

            # make the duplicate labels to -2
            dup = find_duplicate(from_v0_max)
            if dup:
                self.back_ptr.print_log('duplicate terms:', dup)
                for k, v in dup.items():
                    self.back_ptr.print_log(overlaps[k, v])
                    from_v0_max[v[np.argmin(overlaps[k, v])]] = -2

            # maximum in new basis
            arg_v_max = np.argmax(overlaps, axis=1)

            # make the duplicate labels to -2
            dup = find_duplicate(arg_v_max)
            if dup:
                self.back_ptr.print_log(arg_v_max)
                self.back_ptr.print_log('duplicate terms:', dup)
                for k, v in dup.items():
                    self.back_ptr.print_log(overlaps[v, k])
                    arg_v_max[v[np.argmin(overlaps[v, k])]] = -2

            from_v_max = np.ones(len(w0), dtype=int) * -2
            for i, n in enumerate(arg_v_max.flat):
                if n >= 0:
                    from_v_max[n] = i
            self.back_ptr.print_log(from_v0_max + 1)
            self.back_ptr.print_log(from_v_max + 1)

            node_name = np.where(from_v_max >= from_v0_max, from_v_max + 1, from_v0_max + 1)
            self.back_ptr.print_log(node_name, '\n')

            # unnamed nodes: -1 (-2 + 1)
            if len(node_name[node_name == -1]) == 1:
                # name the unnamed node as the unused label
                node_name[node_name == -1] = next((i for i in range(1, len(w0) + 1) if i not in node_name))

            # name the unnamed nodes with larger overlap ones
            if len(node_name[node_name == -1]) == 2:
                v_remain = np.where(node_name == -1)[0]
                v0_remain = np.array([i for i in range(1, len(w0) + 1) if i not in node_name]) - 1
                print(v_remain, v0_remain)
                overlap_remain = overlaps[v0_remain, :][:, v_remain]

                arg_v0_max = np.argmax(overlap_remain, axis=0)
                arg_v_max = np.argmax(overlap_remain, axis=1)

                from_v0_max = list(v0_remain[arg_v0_max])
                from_v_max = list(v_remain[arg_v_max])
                self.back_ptr.print_log(overlaps)
                self.back_ptr.print_log(overlap_remain)
                self.back_ptr.print_log(from_v0_max, from_v_max)

                if not has_duplicate(from_v0_max):
                    for i, n in enumerate(node_name.flat):
                        if n == -1:
                            node_name[i] = from_v0_max.pop(0) + 1

            # corresponding overlap array: overlap between new basis and old basis
            corres_overlap_array = np.diag(overlaps[node_name - 1])
            node_name = [str(i) for i in node_name]
            self.back_ptr.print_log('New node names:', node_name)

            # overlap factor: the average overlap
            overlap_factor = sum(corres_overlap_array) / len(corres_overlap_array) * 100
            if 'log' in self.back_ptr.setting:
                print('corresponding overlap with original states:', corres_overlap_array)
                print('the total overlap factors: {:.2f}%'.format(overlap_factor))

            # pass the diagnosis: yield the H
            # otherwise, regenerate one
            if has_duplicate(node_name):
                self.back_ptr.discard_disorders.append(disorders.reshape(1, -1))
                print('  ... Hamiltonian discarded.')
            else:
                # sort by label
                indexing = [node_name.index(n) for n in reference.ExcitonName]
                self.ExcitonEnergies = w[indexing,]
                self.ExcitonName = reference.ExcitonName
                self.EigenVectors = eigv[:, indexing]
                self.back_ptr.disorders.append(disorders.reshape(1, -1))
                self.back_ptr.overlap_factors[self.__index] = overlap_factor
                self.back_ptr.overlap_arrays[self.__index] = corres_overlap_array

                self.Hamiltonian = hamiltonian
                self.RateConstantMatrix = pd.DataFrame(
                    alg.modified_Redfield_theory_calculation(
                        self.Hamiltonian, *self.back_ptr.setting.get_mrt_params()
                    )[indexing, ][:, indexing],
                    columns=self.ExcitonName, index=self.ExcitonName
                )
                print('Rate constants, by labels')
                print(self.RateConstantMatrix)
                print_1_line_stars()
                break

        return

    def load_h(self, argv):
        if not argv:
            return

        lines = string_to_lines(argv)

        extend_with_identity = False
        if len(self):
            if len(lines[0]) != len(self):
                print("size of Hamiltonian and rate matrix doesn't match: {}, {}".format(len(lines[0]), len(self)))
                if len(lines[0]) < len(self):
                    while 1:
                        s = input("add extra state in Hamiltonian? (Y/N)")
                        if "y" == s.lower():
                            extend_with_identity = True
                            break
                        elif "n" == s.lower():
                            break
                        print("please answer Y or N")
                if not extend_with_identity:
                    raise ValueError('size error occurs in Hamiltonian')
        else:
            self.__NumberSite = len(lines[0])

        # if number of lines = number of line elements + 1:
        # 1st line is site name
        if len(lines) == len(lines[0]) + 1:
            self.SiteName = lines.pop(0)
        elif len(lines) == len(lines[0]):
            # use site numbers:
            self.SiteName = ['Site ' + str(i + 1) for i in range(len(lines[0]))]
        else:
            # unexpect format:
            print()
            print('expected format:')
            print('    First line: Site Names (optional)')
            print('    Effective Hamiltonian (Square Matrix)')
            raise ValueError('unexpected input hamiltonian format')

        # extra sites
        delta = len(self) - len(lines[0])
        for i in range(delta):
            self.SiteName += ["extra {}".format(i)]
        print('Sites:\n', self.SiteName)

        h = np.array(lines, dtype=float)
        if extend_with_identity:
            e_max = np.amax(np.linalg.eigh(h)[0])
            h = np.append(h.reshape(len(lines[0]), len(lines[0])), np.zeros((delta, len(lines[0]))), axis=0)
            h = np.append(h, np.zeros((len(lines[0]) + delta, delta)), axis=1)
            for i in range(len(lines[0]), len(self)):
                h[i, i] = e_max + 100 + i
        else:
            h.reshape(len(self), -1)

        self.ExcitonEnergies, self.EigenVectors = np.linalg.eigh(h)
        self.Hamiltonian = h
        print('Hamiltonian:')
        print(self.Hamiltonian)

        print('Basis transition matrix:')
        print(self.EigenVectors)
        self.back_ptr.print_log('U square:')
        self.back_ptr.print_log(self.EigenVectors ** 2)

    def load_key(self, argv):
        def in_format_error():
            print()
            print('expected format:')
            print('    First line: State Names (optional)')
            print('    Second line: State Energies')
            print('    Rate Constant Matrix (Square Matrix)')
            raise ValueError('unexpected input file format')

        lines = string_to_lines(argv)
        tmp = []
        for l in lines:
            tmp.extend(l)
        lines = tmp

        # check the elements in the input file:
        # n(n+2) = number of elements
        # if possible n exist: ok
        size = (len(lines) + 1) ** 0.5 - 1
        if int(size) == size:
            self.__NumberSite = int(size)
            # name of exciton states
            self.ExcitonName = lines[:len(self)]
            # remove the first line
            lines = lines[len(self):]

        else:
            # also, state name can be generated automatically:
            # this time, n(n+1) = number of elements
            # I AM sure that n(n+1) = k(k+2) have no solution that both n, k are int
            # proved by contradiction
            # the proof is not shown because of space issue
            size = (4 * len(lines) + 1) ** 0.5 / 2 - 0.5
            if int(size) == size:
                self.__NumberSite = int(size)

                # name of states are generated automatically
                self.ExcitonName = list(map(str, range(1, len(self) + 1)))
            else:
                in_format_error()

        print('Size = {}'.format(len(self)))
        self.ExcitonEnergies = list(map(float, lines[:len(self)]))
        self.RateConstantMatrix = pd.DataFrame(np.array(lines[len(self):], dtype=float).reshape(len(self), -1),
                                               columns=self.ExcitonName, index=self.ExcitonName)
        self.RateConstantMatrix.values[self.RateConstantMatrix.values <= 0] = 0  # '=' to modify -0
        self.RateConstantMatrix[:] -= np.diag(np.sum(self.RateConstantMatrix.values, axis=0))

        # sorted the excitons
        if 'labelorder' in self.back_ptr.setting:
            for n in self.ExcitonName.copy():
                if not n.isdigit():
                    sort_list = zip(*sorted(zip(self.ExcitonName, self.ExcitonEnergies)))
                    break
            else:
                sort_list = zip(*sorted(zip(self.ExcitonName, self.ExcitonEnergies), key=lambda x: int(x[0])))
            self.ExcitonName, self.ExcitonEnergies = sort_list
        else:
            self.ExcitonEnergies, self.ExcitonName = zip(*sorted(zip(self.ExcitonEnergies, self.ExcitonName)))
        self.RateConstantMatrix = self.RateConstantMatrix.loc[self.ExcitonName, self.ExcitonName]

        print("Exciton States: \n    {}".format(self.ExcitonName))
        print('State Energies: \n    {}'.format(self.ExcitonEnergies))

        print('Rate constants: ')
        print(self.RateConstantMatrix)

    def __len__(self):
        return self.__NumberSite

    def get_output_name(self, str1=''):
        if str1:
            str1 = '_' + str1
        return self.back_ptr.get_output_name('{}'.format(H_suffix(self.get_index()))) + str1

    def has_hamiltonian(self):
        return self.Hamiltonian is not None

    def get_new_map(self, job_name, one_group=False, site_map=False):
        return ClusterMap(self, job_name, one_group, site_map)

    def get_index(self):
        return self.__index

    def get_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(
            [(n, {'energy': e}) for n, e in zip(self.ExcitonName, self.ExcitonEnergies)]
        )
        graph.add_weighted_edges_from(
            ((m, n, self.RateConstantMatrix[m][n]) for m, n in permutations(self.ExcitonName, 2))
        )
        return graph

    def get_tree(self, copy=False):
        if self.__Tree is None:
            self.__Tree = MinCutTree(self)
        if copy:
            return self.__Tree.copy()
        return self.__Tree

    def get_plot_name(self, cgm=None):
        if cgm is None:
            n_c = 'Full'
        else:
            n_c = '{}_{}c'.format(cgm.method, len(cgm))
        return self.back_ptr.get_output_name('{}_{}_'.format(H_suffix(self.get_index()), n_c))

    def get_cluster(self, clx_map):
        """
        build the coarse-grained model
        :param clx_map: cluster_map
        :return: rate matrix, energies (2-tuple)
        """
        rate = self.RateConstantMatrix.values
        energies = np.array(self.ExcitonEnergies)

        if self.get_index() == 0:
            ref_energies = energies
        else:
            ref_energies = np.array(self.get_original().ExcitonEnergies)

        groups = clx_map.groups()

        # sort clusters by minimum energy member:
        min_energies = [min((ref_energies[self.ExcitonName.index(n)] for n in cluster)) for cluster in groups]
        groups = [s for _, s in sorted(zip(min_energies, groups))]

        cluster_names = tuple(map(lambda x: ' '.join(sorted(x)), groups))
        cluster_energies = np.zeros(len(clx_map))
        weighted_rates = np.zeros(len(clx_map))

        temperature = self.back_ptr.setting.get_temperature()
        boltz_weights = get_boltz_factor(np.array(self.ExcitonEnergies), temperature)
        weighted_energies = boltz_weights * energies
        indexing = {n: self.ExcitonName.index(n) for n in self.ExcitonName}

        indices = []
        energy_min = []
        partition_functions = []

        for i in range(len(groups)):
            group_nodes = np.array(groups[i], dtype=str)

            indices.append(np.vectorize(indexing.get)(group_nodes))
            energy_min.append(np.min(energies[indices[-1], ]))
            partition_functions.append(np.sum(boltz_weights[indices[-1], ]))

            cluster_energies[i] = np.sum(weighted_energies[indices[-1], ]) / partition_functions[-1]
            weighted_rates[i] = np.sum(boltz_weights[indices[-1], ] * rate[:, indices[-1]])

        # build rate matrix:
        # source: boltzmann weighted sum
        rate = np.concatenate(
            [np.sum(rate[:, i] * boltz_weights[i], axis=1).reshape(-1, 1)/z
             for i, z in zip(indices, partition_functions)], axis=1
        )
        # target: simple sum
        rate = np.concatenate([[np.sum(rate[i, :], axis=0)] for i in indices], axis=0)

        print('rate constant matrix:')
        cluster_rate = pd.DataFrame(rate, columns=cluster_names, index=cluster_names)
        print(cluster_rate)

        print('cluster energies:')
        print(cluster_energies)

        return cluster_rate, cluster_energies

    def to_undirected(self, option):
        """
        :param option:
            1: larger rate
            2: geometric mean
            3. root mean square
            4. by energy
            5. electronic couplings in Hamiltonian
        :return rate constant matrix (symmetric)
        """
        if option == 5:
            return sorted([(self.SiteName[i], self.SiteName[j], abs(self.Hamiltonian.matrix[i, j]))
                           for i, j in combinations(range(len(self)), 2)], reverse=True, key=itemgetter(2))

        r = self.RateConstantMatrix.values
        upper = np.triu(r)
        lower = np.tril(r).T

        if option == 1:
            undirected_r = np.where(upper > lower, upper, lower)
        elif option == 2:
            undirected_r = (upper * lower) ** 0.5
        elif option == 3:
            undirected_r = (upper ** 2 + lower ** 2) ** 0.5
        else:  # 4
            grid = np.meshgrid(self.ExcitonEnergies, self.ExcitonEnergies)
            undirected_r = np.where(grid[0] > grid[1], upper, lower)

        r = deepcopy(self.RateConstantMatrix)
        r[:] = undirected_r + undirected_r.T
        return sorted([(u, v, r[u][v]) for u, v in combinations(r.keys(), 2)], reverse=True, key=itemgetter(2))

    def __cluster_handler(self, cluster):
        """
        :param cluster: should be
                        1. tuple: rate matrix, cluster energies, (job name)
                        2. cluster map object
        :return: rate matrix, energies, job_name
        """
        if cluster is None:
            rate = self.RateConstantMatrix
            cluster_energies = None
            plot_name = self.get_output_name('Full_')
        else:
            if isinstance(cluster, ClusterMap):
                rate, cluster_energies = self.get_cluster(cluster)
                plot_name = self.get_plot_name(cluster)
            elif len(cluster) == 3:
                rate, cluster_energies, plot_name = cluster
            elif len(cluster) == 2:
                rate, cluster_energies = cluster
                plot_name = ''
            else:
                print('param: cluster should be \n'
                      '       1. tuple: rate matrix, cluster energies, (job name)\n'
                      '       2. cluster map object')
                raise KeyError
        return rate, cluster_energies, plot_name

    def plot_dynamics(
            self, cluster=None, save_to_file=False, max_name_len=30
    ):
        _, cluster_energies, plot_name = self.__cluster_handler(cluster)
        is_clustered = cluster_energies is not None

        # get more plot settings
        setting = self.back_ptr.setting
        y_max = pass_float(setting.get('ymax', '0.'))
        x_max = pass_float(setting.get('xmax', '0.'))
        divide = pass_int(setting.get('divide', 100))
        pop_seq2 = None

        if is_clustered:
            pop_seq, time_sequence, clusters = self.__cal_dynamics(cluster=cluster)
            pop_seq2 = self.get_comparison_to_full_dynamics(clusters)

            # for label is too long: cluster X
            pop_names = wraps(clusters, maxlen=30)

            plot.plot_dyanmics(
                pop_seq, time_sequence, pop_names, plot_name,
                pop_seq2=self.get_comparison_to_full_dynamics(clusters),
                y_max=y_max, x_max=x_max, divide=divide,
                legend='nolegend' not in setting,
                save_to_file=save_to_file
            )
        else:
            if self.__pop_tuple is None:
                self.__cal_dynamics()

            pop_names = self.ExcitonName
            pop_seq, propagate_time, time_grid = self.__pop_tuple
            time_sequence = np.linspace(0, propagate_time, time_grid)

        plot.plot_dyanmics(
            pop_seq, time_sequence, pop_names, plot_name,
            pop_seq2=pop_seq2,
            y_max=y_max, x_max=x_max, divide=divide,
            legend='nolegend' not in setting,
            save_to_file=save_to_file
        )

        # basis transform for original network
        # if H_eff is provided
        if self.has_hamiltonian() and 'site' in setting and not is_clustered:
            print('--site: change to site basis')
            pop_seq_site = np.dot(self.EigenVectors ** 2, pop_seq)
            plot.plot_dyanmics(
                pop_seq_site, time_sequence, self.SiteName, plot_name + 'Site_',
                y_max=y_max, x_max=x_max, divide=divide,
                legend='nolegend' not in setting,
                save_to_file=save_to_file
            )

    def get_initial_populations(self, nodes, energies, clusters=None, cluster_energies=None):
        if clusters is not None and cluster_energies is not None:
            size = len(clusters)
            is_clustered = True
        else:
            is_clustered = False
            size = len(self)

        setting = self.back_ptr.setting

        # equal partitions
        init_option = setting['init']
        if init_option.lower() in ['equally', 'eq', 'equipartition', 'same']:
            pop = np.ones(len(self)) / len(self)
        # boltzmann partitions:
        elif init_option.lower() in ['boltz', 'boltzmann', 'thermal']:
            # calculate the exp
            temperature = self.back_ptr.setting.get_temperature()
            # use disordered energies
            exp_beta_e = get_boltz_factor(energies, temperature)
            pop = exp_beta_e / np.sum(exp_beta_e)
        else:
            # specific init node
            pop = np.zeros(len(self))
            if init_option.lower() in ['sink', 'target', 't']:
                # sink: the exciton state with lowest energy
                pop[-1] = 1
            elif init_option.lower() in ['s', 'source']:
                # source: the exciton state with highest energy
                pop[0] = 1
            elif init_option in nodes:
                pop[nodes.index(init_option)] = 1
            elif all([x in nodes for x in re.split(r'[,-]', init_option)]):
                # by a series of node name:
                # 1,2,3,6-12
                s = set()
                for n in init_option.split(','):
                    if "-" in n:
                        a, b = n.split("-")
                        a, b = min((a, b)), max((a, b))
                        s.update([str(i) for i in range(int(a), int(b) + 1)])
                    else:
                        s.add(n)
                for n in s:
                    pop[nodes.index(n)] = 1 / len(s)
            else:
                raise KeyError('initial node not exist')

        # cluster: merge populations:
        if is_clustered:
            # align with rate matrix
            pop_cluster = np.zeros(size)
            for i, c in enumerate(clusters):
                # if c == 'sink':
                #     continue
                for n in c.split():
                    pop_cluster[i] += pop[nodes.index(n)]
            pop = pop_cluster

            wrap_nodes = [wrap_str(n) for n in clusters]

        else:
            wrap_nodes = [wrap_str(n) for n in nodes]

        name_len = max([max([len(n) for n in wrap_nodes]) + 1, 10])
        print('initial populations:')
        for n, p in zip(wrap_nodes, pop.flat):
            print('{{:{}}}: {{:.2f}}'.format(name_len).format(n, p))
        print()

        return pop.reshape(-1, 1)

    def __cal_dynamics(self, cluster=None):
        """
        calculate population dynamics and save to self.__pop_tuple (if not clustered)
        :param cluster: cluster tuple for self.__cluster_handler()

        the return object is 3-tuple:
        :return pop_seq: population dynamics
        :return time_sequence: time grids of pop_seq
        :return nodes: corresponding node names
        """
        is_clustered = False
        nodes = self.ExcitonName
        energies = self.ExcitonEnergies

        rate, cluster_energies, _ = self.__cluster_handler(cluster)

        if cluster_energies is not None:
            clusters = rate.keys()
            is_clustered = True
        else:
            clusters = None

        setting = self.back_ptr.setting

        # setup variables
        propagate_time = pass_int(setting['time'])
        time_grid = pass_int(setting['grid'])
        time_sequence = np.linspace(0, propagate_time, time_grid)
        print(
            'dynamics propagation setting:\n'
            '  propagate time: {} ps\n'
            '  grids on time: {}\n'.format(propagate_time, time_grid)
        )

        pop = self.get_initial_populations(nodes, energies, clusters, cluster_energies)

        # propagation
        if 'log' in setting:
            print('start dynamics calculation')
            timer_start = timeit.default_timer()

            pop_seq = alg.propagate(
                pop, rate.values, time_sequence,
                option=setting['propagate'] == 'poorman',
                print_pop=True
            )

            timer_end = timeit.default_timer()
            print('propagate time {}'.format(timer_end - timer_start))
        else:
            pop_seq = alg.propagate(
                pop, rate.values, time_sequence,
                option=setting['propagate'] == 'poorman',
            )

        if not is_clustered:
            self.__pop_tuple = pop_seq, propagate_time, time_grid
        else:
            nodes = list(clusters)

        return pop_seq, time_sequence, nodes

    def get_integrated_flux(self, cluster=None, spline_size=3000, save_to_file=False):
        pop_seq, time_sequence, nodes = self.get_dynamics(cluster)
        pop_seq2, time_sequence2 = spline_grid(pop_seq, time_sequence, spline_size)

        # retrieve the cluster information
        rate, _, plot_name = self.__cluster_handler(cluster)

        setting = self.back_ptr.setting
        y_max = pass_float(setting.get('ymax', '0.'))
        x_max = pass_float(setting.get('xmax', '0.'))

        return alg.get_integrated_flux(
                pop_seq2, rate, time_sequence2,
                norm=pass_int(setting.get('multiply', 1)),
                plot_details='log' in setting,
                plot_name=plot_name,
                y_max=y_max, x_max=x_max,
                legend='nolegend' not in setting,
                save_to_file=save_to_file
        )

    def get_dynamics(self, cluster=None):
        """
        propagate the population dynamics
        :param cluster: should be
                        1. tuple: rate matrix, cluster energies, (job name)
                        2. cluster map object
        the return object is 3-tuple:
        :return pop_seq: population dynamics
        :return time_sequence: time grids of pop_seq
        :return nodes: corresponding node names
        """
        if cluster is None:
            if self.__pop_tuple is None:
                self.__cal_dynamics()
            pop_seq, propagate_time, time_grid = self.__pop_tuple
            time_sequence = np.linspace(0, propagate_time, time_grid)
            nodes = self.ExcitonName
        else:
            pop_seq, time_sequence, nodes = self.__cal_dynamics(cluster)
        return pop_seq, time_sequence, nodes

    def animate_dynamics(self, ps1_special_option=False, dpi=100, allsite=False):
        """
        save a population transfer animation, 2D projection
        Hamiltonian and site position most be loaded in the project
        :param dpi: integer, video quality
        :param allsite: Boolean, mark all site or not
        :param ps1_special_option: Boolean, mark some important site only
        :return: None
        """
        if not self.has_hamiltonian() or self.back_ptr.SitePos is None:
            raise KeyError('please provide Hamiltonian and Cartesian coordinates')

        if self.__pop_tuple is None:
            self.__cal_dynamics()

        pop_seq, propagate_time, time_grid = self.__pop_tuple
        time_sequence = np.linspace(0, propagate_time, time_grid)

        plot.population_animatation(
            pop_seq, self.back_ptr.SitePos, self.SiteName,
            (self.EigenVectors ** 2).T,
            time_sequence, self.get_output_name(),
            ps1=ps1_special_option, dpi=dpi, allsite=allsite
        )

    def get_population_difference(self, cluster, pop_cluster=None, spline_size=3000):
        if self.__pop_tuple is None:
            print('\ncalculate the full dynamics for comparison')
            self.__cal_dynamics()

        # provided propagation result
        if pop_cluster is None:
            pop_cluster, time_sequence, _ = self.__cal_dynamics(cluster)
        else:
            _, propagate_time, time_grid = self.__pop_tuple
            time_sequence = np.linspace(0, propagate_time, time_grid)

        pop_full = self.get_comparison_to_full_dynamics(cluster[0].keys())

        if spline_size:
            pop_full, _ = spline_grid(pop_full, time_sequence, spline_size)
            pop_cluster, time_sequence = spline_grid(pop_cluster, time_sequence, spline_size)
        else:
            spline_size = pop_cluster.shape[1]
            if spline_size != pop_full.shape[1]:
                # align with provided pop_cluster
                pop_full, _ = spline_grid(pop_full, time_sequence, spline_size)

        b = sum(sum((pop_full - pop_cluster) ** 2))
        pop_diff = b / len(cluster[1]) / spline_size
        print('population dynamics square difference: {:.2e}'.format(pop_diff))
        return pop_diff

    def get_comparison_to_full_dynamics(self, clusters):
        # for population comparison (dash line in dynamics plots of cluster)

        # check if full network dynamics is calculated
        if self.__pop_tuple is None:
            print('\ncalculate the full dynamics for comparison')
            self.__cal_dynamics()

        pop_seq, _, time_grid = self.__pop_tuple
        pop_full = np.zeros((len(clusters), time_grid))

        for i, n in enumerate(self.ExcitonName):
            for v, g in enumerate(clusters):
                if n in g.split():
                    pop_full[v] += pop_seq[i]
        return pop_full
