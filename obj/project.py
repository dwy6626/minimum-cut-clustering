# import local modules
from aux import *
from .system import System
from .setting import Setting
import alg
import plot


# ============================================================


class Project:
    def __init__(self, argv, config):
        self.col = ['Method', 'N', 'CGM', 'OverlapFactor', 'PopDiff']

        self.setting = Setting(argv, config)

        self.reference_system = System(self.setting.InputFileName, self)

        # disorder
        self.disorders = [np.zeros((1, len(self.reference_system)))]
        disorder_counts = 0  # finally: len(self.disorders) - 1
        if self.reference_system.has_hamiltonian():
            disorder = self.setting.Setting.get('disorder', '0')
            if disorder.isdigit():
                disorder_counts = int(disorder)
            else:
                # load file:
                lines = self.load_file(disorder)
                # check size
                if len(lines[0]) != len(self.reference_system):
                    raise ValueError('size of disorders does not match the Hamiltonian')
                disorder_counts = len(lines)
                self.disorders += [np.array(line, dtype=float).reshape(1, -1) for line in lines]

        self.discard_disorders = []
        self.overlap_factors = [100] + [0] * disorder_counts
        self.overlap_arrays = [np.ones(len(self.reference_system))] + \
                              [np.zeros(len(self.reference_system))] * disorder_counts

        # generate disordered Hamiltonians
        self.__systems = [System(self.reference_system, self, i + 1) for i in range(disorder_counts)]

        self.data_frame = []

        self.SitePos = None
        if 'pos' in self.setting.Setting:
            self.load_pos(self.setting.Setting['pos'])

    def __iter__(self):
        yield self.reference_system
        for s in self.__systems:
            yield s

    def __repr__(self):
        return repr(self.concat())

    # load file into lines array with no Null component
    def load_file(self, path):
        path = self.input_path(path)
        print("loading file at:\n    {}".format(path))
        with open(path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        # split:
        lines = map(lambda x: re.split('[\s,;]+', x), lines)
        # remove null string
        lines = [[x for x in l if x] for l in lines]
        # remove null line
        return [l for l in lines if l]

    def input_path(self, str1):
        config = self.setting.config
        if '/' not in str1:
            try:
                with open(str1) as f:
                    pass
            except:
                try:
                    with open(config.input_path(str1)) as f:
                        str1 = config.input_path(str1)
                except:
                    try:
                        with open(config.output_path(str1)) as f:
                            str1 = config.output_path(str1)
                    except FileExistsError:
                        print('input file', str1, 'does not exist')
        return str1

    def load_pos(self, path):
        lines = self.load_file(path)
        if not self.reference_system.has_hamiltonian():
            print('please read the Hamiltonian file first')
            return

        if len(lines) != len(self.reference_system.SiteName):
            raise ValueError("Number of sites doesn't match")

        self.SitePos = np.zeros((len(self.reference_system), 3), dtype=float)
        if len(lines[0]) == 4:
            # site name in .pos
            coordinates = {site: r for site, *r in lines}
            for i, site in enumerate(self.reference_system.SiteName):
                try:
                    self.SitePos[i] = coordinates[site]
                except KeyError:
                    print("Site name doesn't match")
                print(site, self.SitePos[i])
        elif len(lines[0]) == 3:
            # x, y, z
            for i, l in enumerate(lines):
                try:
                    self.SitePos[i] = l
                except ValueError:
                    print("[x, y, z] should be numbers")
                print(self.reference_system.SiteName[i], self.SitePos[i])
        else:
            raise ValueError("[site, x, y, z] should be provided in the position input file")

    # collect individual frame (for each Hamiltonian) into a big one
    def concat(self):
        dfs = self.data_frame

        if len(dfs) == 1:
            df = dfs[0]
            column = ['OverlapFactor']
            if not any(df['PopDiff'].values):
                column.append('PopDiff')
            return df[df.columns.difference(column)]

        # delete column if all none
        # check the first dataframe only: assume all are similar
        for column in dfs[0].columns:
            if column not in ['Method', 'N', 'CGM']:
                for i in dfs[0][column].values.flat:
                    if i is not None:
                        break
                else:
                    dfs = [df[df.columns.difference([column])] for df in dfs]

        return pd.concat(dfs, keys=range(len(dfs)))

    # save the results (this object) into a pickle file
    def save(self):
        if len(self.disorders) > 1:
            shift_file = self.get_output_name('_disorder.csv')
            print('save disorder values:', shift_file)
            np.savetxt(shift_file,
                       reduce(lambda x, y: np.append(x, y, axis=0), self.disorders),
                       delimiter=",", fmt='%.8f')

        if self.discard_disorders:
            shift_file = self.get_output_name('_discard_disorder.csv')
            print('save discard disorder values:', shift_file)
            np.savetxt(shift_file,
                       reduce(lambda x, y: np.append(x, y, axis=0), self.discard_disorders),
                       delimiter=",", fmt='%.8f')

        if len(self.data_frame) == 0:
            return
        print(self)

        cgm_file = self.get_output_name('.p')
        print('save the clustering results to python3 pickle file (binary):', cgm_file)
        with open(cgm_file, 'wb') as f:
            pk.dump(self, f)
        self.save_raw()

    # to a .csv file, raw data format
    def save_raw(self):
        raw_file = self.get_output_name("_results.csv")
        print('save the clustering results to .csv file:', raw_file)
        self.concat().to_csv(raw_file)

    def print_log(self, *strs, **kwargs):
        if 'log' in self.setting.KeyWords:
            print(*strs, **kwargs)

    def get_output_name(self, str1='_'):
        return self.setting.config.output_path(self.setting.JobName + str1)

    # output controller
    def output_cluster_results(self, options, latex=False, cost=False):
        if len(self.data_frame) == 0:
            return

        for h_id, system in enumerate(self):
            df = self.data_frame[h_id]
            for i, (m, n, cgm, _, _) in df.iterrows():
                judge_ls = (n, 'c')

                if system.has_hamiltonian() and options['I'].intersection(judge_ls):
                    plot.plot_tf(system, cgm)
                    print_1_line_stars()

                if options['e'].intersection(judge_ls):
                    plot.plot_exst(system, allsite='allsite' in self.setting.KeyWords, clx_map=cgm)

                dot = options['d'].intersection(judge_ls)
                ffa = options['F'].intersection(judge_ls)
                dynamics = options['M'].intersection(judge_ls)
                flux = options['p'].intersection(judge_ls)
                rate = options['r'].intersection(judge_ls)

                if any([dot, ffa, dynamics, flux, rate, cost]):
                    # tuple: rate matrix, energies, name
                    cluster = *system.get_cluster(cgm), system.get_plot_name(cgm)

                    # graphviz / dot files
                    if dot:
                        nx_aux.nx_graph_draw(
                            get_cluster_graph(cluster), system, cluster[2] + 'Rate', rc_order=list(cluster[0].keys())
                        )
                        print_1_line_stars()

                    if ffa:
                        alg.flow_analysis(system, cluster)
                        print_1_line_stars()

                    if rate:
                        if latex:
                            alg.print_rate_matrix(cluster[0], pass_int(self.setting.Setting['decimal']))
                        alg.save_rate(cluster[0], cluster[2], cluster[1])
                        print_1_line_stars()

                    # dynamics
                    dynamics_opt = [dynamics, flux, cost]
                    if any(dynamics_opt):
                        system.get_dynamics(
                            cluster,
                            pyplot_output=dynamics_opt[0],
                            flux=dynamics_opt[1],
                            cost=i if dynamics_opt[2] else None
                        )
                        print_1_line_stars()

        if cost:
            self.plot_cost()

    def plot_cost(self):
        for h_id, system in enumerate(self):
            if not system.is_population_difference_calculated():
                system.get_dynamics(cost=h_id, pyplot_output=False)

            plot.plot_cost(
                system,
                assigned_cost=pass_int(self.setting.Setting['cost']),
                print_marker=self.setting.Setting['marker'] == 'true',
                y_max=pass_float(self.setting.Setting.get('ymax', '0.')),
                legend='nolegend' not in self.setting.KeyWords
            )
