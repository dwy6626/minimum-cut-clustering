# import local modules
from lib import *
from .system import System
from .config import Config
import plot


# ============================================================


class Project:
    def __init__(self, setting, config=None):
        self.col = ['Method', 'N', 'CGM', 'OverlapFactor', 'PopDiff']

        self.setting = setting

        if config is None:
            config = Config()
        self.config = config

        self.__reference_system = None

        self.data_frame = []
        self.SitePos = None

        self.disorders = None
        self.discard_disorders = []
        self.__systems = []
        self.overlap_factors = []
        self.overlap_arrays = []

    def build_reference_system(self, argv,
                               is_rate_matrix=False, as_file_path=False,
                               additional_hamiltonian_load='',
                               additional_hamiltonian_string=''):
        if as_file_path:
            argv = self.load_file(argv)
        if additional_hamiltonian_load:
            additional_hamiltonian_string = self.load_file(additional_hamiltonian_load)
        self.__reference_system = System(
            argv, back_ptr=self, is_rate_matrix=is_rate_matrix,
            additional_hamiltonian_string=additional_hamiltonian_string
        )

        self.load_pos(self.setting.get('pos'))

        # disorder
        self.disorders = [np.zeros((1, len(self.__reference_system)))]
        disorder_counts = 0  # finally: len(self.disorders) - 1
        if self.__reference_system.has_hamiltonian():
            disorder = self.setting.get('disorder', '0')
            if isinstance(disorder, int):
                disorder_counts = disorder
            elif disorder.isdigit():
                disorder_counts = int(disorder)
            else:
                # load file:
                lines = string_to_lines(self.load_file(disorder))
                # check size
                if len(lines[0]) != len(self.__reference_system):
                    raise ValueError('size of disorders does not match the Hamiltonian')
                disorder_counts = len(lines)
                self.disorders += [np.array(line, dtype=float).reshape(1, -1) for line in lines]

        self.overlap_factors = [100] + [0] * disorder_counts
        self.overlap_arrays = [np.ones(len(self.__reference_system))] + \
                              [np.zeros(len(self.__reference_system))] * disorder_counts

        # generate disordered Hamiltonians
        self.__systems = [System(self.__reference_system, self, i + 1) for i in range(disorder_counts)]

    def get_reference_system(self):
        return self.__reference_system

    def __iter__(self):
        yield self.__reference_system
        for s in self.__systems:
            yield s

    def __repr__(self):
        return repr(self.concat())

    def re_config(self):
        self.config = Config()

    def load_file(self, path):
        path = self.input_path(path)
        print_normal("loading file at:\n    {}".format(path))
        with open(path, 'r', encoding='utf-8-sig') as f:
            file_str = f.read()
        return file_str

    def input_path(self, str1):
        if '/' not in str1:
            try:
                with open(str1) as f:
                    pass
            except:
                try:
                    with open(self.config.input_path(str1)) as f:
                        str1 = self.config.input_path(str1)
                except:
                    try:
                        with open(self.config.output_path(str1)) as f:
                            str1 = self.config.output_path(str1)
                    except FileExistsError:
                        print('input file', str1, 'does not exist')
        return str1

    def load_pos(self, path):
        if not path:
            return

        lines = string_to_lines(self.load_file(path))
        if not self.__reference_system.has_hamiltonian():
            print('please read the Hamiltonian file first')
            return

        if len(lines) != len(self.__reference_system.SiteName):
            raise ValueError("Number of sites doesn't match")

        self.SitePos = np.zeros((len(self.__reference_system), 3), dtype=float)
        if len(lines[0]) == 4:
            # site name in .pos
            coordinates = {site: r for site, *r in lines}
            for i, site in enumerate(self.__reference_system.SiteName):
                try:
                    self.SitePos[i] = coordinates[site]
                except KeyError:
                    print("Site name doesn't match")
                print_normal('{}: {}'.format(site, self.SitePos[i]))
        elif len(lines[0]) == 3:
            # x, y, z
            for i, l in enumerate(lines):
                try:
                    self.SitePos[i] = l
                except ValueError:
                    print("[x, y, z] should be numbers")
                print_normal('{}: {}'.format(self.__reference_system.SiteName[i], self.SitePos[i]))
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
            print_normal('save disorder values:', shift_file)
            np.savetxt(shift_file,
                       reduce(lambda x, y: np.append(x, y, axis=0), self.disorders),
                       delimiter=",", fmt='%.8f')

        if self.discard_disorders:
            shift_file = self.get_output_name('_discard_disorder.csv')
            print_normal('save discard disorder values:', shift_file)
            np.savetxt(shift_file,
                       reduce(lambda x, y: np.append(x, y, axis=0), self.discard_disorders),
                       delimiter=",", fmt='%.8f')

        if len(self.data_frame) == 0:
            return
        print_normal(self)

        cgm_file = self.get_output_name('.p')
        print_normal('save the clustering results to python3 pickle file (binary): {}'.format(cgm_file))
        with open(cgm_file, 'wb') as f:
            pk.dump(self, f)
        self.save_raw()

    # to a .csv file, raw data format
    def save_raw(self):
        raw_file = self.get_output_name("_results.csv")
        print_normal('save the clustering results to .csv file: {}'.format(raw_file))
        self.concat().to_csv(raw_file)

    def get_output_name(self, str1='_'):
        return self.config.output_path(self.setting.JobName + str1)

    def plot_cost(self, selector=None, save_to_file=False):
        if selector is None:
            selector = range(len(self.__systems) + 1)
        elif isinstance(selector, int):
            selector = [selector]

        for h_id, system in enumerate(self):
            if h_id not in selector:
                continue

            plot.plot_cost(
                *system.get_all_population_difference(
                    spline_size=pass_int(self.setting['spline']), save_back=True
                ),
                system.get_output_name('PopDiff'),
                x_max=pass_int(self.setting['cost']),
                print_marker=self.setting['marker'] == 'true',
                y_max=pass_float(self.setting.get('ymax', '0.')),
                legend='nolegend' not in self.setting,
                save_to_file=save_to_file
            )
