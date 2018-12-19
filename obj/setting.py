# import local modules
from aux import *


# ============================================================


class Setting:
    def __init__(self, sys_argv, config):
        # Set output options
        np.set_printoptions(linewidth=150)
        np.set_printoptions(formatter={'float': lambda x: "{:>8.2f}".format(x)})
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 100)

        self.OptionText = {
            'd': 'Network/CG models visualization utilizing Graphviz',
            'p': 'Time-integrated flux',
            'F': 'FFA flow analysis',
            'a': 'Population dynamics animation',
            'I': 'site-state corresponding diagram',
            'r': 'Save rate constant matrices and rate input files',
            'M': 'Propagate population dynamics',
            'e': 'plot exciton population on each site'
        }

        # system info
        self.config = config

        # Project Options
        self.KeyWords = []
        self.InputFileName = self.JobName = ''

        self.Setting = copy(DefaultSetting)

        self.__mrt_param = []

        # initialize
        option_set = {opt: set() for opt in self.OptionText}
        cmd_opt = set()
        clx_opt = set()
        for item in sys_argv:
            if item[0] == '-':
                opt, details = item[1], item[2:]
                if opt == '-':
                    self.KeyWords.append(details)
                    continue

                # options
                cmd_opt.add(opt)
                if opt == 'c':
                    clx_opt.update([x for x in details])

                elif opt in self.OptionText:
                    cmd_opt.update(opt)
                    option_set[opt] = opt_processing(details[1:-1])

            # setting dict
            elif '=' in item:
                k, v, *others = item.split('=')
                if others:
                    help_message()
                self.Setting[k] = v

            else:
                if not self.InputFileName:
                    self.InputFileName = item
                elif not self.JobName:
                    self.JobName = item
                else:
                    help_message()

        if not self.InputFileName or 'h' in cmd_opt:
            # help message and exit
            help_message()

        if not self.JobName:
            self.JobName = time_string()

        self.__run_opt = option_set, sorted(clx_opt), cmd_opt
        self.print_all(*self.__run_opt)

        # plot parameters
        self.ErrorBarParams = {
            'capthick': 2.5
        }
        mpl_update = {
            'savefig.format': self.Setting['format'] if self.Setting['format'] in ('ps', 'pdf', 'svg') else 'png',
            'figure.dpi': pass_int(self.Setting['dpi'], 100)
        }
        mpl.rcParams.update(mpl_update)

    def print_all(self, option_set=None, clx_opt=None, cmd_opt=None):
        if not option_set:
            option_set = {}
        if not clx_opt:
            clx_opt = set()
        if not cmd_opt:
            cmd_opt = set()

        print('Job name: {}'.format(self.JobName))

        print('Settings:')
        [print('    {}: {}'.format(k, v)) for k, v in self.Setting.items()]

        if self.KeyWords:
            print('Keywords:')
            [print('    {}'.format(k)) for k in self.KeyWords]
            print()

        print_set('Receive options', cmd_opt)
        print_set('Clustering method options', clx_opt)
        [print_set(self.OptionText[k], v) for k, v in option_set.items()]
        print()

    def set_mrt_params(self, *params):
        try:
            if len(params) == 3:
                self.__mrt_param = tuple(map(float, params))
            else:
                params = None
        except:
            params = None

        if params is None:
            while 1:
                try:
                    params = input('please enter the over-damped Brownian oscillator bath parameters:\n'
                                   '[temperature] [reorganization energy (cm-1)] [cut-off frequency (cm-1)]\n')
                    self.__mrt_param = tuple(map(float, params.split()))
                    if len(self.__mrt_param) != 3:
                        raise KeyError
                    break
                except:
                    print('wrong format')

    def get_mrt_params(self):
        if len(self.__mrt_param) != 3:
            if 'FMO' in self.InputFileName:
                self.__mrt_param = (300, 35, 666.7)
            elif 'LHC' in self.InputFileName:
                self.__mrt_param = (300, 85, 628.4)
            elif 'PS1' in self.InputFileName or ('PSI' in self.InputFileName and 'PSII' not in self.InputFileName):
                self.__mrt_param = (300, 100, 100.4)

            else:
                self.set_mrt_params()

        print('parameters for spectral density:')
        print('    T = ', self.__mrt_param[0])
        print('    Reorganization Energy = ', self.__mrt_param[1])
        print('    Cut-off Frequency = ', self.__mrt_param[2])

        return self.__mrt_param

    def set_temperature(self, temperature=None):
        try:
            # None: will go to except
            temperature = float(temperature)
        except:
            temperature = None

        if temperature is None:
            while 1:
                try:
                    temperature = float(input('please enter temperature:'))
                    break
                except:
                    print('wrong format')
        if not self.__mrt_param:
            self.__mrt_param = [temperature]
        else:
            self.__mrt_param[0] = temperature

    def get_temperature(self):
        if not self.__mrt_param:
            if 'FMO' in self.InputFileName:
                self.__mrt_param = (300, 35, 666.7)
            elif 'LHC' in self.InputFileName:
                self.__mrt_param = (300, 85, 628.4)
            elif 'PS1' in self.InputFileName or ('PSI' in self.InputFileName and 'PSII' not in self.InputFileName):
                self.__mrt_param = (300, 100, 100.4)
            else:
                self.set_temperature()
        return self.__mrt_param[0]
