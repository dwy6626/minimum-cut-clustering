# import local modules
from lib import *


string_to_logger_level = {
    # 'verbose': verboselogs.VERBOSE,
    'verbose': logging.DEBUG,
    'normal': logging.INFO,
    'quiet': logging.WARNING
}


# ============================================================


class Setting:
    def __init__(self):
        # now, in 24-hour format
        # %H = 24-hour, %l = 12-hour
        print_normal("Now: " + strftime("%X, %x"))

        print_normal("Terminal path: ")
        print_normal("  " + os.getcwd() + '\n')

        print_1_line_stars()

        self.__OptionText = {
            'd': 'Network/CG models visualization utilizing Graphviz',
            'p': 'Time-integrated flux',
            'F': 'FFA flow analysis',
            'a': 'Population dynamics animation',
            'I': 'site-state corresponding diagram',
            'r': 'Save rate constant matrices and rate input files',
            'M': 'Propagate population dynamics',
            'e': 'plot exciton population on each site'
        }

        # Project Options
        self.__keywords = set()
        self.InputFileName = self.JobName = ''

        self.__dict = copy(DefaultSetting)

        self.__mrt_param = []

    def receive_arguments(self, sys_argv):
        """
        :param sys_argv: list of strings, command line options
        :return: output options
        """
        default_name = strftime("%Y%m%d%H%M")

        option_set = {opt: set() for opt in self.__OptionText}
        cmd_opt = set()
        clx_opt = set()
        for item in sys_argv:
            if item[0] == '-':
                opt, details = item[1], item[2:]
                if opt == '-':
                    self.__keywords.add(details)
                    continue

                # options
                cmd_opt.add(opt)
                if opt == 'c':
                    clx_opt.update([x for x in details])

                elif opt in self.__OptionText:
                    cmd_opt.update(opt)
                    option_set[opt] = opt_processing(details[1:-1])

            # setting dict
            elif '=' in item:
                k, v, *others = item.split('=')
                if others:
                    help_message()
                self[k] = v

            else:
                if not self.InputFileName:
                    self.InputFileName = item
                elif not self.JobName:
                    self.JobName = item
                else:
                    help_message()

        if 'h' in cmd_opt:
            # help message and exit
            help_message(0)

        if not self.JobName:
            self.JobName = default_name

        run_opt = option_set, sorted(clx_opt), cmd_opt

        # update matplotlib rc from arguments
        mpl_update = {
            'savefig.format': self['format'] if self['format'] in ('ps', 'pdf', 'svg') else 'png',
            'figure.dpi': pass_int(self['dpi'], 100)
        }
        mpl.rcParams.update(mpl_update)

        # update the parameters for modified Redfield theory
        temperature = pass_float(self.get('temperature'))
        lambda0 = pass_float(self.get('lambda'))
        gamma0 = pass_float(self.get('gamma'))

        # set logging option
        if 'v' in cmd_opt:
            logger_option = 'verbose'
        elif 'q' in cmd_opt:
            logger_option = 'quiet'
        else:
            logger_option = 'normal'

        self.set_logger(logger_option)
        if get_module_logger_level() < 30:
            self.print_all(*run_opt)

        if temperature > 0 and lambda0 > 0 and gamma0 > 0:
            self.set_mrt_params([temperature, lambda0, gamma0])
        return run_opt

    @staticmethod
    def set_logger(string):
        """
        set the logger level
        :param string: 'verbose' / 'normal' / 'quiet'
               otherwise, input is ignored
        """
        if string in string_to_logger_level:
            module_log.set_module_logger(string_to_logger_level[string])
        else:
            print('please choose the following one mode:')
            print(string_to_logger_level.keys())

    def print_all(self, option_set=None, clx_opt=None, cmd_opt=None):
        if not option_set:
            option_set = {}
        if not clx_opt:
            clx_opt = set()
        if not cmd_opt:
            cmd_opt = set()

        print('Job name: {}'.format(self.JobName))

        print('Settings:')
        [print('    {}: {}'.format(k, v)) for k, v in self.items()]

        if self.__keywords:
            print('Keywords:')
            [print('    {}'.format(k)) for k in self.__keywords]
            print()

        print_set('Receive options', cmd_opt)
        print_set('Clustering method options', clx_opt)
        [print_set(self.__OptionText[k], v) for k, v in option_set.items()]
        print()

    def set_mrt_params(self, params=None):
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

        print_normal('parameters for spectral density:')
        print_normal('    T = {:.2f}'.format(self.__mrt_param[0]))
        print_normal('    Reorganization Energy = {:.2e}'.format(self.__mrt_param[1]))
        print_normal('    Cut-off Frequency = {:.2e}'.format(self.__mrt_param[2]))

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

    def __getitem__(self, n):
        return self.__dict[n]

    def __setitem__(self, k, v):
        return self.__dict.__setitem__(k, v)

    def get(self, n, default=None):
        return self.__dict.get(n, default)

    def items(self):
        return self.__dict.items()

    def __contains__(self, k):
        return self.__keywords.__contains__(k)
