# import local modules
from aux import *


# ============================================================


class Config:
    def __init__(self):
        # Global Variables
        self.__OutPutFolder = 'output/'
        self.__InPutFolder = 'input/'
        self.DotPath = ''
        self.__mpl_dict = {
            "font.family": 'Arial',
            'font.size': 30,
            'lines.linewidth': 3.5,
            'lines.markersize': 20,
            'markers.fillstyle': 'none',
            'errorbar.capsize': 10,
            'legend.fontsize': 25,
            'axes.labelsize': 30,
            'axes.titlesize': 30,
            'figure.figsize': (9, 6),
            'legend.frameon': False,
            'lines.markeredgewidth': 3
        }
        self.config()

    def config(self):
        from time import strftime

        # now, in 24-hour format
        # %H = 24-hour, %l = 12-hour
        print("Now: " + strftime("%X, %x"))

        print("Terminal path: ")
        print("  " + os.getcwd(), '\n')

        # check packages:
        checked_file = '.mincut_configure'
        env_checked = os.popen('ls ' + checked_file).read()
        if not env_checked:

            print("Check required packages:")
            requirements = {
                'networkx': '2.1',
                'pydot': '1.2.3',
                'scikit-learn': '0.20.1',
                'matplotlib': '2.2.2',
                'pandas': '0.22.0',
                'scipy': '1.0.0',
            }

            import pip
            from pkg_resources import parse_version
            if parse_version(pip.__version__) > parse_version('9.0.3'):
                from pip._internal.utils.misc import get_installed_distributions
                installed_packages = get_installed_distributions()
            else:
                installed_packages = pip.get_installed_distributions()

            for pkg in installed_packages:
                ver = requirements.get(pkg.key, None)
                if ver is not None:
                    chk = 'outdated'
                    if parse_version(ver) <= parse_version(pkg.version):
                        chk = 'ok'
                        requirements.pop(pkg.key)
                    print('  {:<25}{:<10}    {:10}'.format(pkg.key, pkg.version, chk))

            if requirements:
                print("\nplease use pip to install the following requirements")
                for pkg, ver in requirements.items():
                    print(" ", pkg, '>=', ver)
                exit(1)
            else:
                print('All requirements installed\n')

            # check graphviz:
            print("Check graphviz: ")
            dot_path = os.popen('which dot').read()

            if not dot_path:
                print('  which dot not found, ')
                print('  please input the path of your dot, ')
                dot_path = input('  or input enter to exit.')

                if dot_path:

                    try:
                        dot_ver = os.popen(dot_path + ' -V').read()
                    except:
                        print('  path error')
                        path_exit()
                else:
                    path_exit()

            else:
                print("  " + dot_path)

            with open(checked_file, 'w') as f:
                f.write(dot_path)
        else:
            with open(checked_file, 'r') as f:
                dot_path = f.read()

        self.DotPath = dot_path.replace('\n', '')
        print_1_line_stars()

        # plot setting:
        self.update_matplotlibrc()

        # check output folder
        self.check_output()

        print('Environment checking： pass')
        print('Job started.\n')

    def output_path(self, job):
        return self.__OutPutFolder + job

    def input_path(self, job):
        return self.__InPutFolder + job

    def check_output(self):
        try:
            f = open(self.__OutPutFolder + '.output.log', 'w')
            f.close()
        except:
            os.system("mkdir " + self.__OutPutFolder)

    def update_matplotlibrc(self):
        # avoid backend error
        import platform
        local_os = platform.system()
        print('Your os is', local_os, '\n')
        if local_os == 'Linux':
            mpl.use('Agg')
        # else:
        #     mpl.use('Qt4Agg')

        print('upadate the matplotlib rc:')
        for k, v in self.__mpl_dict.items():
            print('    {}: {}'.format(k, v))
        mpl.rcParams.update(self.__mpl_dict)


# ============================================================


def path_exit():
    print("  you need to install graphviz first\n"
          "  visit https://www.graphviz.org for more information")
    exit(1)
