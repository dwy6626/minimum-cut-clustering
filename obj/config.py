# import local modules
from aux import *


CheckFile = '.mincut_configure'


# ============================================================


class Config:
    def __init__(self, colab=False):
        # Set output options
        np.set_printoptions(linewidth=150)
        np.set_printoptions(formatter={'float': lambda x: "{:>8.2f}".format(x)})
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 100)

        # Global Variables
        self.__OutPutFolder = 'output/'
        self.__InPutFolder = 'input/'
        self.__DotPath = ''
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
        self.__is_colab = colab
        self.config()

    def config(self):
        # plot setting:
        self.update_matplotlibrc()

        # check output folder
        self.check_output()

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
        if local_os == 'Linux' and not self.__is_colab:
            # if cannot modify backend, than don't do that
            try:
                mpl.use('Agg')
            except:
                pass

        print('upadate the matplotlib rc:')
        for k, v in self.__mpl_dict.items():
            print('    {}: {}'.format(k, v))
        mpl.rcParams.update(self.__mpl_dict)

    def get_graphaviz_dot_path(self):
        if not self.__DotPath:
            dot_path_file = os.popen('ls ' + CheckFile).read()
            if not dot_path_file:
                # check graphviz:
                print("Check graphviz: ")
                dot_path = os.popen('which dot').read()

                if not dot_path:
                    Warning('  which dot: dot not found, ')
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

                with open(CheckFile, 'w') as f:
                    f.write(dot_path)
            else:
                with open(CheckFile, 'r') as f:
                    dot_path = f.read()

            self.__DotPath = dot_path.replace('\n', '')

        return self.__DotPath


# ============================================================


def path_exit():
    print("  you need to install graphviz first\n"
          "  visit https://www.graphviz.org for more information")
    exit(1)
