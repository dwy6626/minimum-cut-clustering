# import build-in modules
import re
from itertools import combinations, permutations, count


# import 3rd party modules
import numpy as np
from scipy import constants as const
import networkx as nx


# ============================================================


DefaultSetting = {
    'format': 'pdf',
    'init': 'equally',
    'norm': 'true',
    'propagate': 'exponent',
    'cutoff': '0.01',
    'decimal': '2',
    'time': '6',
    'grid': '100',
    'spline': '3000',
    'marker': 'true',
    'dpi': '100',
    'cost': '15'
}


# ============================================================


# Print many stars
def print_stars():
    for i in range(72):
        print("*", end="")  # no new line
    print()  # new line


# Print many many stars
def print_1_line_stars():
    print()
    print_stars()
    print()


# Print help message
def help_message():
    from os import popen
    print('Preparing help message...\n')
    print(popen('cat doc/usage').read())
    exit(1)


# format: -n[1,3,6,7-11,n,c]
# default: -n => -n[n, c]
def opt_processing(item):
    return_opt = set()
    for str1 in item.split(','):
        if not str1:
            continue

        # a-b
        if '-' in str1:
            if len(str1.split('-')) == 2:
                a, b = str1.split('-')
                if a.isdigit() and b.isdigit():
                    c = [int(a), int(b)]
                    return_opt.update(range(min(c), max(c) + 1))
                else:
                    help_message()
            else:
                help_message()
        elif str1.isdigit():
            return_opt.add(int(str1))
        elif str1 in ['n', 'c']:
            return_opt.add(str1)
        else:
            help_message()

    if not return_opt:
        return {'n', 'c'}
    else:
        return return_opt


def print_set(set_name, set1):
    if not set1:
        return
    print('{}: '.format(set_name), end='')
    opt_ls = list(map(str, set1))
    for i in opt_ls[:-1]:
        print("'" + i + "'", end=', ')
    print("'" + opt_ls[-1] + "'")


def H_suffix(disorder_id):
    if disorder_id > 0:
        return '_H' + str(disorder_id)
    else:
        return ''


def pass_float(str1, default=0.):
    if isinstance(str1, float):
        return str1
    if isinstance(str1, int):
        return float(str1)
    p = re.compile('^\d+(\.\d+)?$')
    match = p.match(str1)
    if match:
        return float(match.group(0))
    if str1 in DefaultSetting:
        return float(DefaultSetting[str1])
    return default


def pass_int(str1, default=0):
    if isinstance(str1, int):
        return str1
    if isinstance(str1, float):
        return int(str1)
    if str1.isdigit():
        return int(str1)
    if str1 in DefaultSetting:
        return int(DefaultSetting[str1])
    return default


def find_duplicate(ls):
    r = {}
    for i, n in enumerate(ls):
        if n in r:
            r[n].append(i)
        else:
            r[n] = [i]
    return {k: v for k, v in r.items() if len(v) > 1}


def has_duplicate(ls):
    k = set()
    for i in ls:
        if i in k:
            return True
        else:
            k.add(i)
    else:
        return False


def set_to_str(set1):
    return str(sorted(set1)).replace("'", '').replace(',', '')


def get_ranking(ls):
    r = [0] * len(ls)
    for i, x in enumerate(sorted(range(len(ls)), key=lambda y: ls[y])):
        r[x] = i
    return r


# for boltzmann dist
def get_boltz_factor(energies, t):
    beta = get_beta_from_T(t)
    return np.exp(-beta * energies)


# constant
def get_beta_from_T(t):
    return 1 / t / const.k * const.h * const.c * 100


def paper_method(this_method, option=1):
    """
    :param this_method: method (str)
    :param option:
        0: unmodified
        1: BUC -> minimum cut
        2: BUC -> bottom-up clustering
        3: TDC -> AC (old)
    :return: renamed method (str)
    """
    if option == 3:
        c_method = this_method
        return c_method.replace('TDC', 'AC')

    elif option in (1, 2):
        if this_method == 'BUC':

            if option == 1:
                return 'minimum cut'
            else:
                return 'bottom-up clustering'
        elif this_method == 'KM':
            return 'k-means'
        elif this_method == 'DC':
            return 'cut-off'  # rate constant cut-off
        elif this_method == 'TDC':
            return 'top-down clustering'

    return this_method


def method_to_number(method):
    # main methods
    m_ls = ['BUC', 'TDC', 'SR', 'SC', 'DC', 'KM']
    v_ls = [0, 1, 11, 12, 20, 25]
    for m, v in zip(m_ls, v_ls):
        if m in method:
            r = v
            method.replace(m, '')
            break
    else:
        r = 100

    # flow normalization
    if 'uN' in method:
        r += 0.4
        method.replace('_uN', '')

    # other alphabets
    for c in method:
        if 97 >= ord(c) >= 122:
            r += 0.01 * (ord(c) - 90)

    return r


def get_pattern_size(val, cutoff=0.0, minsize=0.0, maxsize=10000):
    if val < cutoff:
        return minsize, 'black'
    else:
        # return the size and default color 'b' => to be modified by groups
        return val * (maxsize - minsize) / (1 - cutoff), 'b'


def get_figsize_for_position_plot(size):
    if size > 30:
        return [10, 6]
    else:
        return [6, 4]


def get_cluster_graph(cluster):
    # tuple: rate matrix, energies, name
    graph = nx.DiGraph()
    graph.add_nodes_from(
        ((cluster[0].keys()[i], {'energy': cluster[1][i]}) for i in range(len(cluster[0].keys())))
    )
    graph.add_weighted_edges_from(
        ((m, n, cluster[0][m][n]) for m, n in permutations(cluster[0].keys(), 2))
    )
    return graph
