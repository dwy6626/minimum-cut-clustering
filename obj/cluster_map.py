from lib import *
from numpy.random import randint


# ============================================================


# the class manipulate nodes and cluster
class ClusterMap:
    def __init__(self, system, method, one_group=True, site_map=False):
        # initialize: a cluster list
        self.__data = []

        if site_map:
            nodes = system.SiteName
        else:
            nodes = system.ExcitonName
        self.back_ptr = system

        if one_group:
            self.__data.append(copy(nodes))
            self.__number_of_cluster = 1
            self.__dict = {n: self.__data[0] for n in nodes}
            for i in range(len(nodes) - 1):
                self.__data.append([])
        else:
            self.__number_of_cluster = len(nodes)
            self.__dict = dict()
            for n in nodes:
                # refer self.__data to the cluster which it belongs to
                self.__data.append([n])
                self[n] = self.__data[-1]

        for n in self.__dict:
            if not n.isdigit():
                self.__sort_key = sort_str_key
                break
        else:
            self.__sort_key = sort_int_key

        # some info
        self.method = method
        self.__info = None

        self.__sorted = False
        self.__print_decimal = '.' + self.back_ptr.back_ptr.setting['decimal'] + 'f'

    def __len__(self):
        return self.__number_of_cluster

    def __getitem__(self, n):
        return self.__dict[n]

    def __setitem__(self, k, v):
        return self.__dict.__setitem__(k, v)

    def __str__(self):
        r = ''
        for c in self.groups():
            r += str(c) + ' '
        r = r[:-1].replace("'", '').replace(',', '')
        return r

    def __repr__(self):
        return str(self)

    # for the output of pandas
    def __iter__(self):
        return repr(self)

    def keys(self):
        return self.__dict.keys()

    # make n into a single-node cluster
    def cut(self, n):
        # check if n is already single? => do nothing
        if len(self[n]) == 1:
            return

        self.__sorted = False
        self.__number_of_cluster += 1
        self[n].remove(n)
        self[n] = self.__data[self.__number_of_cluster - 1] = [n]

    # cut but no re-group
    def group_cut(self, set1):
        new_cluster = list(set1)
        if len(new_cluster) == 1:
            self.cut(new_cluster.pop())
            return

        self.__sorted = False

        # pre-grouping by existing groups
        pre_grouping = {}
        for n in new_cluster:
            for g in pre_grouping:
                if id(self[n]) == g:
                    pre_grouping[id(self[n])].append(n)
                    break
            else:
                pre_grouping[id(self[n])] = [n]

        for _, s in pre_grouping.items():
            self.group_up(s)

    # make the input set into a new cluster:
    def group_up(self, set1):
        new_cluster = list(set1)

        if len(new_cluster) == 1:
            self.cut(new_cluster.pop())
            return

        self.__sorted = False
        self.__number_of_cluster += 1
        for n in new_cluster:
            self[n].remove(n)
            # if a cluster become empty: n_c minus 1
            if not self[n]:
                self.__number_of_cluster -= 1

        # put all empty to the back
        self.__organize()
        # use the first empty as the new cluster
        self.__data[self.__number_of_cluster - 1] = new_cluster
        for n in new_cluster:
            self[n] = self.__data[self.__number_of_cluster - 1]

    # merge two sets
    def merge(self, n1, n2):
        if self[n1] == self[n2]:
            return
        self.__number_of_cluster -= 1
        self.__sorted = False
        self[n1].extend(self[n2])
        self[n2].clear()
        self[n2] = self[n1]
        # put all empty to the back
        self.__organize()

    # move n into an existing cluster
    def move(self, n, s):
        self.__sorted = False
        new_cluster = list(s)
        # test if the set s is really exist as a single cluster
        if set(s).difference(self[new_cluster[0]]):
            raise KeyError('move method of Map accepts only an existing cluster')

        self[n].remove(n)
        if not self[n]:
            self.__number_of_cluster -= 1
            self.__organize()

        self[new_cluster[0]].append(n)
        self[n] = self[new_cluster[0]]

    # keep the total number of cluster
    def move_and_kick(self, n, s):
        kick_able = [g for g in self.__data[:self.__number_of_cluster] if len(g) > 1]
        kick = len(self[n]) == 1
        self.move(n, s)

        # in this situation: due to n is merged with s, total number of cluster will -= 1
        # find a cluster with 2 or more nodes and cut one of them from others
        if kick:
            s2 = kick_able[randint(len(kick_able))]
            n2 = s2[randint(len(s2))]
            self.cut(n2)

    def adjust_total_number(self, sign):
        if sign > 0:
            kick_able = [g for g in self.__data[:self.__number_of_cluster] if len(g) > 1]
            s = kick_able[randint(len(kick_able))]
            n2 = s[randint(len(s))]
            self.cut(n2)
        elif sign < 0:
            while True:
                x, y = randint(self.__number_of_cluster), randint(self.__number_of_cluster)
                if x != y:
                    break
            self.merge(self.__data[x][0], self.__data[y][0])

    # check if there is a empty cluster:
    def __organize(self):
        new_data = []
        new_empty = []
        for c in self.__data:
            if c:
                new_data.append(c)
            else:
                new_empty.append([])
        self.__data[:] = new_data + new_empty

        self.__update_pointer()

    def __update_pointer(self):
        for i, c in enumerate(self.__data[:self.__number_of_cluster]):
            for n in c:
                self[n] = self.__data[i]

    def __sort(self):
        self.__sorted = True

        new_order = []
        # sort node by number
        for i in range(self.__number_of_cluster):
            new_order.append(sorted(self.__data[i], key=self.__sort_key))
        # sort cluster by smallest number
        new_order = sorted(new_order, key=lambda x: self.__sort_key(x[0]))
        self.__data[:self.__number_of_cluster] = new_order

        self.__update_pointer()

    def update_info(self, str1):
        self.__info = str1

    def update_cutoff(self, cutoff):
        self.update_info('Cut = ' + format(cutoff, self.__print_decimal) + ':')

    def groups(self):
        if not self.__sorted:
            self.__sort()

        return self.__data[:self.__number_of_cluster].copy()

    def print_all(self):
        if not self.__sorted:
            self.__sort()

        if self.__info:
            print(self.__info, self.__number_of_cluster, 'clusters:', self)
        else:
            print(self.__number_of_cluster, 'clusters:', self)

    # generate a copy with same back_ptr
    # site_map options is not supported
    def copy(self):
        new_map = ClusterMap(self.back_ptr, self.method, one_group=False)
        for g in self.groups():
            new_map.group_up(g)
        return new_map

    def save(self):
        """
        after clustering, print the cluster result and save back to the project
        """
        self.print_all()
        proj = self.back_ptr.back_ptr

        # add a coarse-grained model
        h = self.back_ptr.get_index()

        if len(proj.data_frame) <= h:
            proj.data_frame.append(pd.DataFrame(columns=proj.col, dtype=object))

        df = proj.data_frame[h]

        # run an self check to prevent adding redundant table element
        for i, (m, n, _, _, _) in df.iterrows():
            if m == self.method and n == len(self):
                return

        m = self.method
        n = len(self)
        mp = self.copy()
        c = None
        olfac = proj.overlap_factors[h - 1]
        sr = pd.Series((m, n, mp, olfac, c), index=proj.col)

        df.loc[len(proj.data_frame[h])] = sr


# ============================================================


def sort_str_key(x):
    return x


def sort_int_key(x):
    return int(x)


