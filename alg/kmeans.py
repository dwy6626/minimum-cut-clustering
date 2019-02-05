from lib import *
from . import clx


KM_inf = 10 ** 5


# ============================================================


# TODO: how to go faster? this algorithm actually search for local minimum only
# TODO: I have wrote an GA-assisted algorithm in the old version, should I implement here?
def k_means_like(system, power=2):
    def km_init(_n_c, job_name):
        # check the initial state:
        _shift = 0
        _sign = 1
        while True:
            if _n_c + _shift in ref_maps:
                break
            elif _n_c - _shift in ref_maps:
                _sign = -1
                break
            _shift += 1

        _cluster_map = ref_maps[_n_c + _shift * _sign].copy()
        if _shift != 0:
            print_more("random generate map from: {}".format(_cluster_map))

        while _shift > 0:
            _cluster_map.adjust_total_number(-_sign)
            _shift -= 1

        _cluster_map.method = job_name
        _cluster_map.update_info(None)

        print_more('initial: {}'.format(_cluster_map))
        return _cluster_map

    def var_cal(_map):
        r = 0
        for g in _map.groups():
            sub_sum = 0
            for _m, _n in combinations(g, 2):
                sub_sum += aux_matrix[_m][_n]
            r += sub_sum / len(g) ** 2 / 2
            if r > VarNow:
                return float('inf')
        return r

    # a new KM-like function:
    def __clx(_nc, epsilon=0):
        job_name = 'KM'
        cluster_map = km_init(_nc, job_name)
        global VarNow
        VarNow = init_var = var_cal(cluster_map)
        print_more('init var = {:.2e}'.format(init_var))
        # start algorithm
        while 1:
            # record the original state
            index_now = {}
            # a matrix that move node _n to set i (in list_set)
            var_matrix = np.zeros((len(system), _nc))
            list_set = deepcopy(cluster_map.groups())

            # elements allow kick
            kick_ables = []
            for _s in list_set:
                if len(_s) > 1:
                    kick_ables.extend(_s)

            # kick out
            dic_kick = {}
            for _j, _n in enumerate(system.ExcitonName):
                if len(cluster_map[_n]) > 1:
                    # move n to set i
                    for _i, _s in enumerate(list_set):
                        if _n in _s:
                            index_now[_j] = _i
                            var_matrix[_j][_i] = VarNow
                        else:
                            s0 = cluster_map[_n]
                            cluster_map.move(_n, _s)
                            var_matrix[_j][_i] = var_cal(cluster_map)
                            cluster_map.move(_n, s0)
                else:
                    # if _n in a single-node cluster:
                    # enter set i and kick one out
                    for _i, _s in enumerate(list_set):
                        if _n in _s:
                            index_now[_j] = _i
                            var_matrix[_j][_i] = VarNow
                        else:
                            cluster_map.move(_n, _s)
                            var_kick_min = float('inf')
                            for m in kick_ables:
                                s0 = cluster_map[m]
                                cluster_map.cut(m)
                                v = var_cal(cluster_map)
                                cluster_map.move(m, s0)
                                if var_kick_min > v:
                                    var_kick_min = v
                                    dic_kick[_n] = m
                            var_matrix[_j][_i] = var_kick_min
                            cluster_map.cut(_n)

            # check and move
            diff = VarNow - np.min(var_matrix)

            if diff > epsilon:
                _j, _i = np.unravel_index(np.argmin(var_matrix, axis=None), var_matrix.shape)
                if system.ExcitonName[_j] in dic_kick:
                    print_more(
                        'add {} to {}, kick {} from {}'.format(
                            system.ExcitonName[_j], set_to_str(list_set[_i]), dic_kick[system.ExcitonName[_j]],
                            set_to_str(cluster_map[dic_kick[system.ExcitonName[_j]]])
                        )
                    )
                    cluster_map.move(system.ExcitonName[_j], list_set[_i])
                    cluster_map.cut(dic_kick[system.ExcitonName[_j]])
                else:
                    print_more(
                        'move {} from {} to {}'.format(
                            system.ExcitonName[_j], set_to_str(list_set[index_now[_j]]), set_to_str(list_set[_i])
                        )
                    )
                    cluster_map.move(system.ExcitonName[_j], list_set[_i])
                print_more(cluster_map)

                VarNow = var_cal(cluster_map)
                print_more('var = {:.2e}'.format(VarNow))

            # no move: output the results
            else:
                print_more('init var - final var = {:.2e}'.format(init_var - var_cal(cluster_map)))
                cluster_map.save()
                print_more('')
                break

    # a simple brute-force solution
    def __km_bf():
        def __var_cal(_group):
            _r = 0
            for _m, _n in combinations(_group, 2):
                _r += aux_matrix[_m][_n]
            return _r / len(_group) ** 2 / 2

        def __partition(_collection):
            if len(_collection) == 1:
                yield [(_collection, 0)]
                return

            _first = _collection[0]
            for _smaller in __partition(_collection[1:]):
                # _now_val = sum([_val for _subset, _val in _smaller])
                # insert `first` in each of the subpartition's subsets
                for _n, (_subset, _subset_val) in enumerate(_smaller):
                    yield _smaller[:_n] + [([_first] + _subset, __var_cal([_first] + _subset))] + _smaller[
                                                                                                  _n + 1:]
                # put `first` in its own subset
                yield [([_first], 0)] + _smaller

        _results = {}

        for _p in __partition(system.ExcitonName):
            # put the smaller "cost" into the hashing
            _val = sum([v for g, v in _p])
            if len(_p) not in _results:
                _results[len(_p)] = _p, _val
            elif _results[len(_p)][1] > _val:
                _results[len(_p)] = _p, _val

        for _nc in range(2, len(system)):
            _clx_map = system.get_new_map('KM', one_group=False)
            _p, _v = _results[_nc]
            for _g, _ in _p:
                _clx_map.group_up(_g)
            print_normal('var = {}'.format(_v))
            _clx_map.save()

    # modify the rate matrix to up_triangle (pickup larger one)
    aux_rate = system.RateConstantMatrix
    uptri = np.triu(aux_rate.values, k=1)
    lowtri = np.tril(aux_rate.values, k=-1).T
    aux_2 = np.where(uptri > lowtri, uptri, lowtri)

    # create an aux matrix to calculate variance:
    def __inverse_aux(_val):
        if _val == 0:
            return float('inf')
        else:
            return 1 / _val
    aux_2 = np.vectorize(__inverse_aux)(aux_2) ** power
    aux_2 = np.triu(aux_2, k=1)
    aux_matrix = deepcopy(system.RateConstantMatrix)
    aux_matrix.loc[:] = aux_2 + aux_2.T

    # renew the "infinity"
    n_inf = len(aux_matrix.values[aux_matrix == np.inf])
    aux_matrix.values[aux_matrix == np.inf] = -1
    not_inf_sum = sum(sum(aux_matrix.values)) + n_inf
    global KM_inf
    if not_inf_sum > KM_inf:
        KM_inf += int(not_inf_sum) + 1
        print_more('renew the KM_inf value to {:.2e}'.format(KM_inf))
    # replace inf with this value
    aux_matrix.values[aux_matrix == -1] = KM_inf

    # use the keyword to run brute-force algorithm
    if 'bfkm' in system.back_ptr.setting:
        __km_bf()
        return

    cost = pass_int(system.back_ptr.setting['cost'])
    iter_range = range(2, min([len(system), 1+int(cost)]))

    global VarNow

    # run k_means
    # check if DC is run
    df = system.back_ptr.data_frame

    if len(df) > system.get_index():
        df = df[system.get_index()]
        df = df.loc[df['Method'] == 'DC']
        ref_maps = {}
        for _, (_, n, cgm, *_) in df.iterrows():
            ref_maps[n] = cgm

    else:
        print_more('  run cut-off methods for initial clusters')
        ref_maps = clx.cut_off_method(system, 4, pass_map=True)

    for nc in iter_range:
        VarNow = float('inf')
        __clx(nc)


# ============================================================


def flow_kmeans(list_cap, n_clusters=6, lifetime=False):
    from sklearn.cluster import KMeans
    import plot

    # prevent: sample < group
    if len(list_cap) < n_clusters:
        n_clusters = len(list_cap)

    # no grouping for number(sample) = 1
    if len(list_cap) > 1:
        kmeans_fit = KMeans(n_clusters=n_clusters).fit(
            np.array(list_cap).reshape(-1, 1)
        )
        centers = kmeans_fit.cluster_centers_
        labels = kmeans_fit.labels_

        # 0-> smaller rates
        # max -> larger rates
        ranking = get_ranking(centers)

        # reverse sort_list_cent for lifetime diagram
        if lifetime:
            ranking.reverse()

        dic = {i: ranking[g] for i, g in zip(list_cap, labels)}

    else:
        centers = [0]
        dic = {list_cap[0]: 0}

    # color
    a = len(centers)
    cmap = plot.plt.cm.get_cmap('Blues', a * 5)
    cmap2 = []
    for i in range(2*a, 3*a):
        rgb = cmap(i)[:3]  # first 3 : rgb
        cmap2.append(plot.rgb2hex(rgb))

    r = {}
    for f, k in dic.items():
        # width, color
        r[f] = 3 + k * 3 / a, cmap2[k]

    return r
