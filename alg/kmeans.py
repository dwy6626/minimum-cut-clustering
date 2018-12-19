from aux import *
from . import clx


KM_inf = 10 ** 5


# ============================================================


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
            system.back_ptr.print_log("random generate map from:", _cluster_map)

        while _shift > 0:
            _cluster_map.adjust_total_number(-_sign)
            _shift -= 1

        _cluster_map.method = job_name
        _cluster_map.update_info(None)

        if 'log' in system.back_ptr.setting.KeyWords:
            print('initial: ', end='')
            print(_cluster_map)
        return _cluster_map

    def var_cal_gakm(_map):
        r = 0
        for g in _map.groups():
            sub_sum = 0
            for _m, _n in combinations(g, 2):
                sub_sum += aux_matrix[_m][_n]
            r += sub_sum / len(g) ** 2 / 2
        return r

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

    def _new_random_map(_nc, job_name):
        node_ls = system.ExcitonName
        # generate a array of graph size
        # with random numbers 0 to _nc-1
        while 1:
            seeds = np.ceil(np.random.rand(len(system)) * _nc).astype('int') - 1
            if len(np.unique(seeds)) == _nc:
                break

        _cluster_map = system.get_new_map(job_name, one_group=False)
        for j in range(_nc):
            group = []
            for k in np.where(seeds == j)[0]:
                group.append(node_ls[k])
            _cluster_map.group_up(group)
        return _cluster_map

    def __ga_km(_nc, n_run=5, n_generation=30, n_pool=20, p_mutation1=.8, p_mutation2=.97, p_replace=.8):
        r = []
        for run in range(n_run):
            job_name = 'GA_KM_{}'.format(run)

            # initialize pool
            r.append([])
            p_mut = p_mutation1
            pool = []
            for _ in range(n_pool):
                pool.append(_new_random_map(_nc, job_name))

            var_of_pool = [var_cal_gakm(m) for m in pool]
            node_ls = system.ExcitonName

            r[-1].append(var_of_pool)

            for generation in range(n_generation):
                # selection with probability proportional to fitness
                fit_of_pool = 1/np.array(var_of_pool)
                selected_id = np.random.choice(range(n_pool), n_pool, p=fit_of_pool/np.sum(fit_of_pool))
                selected = [pool[i].copy() for i in selected_id]

                # a simple crossover:
                # no idea so jump

                # mutation
                for m in selected:
                    while 1:
                        if np.random.rand() < p_mut:
                            break
                        chosen_n = node_ls[np.random.randint(0, len(system))]
                        groups = m.groups()
                        groups.remove(m[chosen_n])  # prevent effortless move
                        m.move_and_kick(chosen_n, groups[np.random.randint(len(groups))])

                # comes some foreigner
                while 1:
                    if np.random.rand() < p_replace:
                        break
                    selected.append(_new_random_map(_nc, job_name))

                # kill
                # remain the least var
                var_of_selected = [var_cal_gakm(m) for m in selected]
                var_of_pool, pool = zip(*sorted(zip(var_of_pool + var_of_selected, pool + selected),
                                                key=lambda x: x[0])[:n_pool])
                var_of_pool, pool = list(var_of_pool), list(pool)
                r[-1].append(var_of_pool)

                # variance control
                if len(np.unique(var_of_pool)) < 0.25 * n_pool:
                    p_mut = p_mutation2
                else:
                    p_mut = p_mutation1

            # for m in pool:
            pool[0].save()
        return r

    # a new KM-like function:
    def __clx(_nc, epsilon=0):
        job_name = 'KM'
        cluster_map = km_init(_nc, job_name)
        global VarNow
        VarNow = init_var = var_cal(cluster_map)
        system.back_ptr.print_log('init var =', init_var)
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
                    system.back_ptr.print_log('add', system.ExcitonName[_j], 'to', set_to_str(list_set[_i]) + ', kick',
                                              dic_kick[system.ExcitonName[_j]], 'from',
                                              set_to_str(cluster_map[dic_kick[system.ExcitonName[_j]]]))
                    cluster_map.move(system.ExcitonName[_j], list_set[_i])
                    cluster_map.cut(dic_kick[system.ExcitonName[_j]])
                else:
                    system.back_ptr.print_log('move', system.ExcitonName[_j], 'from', set_to_str(list_set[index_now[_j]]),
                                              'to', set_to_str(list_set[_i]))
                    cluster_map.move(system.ExcitonName[_j], list_set[_i])
                system.back_ptr.print_log(str(cluster_map))

                VarNow = var_cal(cluster_map)
                system.back_ptr.print_log('var =', VarNow)

            # no move: output the results
            else:
                system.back_ptr.print_log('init var - final var =', init_var - var_cal(cluster_map))
                cluster_map.save()
                system.back_ptr.print_log()
                break

    # modify the rate matrix to up_triangle (pickup larger one)
    aux_rate = system.RateConstantMatrix
    uptri = np.triu(aux_rate.values, k=1)
    lowtri = np.tril(aux_rate.values, k=-1).T
    aux_2 = np.where(uptri > lowtri, uptri, lowtri)

    # create an aux matrix to calculate variance:
    aux_2 = (1 / aux_2) ** power
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
        system.back_ptr.print_log('renew the KM_inf value to', KM_inf)
    # replace inf with this value
    aux_matrix.values[aux_matrix == -1] = KM_inf

    cost = pass_int(system.back_ptr.setting.Setting['cost'])
    iter_range = range(2, min([len(system), 1+int(cost)]))

    global VarNow

    # run k_means
    # check if DC is run
    df = system.back_ptr.data_frame

    if len(df) != 0:
        df = df[system.get_index()]
        df = df.loc[df['Method'] == 'DC']

    if len(df) == 0:
        system.back_ptr.print_log('  run cut-off methods for initial clusters')
        ref_maps = clx.cut_off_method(system, 4, pass_map=True)

    else:
        ref_maps = {}
        for _, (_, n, cgm, *_) in df.iterrows():
            ref_maps[n] = cgm

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
