from aux import *
import plot


# ============================================================


def check_rate(system, cut_off=0.001):
    print('rate constant matrix analysis')
    print('{:14}{:14}{:10}{:10}{:10}{:13}'
          .format('higher state', 'lower state', 'h->l', 'l->h', 'diff', 'energy diff'))
    max_e_diff = 0
    rate = system.RateConstantMatrix
    for i, j in combinations(range(len(system)), 2):
        e_diff = system.ExcitonEnergies[j] - system.ExcitonEnergies[i]
        if e_diff < 0:
            i, j = j, i
            e_diff = -e_diff
        u, v = system.ExcitonName[i], system.ExcitonName[j]

        max_e_diff = e_diff if e_diff > max_e_diff else max_e_diff
        rate_diff = rate[v][u] - rate[u][v]

        # print if h->l with larger reverse transfer rate
        if - rate_diff > cut_off:
            print('{:<14}{:<14}{:<10.3f}{:<10.3f}{:<10.3f}{:<13.1f}'.
                  format(v, u, rate[v][u], rate[u][v], rate_diff, e_diff))
    print('max energy difference = ', format(max_e_diff, '.1f'))


def bottleneck_rate(system):
    print('find bottlenecks in rate matrix:')
    print('{:8}{:8}{:8}{:8}{:8}'
          .format('node', 'in', 'out', 'diff', 'ratio'))
    rate = system.RateConstantMatrix
    for n in system.ExcitonName:
        out_sum = -rate[n][n]
        in_sum = np.sum(rate.loc[n, :]) + out_sum
        ratio = in_sum / out_sum
        diff = in_sum - out_sum
        print('{:8}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}'.format(n, in_sum, out_sum, diff, ratio))


def propagate(pop0, rate, time_sequence, option=0, print_pop=False):
    """
    :param option:
        0: matrix exponent
        else: poorman
    """
    size, _ = np.shape(rate)

    # initial population
    pop_seq = pop0

    # propagate the dynamics
    if option:
        pop = pop0
        delta_t = time_sequence[1] - time_sequence[0]
        for t in range(len(time_sequence) - 1):
            c = np.eye(size) + rate * delta_t

            pop = c.dot(pop)
            # pop = pop / sum(abs(pop))
            if print_pop:
                print(pop.reshape(1, -1))
            # append the result on a sequence to plot
            pop_seq = np.append(pop_seq, pop, axis=1)
    else:
        # exponential method
        for t in time_sequence[1:]:
            pop2 = expm(rate * t).dot(pop0)
            if print_pop:
                print(pop2.reshape(1, -1))
            pop_seq = np.append(pop_seq, pop2, axis=1)

    return pop_seq


def cal_flux(P, R):
    F = R.dot(np.diag(P))
    return F - F.T


def get_integrated_flux(
        pop_seq, rate, time,
        norm=1, plot_details=False,
        plot_name='', divide=5, y_max=0, x_max=0, legend=True
):
    nodes = rate.keys()
    rate = rate.values
    wrap_nodes = [wrap_str(n.split()) for n in nodes]
    name_len = max([max([len(n) for n in wrap_nodes]) + 1, 10])
    size, length = np.shape(pop_seq)
    flux = np.zeros((size, size, length))

    # calculate instant flux in each time interval:
    for i in range(length):
        flux[:, :, i] = cal_flux(pop_seq[:, i], rate) * norm

    # calculate integrated flux:
    flux_long_time = flux[:, :, -1]
    flux_subtract = flux - flux_long_time.reshape(size, size, 1)
    integrated_flux = integrate.cumtrapz(flux_subtract, time, initial=0)
    integrated_flux_wo_subtract = integrate.cumtrapz(flux, time, initial=0)
    abs_integration = np.trapz(np.abs(flux), time, axis=2)

    integrated_flux_matrix = np.zeros((size, size))
    sum_abs = 0
    flux_ls = []
    for i in range(size):
        for j in range(size):
            if i != j:
                f = integrated_flux[i, j, -1]
                if f > 0:
                    sum_abs += abs_integration[i, j]

                    # calculate the population flow
                    delta_flow = integrated_flux[i, j, 1:] - integrated_flux[i, j, :-1]
                    forward = np.sum(delta_flow[delta_flow > 0])
                    backward = np.abs(np.sum(delta_flow[delta_flow < 0]))
                    integrated_flux_matrix[i, j] = forward
                    integrated_flux_matrix[j, i] = backward

                    # sort by "loading" of the energy channel
                    flux_ls.append([abs_integration[i, j], integrated_flux[i, j, -1],
                                    integrated_flux_wo_subtract[i, j, -1], forward, backward, i, j])

    # print results
    flux_ls.sort(reverse=True)
    print('{{:{}}}{{:{}}}{{:10}}{{:15}}{{:10}}{{:10}}{{:10}}{{:10}}{{:10}}{{:7}}{{:10}}'.format(name_len, name_len).format(
        'source', 'target', 'maximum', 'equilibrium', 'integral', 'int - eq', 'abs sum', 'forward', 'backward', 'rate', 'backrate'))
    for f_abs, f_sub, f, fw, bw, i, j in flux_ls:
        if f_sub > 0:
            print('{{:{}}}{{:{}}}{{:<10.3f}}{{:<15.4f}}'
                  '{{:<10.3f}}{{:<10.3f}}{{:<10.3f}}{{:<10.3f}}{{:<10.3f}}{{:<7.2f}}{{:<10.2f}}'.format(name_len, name_len).format(
                    wrap_nodes[j], wrap_nodes[i],
                    max(flux[i, j, :]), flux_long_time[i, j],
                    f, f_sub, f_abs, fw, bw, rate[i, j], rate[j, i]
            ))

    print('integrate over all abs flux:', format(sum_abs, '.2f'))
    print('flow matrix:')
    print(integrated_flux_matrix)

    if plot_details:
        flux_series = []
        int_flux_series = []
        flux_names = []
        int_flux_names = []
        int_flux_wo_series = []
        for _, f_sub, _, fw, bw, i, j in flux_ls:
            name = wrap_nodes[j] + ' \u2192 ' + wrap_nodes[i]
            int_flux_names.append(name)
            flux_names.append('{:.2f}: {}'.format(f_sub, name))
            flux_series.append(flux[i, j, :])
            int_flux_series.append(integrated_flux[i, j, :])
            int_flux_wo_series.append(integrated_flux_wo_subtract[i, j, :])

        axes_names = ('Time (ps)', 'Time Integrated Flux')
        plot.plot_series(
            int_flux_series, time, flux_names, axes_names, plot_name + 'IntegratedFlux',
            divide=divide, y_max=y_max, x_max=x_max, legend=legend
        )
        plot.plot_series(
            int_flux_wo_series, time, flux_names, axes_names, plot_name + 'IntegratedFluxWithoutSubtract',
            divide=divide, y_max=y_max, x_max=x_max, legend=legend
        )
        axes_names = ('Time (ps)', 'Flow (ps' + r'$^{\mathregular{-1}}$' ')')
        plot.plot_series(
            flux_series, time, flux_names, axes_names, plot_name + 'Flux',
            divide=divide, y_max=y_max, x_max=x_max, legend=legend, zero=True
        )

    return integrated_flux_matrix


def save_rate(rate_matrix, save_name, energies):
    if save_name and energies is not None:
        nodes = rate_matrix.keys()
        save_name += 'RateMatrix'
        print('save rate matrix:\n', rate_matrix)

        # input key file
        with open(save_name + '.in', 'w') as f:
            for n in nodes:
                f.write(n.replace(' ', '_') + ' ')

            f.write('\n\n')
            for e in energies:
                f.write(str(e) + ' ')
            f.write('\n\n')

            for n in nodes:
                for m in nodes:
                    f.write(str(rate_matrix[m][n]) + ' ')
                f.write('\n')

        # readable file: csv
        rate_matrix.to_csv(save_name + '.csv')


def print_latex_matrix(system, decimal=2):
    size = len(system)
    formatter = "{{:.{}f}}".format(decimal)
    print_rate_matrix(system.RateConstantMatrix, decimal)

    if system.has_hamiltonian():
        tf = system.EigenVectors ** 2
        print('LaTeX Table of Hamiltonian:\n')
        print('Site:')
        for i in range(size-1):
            print(system.SiteName[i].replace('a', '\\textit{a}').replace('b', '\\textit{b}'), '& ', end='')
        print(system.SiteName[-1])
        print('\nSquare of Eigenvectors (Exciton Populations):')
        print(' & ', end='')
        for i in range(size - 1):
            print('exciton', i + 1, end=' & ')
        print('exciton', size, '\\\\ \\hline')
        for i in range(size):
            print(system.SiteName[i].replace('a', '\\textit{a}').replace('b', '\\textit{b}'), '& ', end='')
            for j in range(size):
                print(formatter.format(tf[j, i]), end=' ')
                if j != size - 1:
                    print('& ', end='')
                else:
                    print('\\\\')
        print('\\hline\n')


def print_rate_matrix(rate_matrix, decimal):
    formatter = "{{:.{}f}}".format(decimal)
    nodes = rate_matrix.keys()
    size = len(nodes)
    print('LaTeX Table of Rate Constant Matrix:\n')
    print('\\hline')
    for i in range(size):
        print('exciton', wrap_str(nodes[i].split()), end='')
        if i != size - 1:
            print(' & ', end='')
        else:
            print(' \\\\ \\hline')
    for n in nodes:
        for i, m in enumerate(nodes):
            print(formatter.format(rate_matrix[m][n]), end=' ')
            if i != size - 1:
                print('& ', end='')
            else:
                print('\\\\')
    print('\\hline\n')
