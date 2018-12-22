from matplotlib.colors import rgb2hex, rgb_to_hsv, hsv_to_rgb
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
import matplotlib.animation as anim


from lib import *

DefaultMap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

PSI_sitelist = [
    'B26', 'B24', 'EC-A1', 'EC-B2', 'EC-A3',
    'B25', 'B38', 'B37', 'A31', 'A24',
    'A39', 'B35', 'A03', 'A14', 'L01',
    'B02', 'B15', 'A35', 'A17'
]

Markers = [
    'x', '+', 'o', 'd', '^', 'v', '>', '<', '1', '2', '3', '4', 'o', '*', 'h', 'D'
]

LHCIImon_IDls_SchlauCohen = [
    5, 6, 4, 3, 13, 12, 9, 11, 10, 1, 2, 8, 7, 0
]

LHCIImon_IDls_InstFlux = [
    12, 13, 3, 5, 7, 8, 6, 4, 9, 11, 10, 2, 1, 0
]

LHCIImon_IDls = [
    12, 13, 3, 6, 4, 5, 7, 8, 2, 1, 10, 11, 9, 0
]

FMO_IDls = [
    7, 0, 1, 2, 3, 6, 5, 4
]

# ============================================================


def node_color_energy(nodes, energies):
    energy_list = sorted(zip(energies, nodes))

    cmap2 = colormap(len(nodes), bright=True)
    cmap2.reverse()

    return {energy_list[i][1]: rgb2hex(cmap2[i]) for i in range(len(nodes))}


def colormap(color_number=256, shift=0, bright=False, dark=False, transparency=1, color_map='rainbow'):
    cmap2 = []
    cmap = plt.cm.get_cmap(color_map, color_number + shift * 2)
    for i in range(color_number):
        rgb = cmap(shift + i)[:3]  # first 3 : rgb
        if bright:
            hsv = rgb_to_hsv(rgb)
            hsv[-1] = 1
            hsv[-2] *= .6
            rgb = hsv_to_rgb(hsv)
        if dark:
            hsv = rgb_to_hsv(rgb)
            hsv[-1] = 0.7
            rgb = hsv_to_rgb(hsv)
        if transparency != 1:
            rgb = list(rgb) + [transparency]
        cmap2.append(rgb)
    return cmap2


def save_fig(name, things=None, output=True):
    """
    save (and close) or show matplotlib picture
    :param name: file name
    :param things: plt.savefig(bbox_extra_artists=things)
    :param output: show picture only
    :return:
    """
    if output:
        print('save figure:', name)
        plt.savefig(fname='{}'.format(name), bbox_extra_artists=things, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_series(
        series, x_grid, y_names, axes_names, plot_name,
        y_max=0, x_max=0,
        series2=None, divide=5, zero=False, custom_colormap=None,
        trimer=False, legend=True, save_to_file=False
):
    """
    split data into different figures
    """

    # grouping the excitons:
    # if trimer:
    #     # the trimer option suppress the divide option and custom_colormap option!
    #     groups = tuple({x[0] for x in inp.StateName})
    #     group_pop = []
    #     # group_org = []
    #     group_names = []
    #     group_cmap = []
    #
    #     for _ in groups:
    #         group_pop.append(np.zeros((1, len(Time))))
    #         # group_org.append(np.zeros((1, len(Time))))
    #         group_names.append([])
    #
    #     # for LHCII trimer: by monomer (prefixes)
    #     for i, cluster_name in enumerate(trimer.rate_matrix.keys()):
    #         # if trimer.dad:
    #         #     # assign the cluster to the monomer by frequencies
    #         #     first_alphabet_dict = {x: 0 for x in groups}
    #         #     for node_name in cluster_name.split():
    #         #         first_alphabet_dict[node_name[0]] += 1
    #         #     this_group = groups.index(max(first_alphabet_dict, key=lambda x: first_alphabet_dict[x]))
    #         # else:
    #         this_group = groups.index(cluster_name[0])
    #
    #         group_cmap.append(custom_colormap)
    #
    #         group_pop[this_group] = np.append(group_pop[this_group], [series[i, :]], axis=0)
    #         # group_org[this_group] = np.append(group_org[this_group], [series2[i, :]], axis=0)
    #         group_names[this_group].append(cluster_name)
    #     group_series = [p[1:] for p in group_pop]
    #     # group_series2 = [p[1:] for p in group_org]

    if len(series) > divide:
        lower_bound = 0
        group_series = []
        group_series2 = []
        group_names = []
        group_cmap = []
        while True:
            upper_bound = min(lower_bound + divide, len(series))
            group_series.append(series[lower_bound:upper_bound])
            if series2 is not None:
                group_series2.append(series2[lower_bound:upper_bound])
            group_names.append(y_names[lower_bound:upper_bound])
            if custom_colormap is not None:
                group_cmap.append(custom_colormap[lower_bound:upper_bound])
            else:
                group_cmap.append(DefaultMap)
            lower_bound += divide
            if lower_bound >= len(series):
                break
    else:
        group_cmap = [custom_colormap] if custom_colormap is not None else [DefaultMap]
        group_series = [series]
        group_series2 = [series2]
        group_names = [y_names]

    for g_i, (plot_ser, plot_names, plot_cmap) in enumerate(zip(group_series, group_names, group_cmap)):
        plt.figure()

        if zero:
            plt.plot(np.linspace(0, x_grid[-1], 10), np.zeros(10), '--', c='gray', linewidth=1)

        for y_i, (y_vals, name, cmap) in enumerate(zip(plot_ser, plot_names, plot_cmap)):
            plt.plot(x_grid, y_vals, label=name, c=cmap)
            if series2 is not None:
                plt.plot(x_grid, group_series2[g_i][y_i], '--', c=cmap)

        # axis maximum
        if y_max > 0:
            ymax = y_max
        else:
            ymax = max(np.amax(plot_ser), 0 if series2 is None else np.amax(group_series2[g_i]))
            if ymax > 1:
                ymax = 1

        if x_max > 0:
            xmax = x_max
        else:
            xmax = x_grid[-1]

        plt.axis([x_grid[0], xmax, np.amin(plot_ser), ymax])

        # push things to file
        fig_objects = []

        if legend:
            lgn = plt.legend()
            fig_objects.append(lgn)

        plt.ylabel(axes_names[1])
        plt.xlabel(axes_names[0])
        plt.tick_params()

        # sci notation
        if ymax < 0.3:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        # don't number the subplot if there is only 1 plot
        if len(group_names) > 1:
            this_name = '{}_{}'.format(plot_name, g_i)
        else:
            this_name = plot_name
        save_fig(this_name, fig_objects, output=save_to_file)


def plot_cost(
        system, x_max=100, print_marker=True,
        y_max=0, legend=True, save_to_file=False
):
    # https://matplotlib.org/api/markers_api.html#module-matplotlib.markers

    if len(system) > x_max:
        len_x = x_max
    else:
        len_x = len(system) - 1

    df = system.back_ptr.data_frame[system.get_index()]
    fig = plt.figure()
    ax = fig.gca()
    i_marker = -1

    plot_params = {}
    for it_m in sorted(set(df['Method']), key=method_to_number):

        # rename methods
        c_method = paper_method(it_m, set(df['Method']))

        to_plot = df.loc[df['Method'] == it_m]
        plot_value = to_plot['PopDiff']
        plot_number = to_plot['N']

        if print_marker:
            i_marker += 1
            plot_params['marker'] = Markers[i_marker]

        # draw cost
        plt.plot(*zip(*sorted(zip(plot_number, plot_value))), label=c_method,
                 **plot_params)

    # figure settings
    plt.xlabel('Number of Clusters')
    plt.ylabel('Population Difference')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    ax.xaxis.grid(linestyle="--")

    if y_max > 0:
        ymax = y_max
    else:
        ymax = ax.get_ylim()[1]

    plt.axis([2, len_x, 0, ymax])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig_objects = []
    if legend:
        lgn = plt.legend(loc='upper right')
        fig_objects.append(lgn)

    save_fig(system.get_output_name('PopDiff'), fig_objects, output=save_to_file)


def plot_tf(system, clx_map=None, cutoff=0.1, save_to_file=False):
    print('option -I: site-exciton corresponding plot')

    def pickup_str(str1, str2):
        r = ''
        for s in str1.split(','):
            if s != str2:
                for i in range(len(s)):
                    r += ' '
            else:
                r += s
            r += '  '
        return r

    setting = system.back_ptr.setting
    energies, excitons = zip(*sorted(zip(system.ExcitonEnergies, system.ExcitonName)))
    excitons = list(excitons)

    if len(system) > 20:
        fig_size = [24, 14]
    elif len(system) > 10:
        fig_size = [10, 8]
    else:
        fig_size = [8, 6]
    ecl_size = fig_size[0] / len(system)

    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()
    patches = []
    min_energy = energies[0]
    max_energy = energies[-1]
    a = ecl_size * (len(system) - 1) / fig_size[0] * 0.7
    b = ecl_size * (max_energy - min_energy) / fig_size[1]
    x_label = []

    if 'LHC2mon' in setting.InputFileName:
        id_ls = LHCIImon_IDls
    elif 'FMO' in setting.InputFileName:
        id_ls = FMO_IDls
    else:
        id_ls = list(range(len(system)))

    # change the sequence of basis in eigenvectors
    v2 = system.EigenVectors ** 2
    indexing = [system.ExcitonName.index(n) for n in excitons]
    tf = v2[id_ls, ][:, indexing]

    color_dict = dict()
    if clx_map:
        # sort clusters by minimum energy member:
        ref_energies = np.array(system.get_original().ExcitonEnergies)
        min_energies = [min((ref_energies[system.ExcitonName.index(n)] for n in cluster)) for cluster in clx_map.groups()]
        groups = [s for _, s in sorted(zip(min_energies, clx_map.groups()))]

        color_number = len(clx_map)
        # for the color code
        for i, c in enumerate(groups):
            for n in c:
                color_dict[n] = len(clx_map) - i - 1

    else:
        color_number = len(system)
        # sort color by the reference system
        for n in system.ExcitonName:
            color_dict[n] = len(system) - system.get_original().ExcitonName.index(n) - 1

    color_array = []
    cmap2 = colormap(color_number)

    # axis parameters
    tick_params = {}
    x_n = np.arange(len(system))
    if len(x_n) > 30:
        plt.xticks(rotation=45)
        tick_params['labelsize'] = 10
    else:
        plt.xticks(rotation=30)
        tick_params['labelsize'] = 25

    for i in range(len(system)):
        x_label.append(system.SiteName[id_ls[i]])
        for j in range(len(system)):
            e = energies[j]
            if tf[i, j] > cutoff:
                size_b, _ = get_pattern_size(tf[i, j], maxsize=b)
                size_b *= 0.6

                # plot line
                plt.plot(np.linspace(i - 1/2, i + 1/2, 10), np.ones(10) * e, c='black', lw=2)

                patches.append(Ellipse((i, e), a, size_b))
                color_array.append(mpl.colors.colorConverter.to_rgb(
                    cmap2[color_dict[excitons[j]]]),)

    p = PatchCollection(patches, zorder=3)

    # y grid
    things = []
    name_ls = excitons.copy()
    print(name_ls)
    print(energies)

    # check too close y: j and j+1
    for i, y in enumerate(energies[:-1]):
        if energies[i+1] - y < b/3 * tick_params['labelsize'] / 20:
            name_ls[i+1] = name_ls[i] + ',' + name_ls[i+1]

    # y2 title
    maxlen = max([len(s.replace(",", "")) for s in name_ls])
    y2title = plt.text(len(system)-1 + a/2 + maxlen*a*0.045*tick_params['labelsize'],
                       (max_energy - min_energy) / 2 + min_energy,
                       'Exciton States', rotation=270,
                       va='center', ha='left')
    things.append(y2title)

    # y2 ticks
    y2_x = len(system)-1+2*a/3
    for i, state in enumerate(excitons):
        line_leng = 10
        line_x = np.linspace(-a/2, len(system)-1+a/2, line_leng)
        line_y = np.ones(line_leng) * energies[i]
        plt.plot(line_x, line_y, '--', c='#d8d8d8', zorder=0)
        color = cmap2[color_dict[state]]
        for j, name in enumerate(name_ls):
            if state == name.split(',')[-1]:
                plt.text(y2_x, energies[j], pickup_str(name_ls[j], state), color=color,
                         va='center', ha='left', size=tick_params['labelsize'])

    p.set_color(color_array)
    ax.add_collection(p)

    plt.axis([-a/2, len(system)-1+a/2, min_energy-b/2, max_energy+b/2])

    ax.set_xticks(x_n)
    ax.set_xticklabels(x_label)

    plt.tick_params(**tick_params)
    ax.xaxis.grid(linestyle="--")

    plt.xlabel('Site')
    plt.ylabel('Exciton Energies (cm' + r'$^{\mathregular{-1}}$' ')')

    if clx_map:
        str1 = '{}_{}c_'.format(clx_map.method, len(clx_map))
    else:
        str1 = 'Full_'
    save_fig(system.get_output_name(str1 + 'SiteExciton'), things, output=save_to_file)


def get_p_shift(site, ps1=False):
    # for PSI:
    if ps1:
        # label_shift_up = []
        # label_shift_down = []
        point_shift_up = 'A37', 'B13', 'EC-A3', 'A34', 'A22', 'A08', \
                         'A03', 'B14', 'B18', 'B05', 'EC-B2', 'B09'
        point_shift_down = 'A21', 'L02', 'A19', 'B23', 'EC-A2', 'EC-B3', \
                           'A33', 'A12', 'A27', 'A38'
        point_shift_up_2 = 'B37', 'A16', 'B26', 'sink'
        point_shift_down_2 = 'B16', 'A28'

    else:
        point_shift_up, point_shift_down, \
        point_shift_up_2, point_shift_down_2 = [], [], [], []

    if site in point_shift_down:
        p_shift = -2
    elif site in point_shift_up:
        p_shift = 2
    elif site in point_shift_up_2:
        p_shift = 5
    elif site in point_shift_down_2:
        p_shift = -5
    else:
        p_shift = 0
    return p_shift


def set_ax_limit(ax, site_positions, ps1=False):
    # for PSI:
    if ps1:
        ax.set_xlim(41.78, 154.09)
        ax.set_ylim(58.28, 177.87)
    else:
        x_min = np.min(site_positions[:, 0])
        x_max = np.max(site_positions[:, 0])
        y_min = np.min(site_positions[:, 1])
        y_max = np.max(site_positions[:, 1])
        shift = 0.2
        x_shift = lim_diff(x_min, x_max) * shift
        y_shift = lim_diff(y_min, y_max) * shift
        ax.set_xlim(x_min - x_shift, x_max + x_shift)
        ax.set_ylim(y_min - y_shift, y_max + y_shift)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def lim_diff(l, u):
    return u-l


def text_pos(ax, right=False, medium=False, xshift=0.02, yshift=0.02):
    if medium:
        y = lim_diff(*ax.get_ylim()) / 2 + ax.get_ylim()[0]
    else:
        y = yshift * lim_diff(*ax.get_ylim()) + ax.get_ylim()[0]
    if right:
        x = -xshift * lim_diff(*ax.get_xlim()) + ax.get_xlim()[1]
    else:
        x = xshift * lim_diff(*ax.get_xlim()) + ax.get_xlim()[0]

    return x, y


def plot_dyanmics(
    pop_seq, time_sequence, pop_names, plot_name='', pop_seq2=None,
    y_max=0, x_max=0, legend=True,
    divide=100, save_to_file=False
):
    """
    plot population dynamics
    handle: the axes names, colors
    """
    size = len(pop_names)
    plot_name += 'Dynamics'
    axes_names = ('Time (ps)', 'Population')

    if divide < min(9, size):
        cmap = DefaultMap[:divide] * ((size - 1) // divide + 1)
    else:
        cmap = colormap(size)
        cmap.reverse()

    plot_series(
        pop_seq, time_sequence, pop_names, axes_names, plot_name,
        series2=pop_seq2, divide=divide, custom_colormap=cmap,
        y_max=y_max, x_max=x_max,
        legend=legend, save_to_file=save_to_file
    )
    return


def population_animatation(
        pop_seq, pos_file, site_names, eigenvectors, time_sequence, anime_name,
        dpi=100, ps1=False, maxsize=20000, allsite=False
):
    print('Animate population dynamics')

    size, length = np.shape(pop_seq)
    fig = plt.figure(figsize=get_figsize_for_position_plot(size))
    ax = fig.gca()

    points = pos_file[:, :2]
    points[:, 1] += np.vectorize(get_p_shift)(site_names, ps1)

    size_ar1 = np.vectorize(get_pattern_size)(pop_seq[:, 0], maxsize=maxsize)
    scat = ax.scatter(points[:, 0], points[:, 1], alpha=.4,
                      s=size_ar1[0], c=size_ar1[1],
                      edgecolors=mpl.colors.colorConverter.to_rgba('w', alpha=0.1))

    set_ax_limit(ax, pos_file, ps1)

    # site label
    for i, site in enumerate(site_names):
        if not ps1 or (allsite or site in PSI_sitelist):
            plt.text(*points[i], site,
                     # va='center',
                     ha='center',
                     size=12, alpha=0.8)

    txt = plt.text(*text_pos(ax), '', size=14)

    def pop_update(frame):
        _time = time_sequence[frame]
        txt.set_text('{:.2f} ps'.format(_time))
        _size_ar = np.vectorize(get_pattern_size)(pop_seq[:, frame].dot(eigenvectors), maxsize=maxsize)
        scat.set_sizes(_size_ar[0])
        scat.set_facecolors(_size_ar[1])
        print('frame {}, {:.2f} ps'.format(frame, _time))

    animation = anim.FuncAnimation(fig, pop_update, frames=length)

    # Set up formatting for the movie files
    writer = anim.writers['ffmpeg'](metadata=dict(artist='YCC lab'), fps=20)
    name = anime_name + '.mp4'
    print('save animation: {}'.format(name))
    animation.save(name, writer=writer, dpi=dpi)


def plot_exst(system, cutoff=0.1, clx_map=None, allsite=False, save_to_file=False):
    """
    plot the positions of excitons
    real space projection to 2D
    """
    site_positions = system.back_ptr.SitePos

    if not system.has_hamiltonian() or site_positions is None:
        raise Warning('please provide Hamiltonian and Cartesian coordinates')

    tf = system.EigenVectors ** 2
    cutoff **= 2

    text_param = {'ha': 'center', 'size': 12}

    if clx_map:
        plot_name = '{}_{}c_'.format(clx_map.method, len(clx_map))
        ref_energies = np.array(system.get_original().ExcitonEnergies)
        min_energies = [min((ref_energies[system.ExcitonName.index(n)] for n in cluster)) for cluster in clx_map.groups()]
        iter_states = [s for _, s in sorted(zip(min_energies, clx_map.groups()))]
    else:
        plot_name = 'Full_'
        iter_states = [[s] for s in system.ExcitonName]

    importance = {}

    for state_ls in iter_states:
        fig = plt.figure(figsize=get_figsize_for_position_plot(len(system)))
        ax = fig.gca()

        for i in range(len(system)):
            site = system.SiteName[i]
            x, y, z = site_positions[i]
            p_shift = get_p_shift(site)

            tf_ls = []

            # sum over the coefficient ^ 2
            for state in state_ls:
                tf_ls.append(tf[i, system.ExcitonName.index(state)])
            importance[site] = sum(tf_ls)

            size, color = get_pattern_size(importance[site], cutoff=cutoff, maxsize=2000, minsize=10)

            plt.scatter(x, y + p_shift,
                        s=size, c=color, alpha=.4,
                        edgecolors=mpl.colors.colorConverter.to_rgba('w', alpha=0.1))

        for site, (x, y, z) in zip(system.SiteName, site_positions):
            p_shift = get_p_shift(site)
            if importance[site] / cutoff >= 1 or allsite:
                plt.text(x, y + p_shift, site, **text_param, va='center')

        exst_name = wrap_str(state_ls)
        title = 'exciton state {}'.format(exst_name)

        set_ax_limit(ax, site_positions)

        plt.text(*text_pos(ax), title, size=14)
        plt.text(*text_pos(ax, True),
                 'cutoff = {:.2f}'.format(cutoff), size=12, ha='right')

        save_fig(system.get_output_name(plot_name + 'Excitons_{}'.format(exst_name)), output=save_to_file)
