# the main script of the program


# ============================================================
# SYSTEM CHECK
# ============================================================


from obj import *
import lib


# ============================================================
# INPUT
# ============================================================

from sys import argv
import alg
import plot

Setting = setting.Setting()
opt_set, cluster_opt, cmd_opt = Setting.receive_arguments(argv[1:])

Project = project.Project(Setting)
input_fname = Setting.InputFileName
if not input_fname:
    print('Warning: please specify the input file position\n')
    lib.help_message()

Project.build_reference_system(
    input_fname,
    is_rate_matrix=input_fname.split('.')[-1].lower() != 'h',
    as_file_path=True,
    additional_hamiltonian_load=Setting.get('H')
)

print('Job started.\n')
lib.print_1_line_stars()

# ============================================================
# OUTPUT: NETWORK
# ============================================================

# check if E1 > E2 but flow back
if 'k' in cmd_opt:
    alg.check_rate(Project.get_reference_system())
    lib.print_1_line_stars()

judge_set = {'n', len(Project.get_reference_system())}

for system in Project:
    # option -l print latex code (Hamiltonian, rate matrix)
    if 'l' in cmd_opt:
        alg.print_latex_matrix(system, decimal=lib.pass_int(Setting['decimal']))

    # option -I: site-exciton corresponding diagram
    if system.has_hamiltonian() and 'n' in opt_set['I']:
        plot.plot_tf(system)

    # option -e: plot exciton population on each site
    if judge_set.intersection(opt_set['e']):
        plot.plot_exst(system, allsite='allsite' in Project.setting)

    # option -b: find bottleneck index
    if 'b' in cmd_opt:
        alg.bottleneck_rate(system)
        lib.print_1_line_stars()

    # option -d: draw network
    if judge_set.intersection(opt_set['d']):
        print('Start plotting original network')
        lib.nx_aux.nx_graph_draw(
            system.get_graph(), Project.config.get_graphviz_dot_path(), Setting,
            system.get_plot_name() + 'Rate', rc_order=system.ExcitonName
        )
        lib.print_1_line_stars()

    # option -F: FFA flow decomposition
    if judge_set.intersection(opt_set['F']):
        alg.flow_analysis(system, draw=True)
        lib.print_1_line_stars()

    # option -r: save rate matrix
    if judge_set.intersection(opt_set['r']):
        alg.save_rate(
            system.RateConstantMatrix,
            save_name=system.get_plot_name(),
            energies=system.ExcitonEnergies,
        )
        lib.print_1_line_stars()

    # option -M: dynamics!
    if judge_set.intersection(opt_set['M']):
        system.plot_dynamics(save_to_file=True)
        lib.print_1_line_stars()

    # option -p: time-integrated flux
    if judge_set.intersection(opt_set['p']):
        integrated_flux_matrix = system.get_integrated_flux(spline_size=lib.pass_int(Setting['spline']),
                                                            save_to_file=True)
        alg.plot_integrate_flux(
            integrated_flux_matrix, system.ExcitonName, system.ExcitonEnergies, system.get_plot_name(),
            Setting, Project.config.get_graphviz_dot_path()
        )
        lib.print_1_line_stars()

    # option -a: population dynamics animation
    if 'a' in cmd_opt:
        system.animate_dynamics(ps1_special_option='PSI' in Setting.InputFileName,
                                dpi=lib.pass_int(Setting['dpi']),
                                allsite='allsite' in Setting)
        lib.print_1_line_stars()

    # ============================================================
    # OUTPUT: COARSE GRAINED MODELS
    # ============================================================

    # manual clustering
    if Setting.get('map', False):
        alg.input_map_clustering(system, system.back_ptr.setting['map'])
        lib.print_1_line_stars()

    # without minimum-cut Tree
    for opt in cluster_opt:
        if opt == 'm':
            print('Method k: k-clustering')
            alg.k_clustering(system)

        # cut-off clustering
        elif opt == 'd':
            print('Method d: cut source-to-target rates')
            alg.cut_off_method(system)

        elif opt == 'g':
            print('Method g: cut geometric mean rates')
            alg.cut_off_method(system, 2)

        elif opt == 'f':
            print('Method f: cut RMS rates')
            alg.cut_off_method(system, 3)

        elif opt == 'e':
            print('Method e: cut maximum rates')
            alg.cut_off_method(system, 1)

        elif opt == 'h' and system.has_hamiltonian():
            print('Method h: cut coupling constants ')
            alg.cut_off_method(system, 5)

        elif opt == 'k':
            print('Method k: k-means (minimize 1/r ** 2, undigraph)')
            alg.k_means_like(system)

        else:
            continue

        lib.print_1_line_stars()

    # construct minimum-cut tree
    if 't' in cmd_opt:
        system.get_tree().draw()

    for opt in cluster_opt:
        if opt == 's':
            print('Method s: simple cut-off method')
            alg.simple_cut(system, 0)

        elif opt == 'r':
            print('Method r: simple ratio cut-off method')
            alg.simple_ratio_cut(system, 1)

        elif opt == 'b':
            print('Method b: bottom-up clustering method')
            alg.bottom_up_clx(system)

        # dimsplendid used this alg:
        elif opt == 't':
            print('Method t: top-down clustering method (target first)')
            alg.ascending_cut(system, 1)

        lib.print_1_line_stars()


# ============================================================
# SUMMARY
# ============================================================


if len(Project.data_frame) > 0:
    cost = 'c' in cluster_opt
    for h_id, system in enumerate(Project):
        df = Project.data_frame[h_id]
        for i, (m, n, cgm, _, _) in df.iterrows():
            judge_ls = (n, 'c')

            if system.has_hamiltonian() and opt_set['I'].intersection(judge_ls):
                plot.plot_tf(system, cgm)
                lib.print_1_line_stars()

            if opt_set['e'].intersection(judge_ls):
                plot.plot_exst(system, allsite='allsite' in Setting, clx_map=cgm)

            dot = opt_set['d'].intersection(judge_ls)
            ffa = opt_set['F'].intersection(judge_ls)
            dynamics = opt_set['M'].intersection(judge_ls)
            flux = opt_set['p'].intersection(judge_ls)
            rate = opt_set['r'].intersection(judge_ls)

            if any([dot, ffa, dynamics, flux, rate, cost]):
                # tuple: rate matrix, energies, name
                cluster_3_tuple = *system.get_cluster(cgm), system.get_plot_name(cgm)

                # graphviz / dot files
                if dot:
                    lib.nx_aux.nx_graph_draw(
                        lib.get_cluster_graph(cluster_3_tuple), dot_path=Project.config.get_graphviz_dot_path(),
                        setting=Setting, plot_name=cluster_3_tuple[2] + 'Rate', rc_order=list(cluster_3_tuple[0].keys())
                    )
                    lib.print_1_line_stars()

                if ffa:
                    alg.flow_analysis(system, cluster_3_tuple)
                    lib.print_1_line_stars()

                if rate:
                    if 'l' in cmd_opt:
                        alg.print_rate_matrix(cluster_3_tuple[0], lib.pass_int(Setting['decimal']))
                    alg.save_rate(*cluster_3_tuple)
                    lib.print_1_line_stars()

                # dynamics
                dynamics_opt = [dynamics, flux, cost]
                if any(dynamics_opt):
                    pop_seq, time_sequence, nodes = system.get_dynamics(cluster_3_tuple)
                    pop_full = system.get_comparison_to_full_dynamics(nodes)

                    # for label is too long: cluster X
                    pop_names = lib.wraps(nodes, maxlen=30)

                    if dynamics:
                        plot.plot_dyanmics(
                            pop_seq, time_sequence, pop_names,
                            plot_name=cluster_3_tuple[2], pop_seq2=pop_full,
                            y_max=lib.pass_float(Setting.get('ymax', '0.')),
                            x_max=lib.pass_float(Setting.get('xmax', '0.')),
                            legend='nolegend' not in Setting,
                            divide=lib.pass_int(Setting.get('divide', 100)),
                            save_to_file=True
                        )

                    if cost or flux:
                        pop_seq2, time_sequence2 = lib.spline_grid(pop_seq, time_sequence,
                                                                   lib.pass_int(Setting['spline']))

                        if flux:
                            integrated_flux_matrix = alg.get_integrated_flux(
                                pop_seq2, cluster_3_tuple[0], time_sequence2,
                                nodes=nodes,
                                norm=lib.pass_int(Setting.get('multiply', 1)),
                                plot_details='log' in Setting,
                                plot_name=cluster_3_tuple[2],
                                y_max=lib.pass_float(Setting.get('ymax', '0.')),
                                x_max=lib.pass_float(Setting.get('xmax', '0.')),
                                divide=lib.pass_int(Setting.get('divide', 100)),
                                legend='nolegend' not in Setting,
                                save_to_file=True
                            )
                            alg.plot_integrate_flux(
                                integrated_flux_matrix, nodes, cluster_3_tuple[1], cluster_3_tuple[2],
                                Setting, Project.config.get_graphviz_dot_path()
                            )

                        if cost:
                            # directly add the result to the end of data frame
                            df.iloc[i, df.columns.get_loc('PopDiff')] = system.get_population_difference(
                                cluster_3_tuple, pop_seq2, False
                            )

                    lib.print_1_line_stars()

    if cost:
        Project.plot_cost(save_to_file=True)

Project.save()
