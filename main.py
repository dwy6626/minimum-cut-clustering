# the main script of the program


# ============================================================
# SYSTEM CHECK
# ============================================================


from obj import *
import aux


# ============================================================
# INPUT
# ============================================================

from sys import argv
import alg
import plot

Project = project.Project(argv[1:])
opt_set, cluster_opt, cmd_opt = Project.setting._Setting__run_opt

print('Job started.\n')
aux.print_1_line_stars()

# ============================================================
# OUTPUT: NETWORK
# ============================================================

# check if E1 > E2 but flow back
if 'k' in cmd_opt:
    alg.check_rate(Project.reference_system)
    aux.print_1_line_stars()

judge_set = {'n', len(Project.reference_system)}

for system in Project:
    # option -l print latex code (Hamiltonian, rate matrix)
    if 'l' in cmd_opt:
        alg.print_latex_matrix(system, decimal=aux.pass_int(Project.setting.Setting['decimal']))

    # option -I: site-exciton corresponding diagram
    if system.has_hamiltonian() and 'n' in opt_set['I']:
        plot.plot_tf(system)

    # option -e: plot exciton population on each site
    if judge_set.intersection(opt_set['e']):
        plot.plot_exst(system, allsite='allsite' in Project.setting.KeyWords)

    # option -b: find bottleneck index
    if 'b' in cmd_opt:
        alg.bottleneck_rate(system)
        aux.print_1_line_stars()

    # option -d: draw network
    if judge_set.intersection(opt_set['d']):
        print('Start plotting original network')
        aux.nx_aux.nx_graph_draw(
            system.get_graph(), system, system.get_plot_name() + 'Rate', rc_order=system.ExcitonName
        )
        aux.print_1_line_stars()

    # option -F: FFA flow decomposition
    if judge_set.intersection(opt_set['F']):
        alg.flow_analysis(system, draw=True)
        aux.print_1_line_stars()

    # option -r: save rate matrix
    if judge_set.intersection(opt_set['r']):
        alg.save_rate(
            system.RateConstantMatrix,
            save_name=system.get_plot_name(),
            energies=system.ExcitonEnergies,
        )
        aux.print_1_line_stars()

    # option -M: dynamics!
    # option -p: time-integrated flux
    # option -a: population dynamics animation
    dynamics_opt = judge_set.intersection(opt_set['M']), judge_set.intersection(opt_set['p']), 'a' in cmd_opt

    if any(dynamics_opt):
        system.get_dynamics(
            pyplot_output=dynamics_opt[0],
            flux=dynamics_opt[1],
            create_pop_animation=dynamics_opt[2]
        )
        aux.print_1_line_stars()

    # ============================================================
    # OUTPUT: COARSE GRAINED MODELS
    # ============================================================

    # manual clustering
    if Project.setting.Setting.get('map', False):
        alg.input_map_clustering(system)
        aux.print_1_line_stars()

    # without minimum-cut Tree
    for opt in cluster_opt:
        if opt == 'm':
            print('Method k: k-clustering')
            alg.k_clustering(system)

        # cut-off clustering
        elif opt == 'd':
            print('Method d: cut source-to-target rates')
            alg.cut_off_method(system, 4)

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

        aux.print_1_line_stars()

    # construct minimum-cut tree
    if 't' in cmd_opt:
        system.get_tree().draw()

    for opt in cluster_opt:
        if opt == 's':
            print('Method s: simple cut-off method')
            alg.simple_cut(system, 0)

        elif opt == 'r':
            print('Method r: simple ratio cut-off method')
            alg.simple_cut(system, 1)

        elif opt == 'b':
            print('Method b: bottom-up clustering method')
            alg.bottom_up_clx(system)

        # dimsplendid used this alg:
        elif opt == 't':
            print('Method t: top-down clustering method (target first)')
            alg.ascending_cut(system, 1)

        aux.print_1_line_stars()


# ============================================================
# SUMMARY
# ============================================================


Project.output_cluster_results(opt_set, 'l' in cmd_opt, 'c' in cluster_opt)
Project.save()
