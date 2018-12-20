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
            system.get_graph(), system, system.get_plot_name() + 'Rate', rc_order=system.ExcitonName
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
    # option -p: time-integrated flux
    # option -a: population dynamics animation
    dynamics_opt = judge_set.intersection(opt_set['M']), judge_set.intersection(opt_set['p']), 'a' in cmd_opt

    if any(dynamics_opt):
        system.get_dynamics(
            pyplot_output=dynamics_opt[0],
            flux=dynamics_opt[1],
            create_pop_animation=dynamics_opt[2]
        )
        lib.print_1_line_stars()

    # ============================================================
    # OUTPUT: COARSE GRAINED MODELS
    # ============================================================

    # manual clustering
    if Setting.get('map', False):
        alg.input_map_clustering(system)
        lib.print_1_line_stars()

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
            alg.simple_cut(system, 1)

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


Project.output_cluster_results(opt_set, 'l' in cmd_opt, 'c' in cluster_opt)
Project.save()
