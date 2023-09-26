from functools import partial
from pathlib import Path
import sys
sys.path.append('..')

# from sklearn.metrics import normalized_mutual_info_score, silhouette_score, davies_bouldin_score, adjusted_rand_score
# from sklearn.neighbors._dist_metrics import DistanceMetric

from src import cost_functions, data_types, plotting
from src.loading import load_GMM, make_mindsets
from src.cut_finding import a_slice
from src import utils
from src.plotting import plot_cuts_in_one

import numpy as np

from src.tree_tangles import ContractedTangleTree, tangle_computation, \
    compute_soft_predictions_children  # , mut_props_per_terminal_node, get_terminal_node_properties
from src.utils import compute_hard_predictions, compute_mindset_prediciton, merge_doubles

import simulate_with_demography
import simulate_with_demography_diploid
import benchmark_data
import admixture_plot
from src import outsourced_cost_computation
import FST
import pickle
import warnings
import time
import psutil
from scipy.sparse import csr_matrix


"""
Simple script for exemplary use of the tangle framework.
The execution is divided in the following steps

    1. Load datasets
    2. Find the cuts and compute the costs
    3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
    4. Postprocess in soft and hard clustering
"""


def tangles_in_pop_gen(sim_data, rho, theta, agreement, seed, pop_membership,
                       data_generation_mode, cost_precomputed=False,
                       output_directory='', plot=True,
                       plot_ADMIXTURE=False,
                       ADMIXTURE_filename = ""):
    print("started")
    xs = np.transpose(sim_data.G[0])    # diploid genotype matrix
    if data_generation_mode == 'out_of_africa':
        mutations_in_sim = sim_data.mutation_id
    else:
        mutations_in_sim = np.arange(xs.shape[1])
    # print("mutations before deletion:", len(mutations_in_sim), mutations_in_sim)
    # print("number of mutations before zero column deletion:", xs.shape[1])
    # num_zero_mut = 0
    # num_n_mut = 0
    # columns_to_delete_0 = []
    # columns_to_delete_n = []
    # for m in range(0, xs.shape[1]):
    #     if np.sum(xs[:, m]) == 0:
    #         columns_to_delete_0.append(m)
    #         num_zero_mut = num_zero_mut + 1
    # xs = np.delete(xs, columns_to_delete_0, axis=1)
    # mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_0)
    # for m in range(0, xs.shape[1]):
    #     if np.all(xs[:, m] > 0):
    #         columns_to_delete_n.append(m)
    #         num_n_mut = num_n_mut + 1
    # xs = np.delete(xs, columns_to_delete_n, axis=1)
    # mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_n)
    # print("mutations after deletion:", len(mutations_in_sim), mutations_in_sim)
    # print("num mutations deleted (mutations carried by no indv.):", num_zero_mut)
    # print("num mutations deleted (mutations carried by all indv.):", num_n_mut)
    # count_larger_2 = np.count_nonzero(xs > 4)
    # print("count larger 2:", count_larger_2)
    n = xs.shape[0]                     # diploid number of individuals
    nb_mut = xs.shape[1]
    print("n diploid:", n)
    print("number of mutations after mutation deletion:", nb_mut)

    #mut_pos = sim_data.ts[0].tables.sites.position

    # if (n_haploid % 2) == 1:
    #     warnings.warn(
    #         'sample size must be even as we consider diploid individuals!',
    #         stacklevel=1)
    #
    # # for diploid individuals, always sum two rows of genotype matrix:
    # n = n_haploid // 2   # number of diploid individuals (as integer)
    # print("n diploid:", n)
    # # array with indices of rows to sum
    # row_indices = np.arange(n) * 2
    # # sum rows and create new numpy ndarray
    # xs = xs_haploid[row_indices] + xs_haploid[row_indices + 1]
    # print("xs:\n", xs)

    #print("mut pos:", mut_pos)
    #print("G:", xs)
    data = data_types.Data(xs=xs)

    # calculate your bipartitions we use the binary questions/features directly as bipartitions
    # print("\tGenerating set of bipartitions", flush=True)
    # bipartitions = data_types.Cuts(values=(data.xs == True).T,
    #                                names=np.array(list(range(0, data.xs.shape[1]))))

    bipartitions = data_types.Cuts(values=(data.xs > 0).T,
                                   names=np.array(list(range(0, data.xs.shape[1]))))

    # Aktuellen Speicherverbrauch ausgeben
    print(f"Current memory usage 1: {psutil.virtual_memory().percent}%")

    print("\tFound {} bipartitions".format(len(bipartitions.values)), flush=True)
    print("\tCalculating costs if bipartitions", flush=True)
    #bipartitions = csr_matrix(bipartitions.values, dtype=bool)

    # Speed up bei paarweiser Kostenfuntkinon
    # bipartitions = utils.precompute_cost_and_order_cuts(bipartitions,
    #                                                     partial(cost_functions.normalized_mean_distances,
    #                                                             cost_functions.all_pairs_manhattan_distance(xs))
    #                                                     )

    cost = "FST_fast"  # FST_expected FST_observed  HWE_divergence
    # mean_manhattan_distance HWE_FST_exp FST_Wikipedia FST_wikipedia_fast
    # saved_bipartitions_filename = (str(data_generation_mode) + "_n_" + str(n) + "_n_"+str(cost))

    saved_bipartitions_filename = (str("sim") + "_n_" + str(n) + "_n_" + str(cost))

    start = time.time()
    print("time started")
    if cost_precomputed == False:
        #print("Precompute costs of bipartitions.")
        bipartitions = outsourced_cost_computation.compute_cost_and_order_cuts(
            bipartitions,
                                                         partial(
                                                             cost_functions.FST_expected_fast,
                                                             data.xs, None))

        with open('../tangles_in_pop_gen/data/saved_costs/' + str(saved_bipartitions_filename),
                   'wb') as handle:
            pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)


    else:
        print("Load costs of bipartitions.")
        with open('../tangles_in_pop_gen/data/saved_costs/' + str(saved_bipartitions_filename), 'rb') as handle:
            bipartitions = pickle.load(handle)
    end = time.time()
    print("time needed:", end - start)


    # bipartitions = utils.compute_cost_and_order_cuts(bipartitions,
    #                                                  partial(
    #                                                         cost_functions.mean_manhattan_distance_weighted_mut_pos,
    #                                                      data.xs, None, mut_pos,
    #                                                      lambda x: 0.07))

    np.save("data/mutation_labels", bipartitions.names)
    loaded = np.load("data/mutation_labels.npy", allow_pickle=True)

    # merge duplicate bipartitions
    print("Merging doublicate mutations.")
    bipartitions = merge_doubles(bipartitions)

    print("Tangle algorithm", flush=True)
    # calculate the tangle search tree
    print("\tBuilding the tangle search tree", flush=True)
    tangles_tree = tangle_computation(cuts=bipartitions,
                                      agreement=agreement,
                                      verbose=3)#,  # print everything
                                      # max_clusters=3)



    #plot_cuts_in_one(data, bipartitions, Path('tmp'))

    typ_genome_per_pop = tangles_tree._get_path_to_leaf(tangles_tree,
                                                        tangles_tree.root, [], n)
    print(typ_genome_per_pop)

    print("Built tree has {} leaves".format(len(tangles_tree.maximals)), flush=True)
    # postprocess tree
    print("Postprocessing the tree.", flush=True)
    # contract to binary tree
    print("\tContracting to binary tree", flush=True)
    contracted_tree = ContractedTangleTree(tangles_tree)

    # prune short paths
    # print("\tPruning short paths (length at most 1)", flush=True)
    contracted_tree.prune(0)

    # calculate
    print("\tcalculating set of characterizing bipartitions", flush=True)
    contracted_tree.calculate_setP()
    # print("caracterizing cuts:", contracted_tree.root)
    # print("test print nodes:", tangles_tree.root.right_child.right_child.right_child)
    # compute soft predictions
    # assign weight/ importance to bipartitions
    weight = np.exp(-utils.normalize(bipartitions.costs)) * np.array([name.count("'") + 1 for name in bipartitions.names])

    # propagate down the tree
    print("Calculating soft predictions", flush=True)
    compute_soft_predictions_children(node=contracted_tree.root,
                                      cuts=bipartitions,
                                      weight=weight,
                                      verbose=3)

    contracted_tree.processed_soft_prediction = True

    print("Calculating hard predictions", flush=True)
    ys_predicted, _ = utils.compute_hard_predictions(contracted_tree, cuts=bipartitions)
    print(ys_predicted)

    if plot:
        print("Plotting the data.", flush=True)
        output_directory.mkdir(parents=True, exist_ok=True)
        ## plot the tree
        # filename1 = "tree_n_" + str(n) + "_rho_" + str(rho) + "_theta_" + str(
        #     theta) + "_seed_" + str(
        #     seed) + "_agreement_" + str(agreement) + "_noise_" + str(noise) + ".svg"
        # filename2 = "contracted_n_" + str(n) + "_rho_" + str(rho) + "_theta_" + str(
        #     theta) + "_seed_" + str(
        #     seed) + "_agreement_" + str(agreement) + "_noise_" + str(noise) + ".svg"
        # tangles_tree.plot_tree(path=output_directory / filename1)
        # # plot contracted tree
        # contracted_tree.plot_tree(path=output_directory / filename2)
        #
        # # plot tree summary
        # tangles_tree.print_tangles_tree_summary_hard_predictions(nb_mut,
        #                                                         bipartitions.names,
        #                                                         ys_predicted)
        #
        # contracted_tree.print_summary(contracted_tree.root)

        # plot soft predictions
        # plotting.plot_soft_predictions(data=data,
        #                                contracted_tree=contracted_tree,
        #                                eq_cuts=bipartitions.equations,
        #                                path=output_directory / 'soft_clustering')

        matrices, char_cuts = contracted_tree.to_matrix()
        print("char cuts:", char_cuts)

        pop_splits = [[0, 400, 800], [0, 700, 800], [0, 200, 800], [0, 600, 700, 800],
                      [0, 200, 300, 800],
                      [0, 400, 500, 800], [0, 100, 200, 800]]

        #FST_sim = FST.FST_values_sim(xs, mutations=mutations_in_sim,
        # pop_splits=pop_splits)
        #FST_tangles = FST.FST_values_tangles(xs, bipartitions=bipartitions,
        #                                        characterizing_cuts=char_cuts)
        #print("FST tangles:", FST_tangles)
        #print("FST_sim:", FST_sim)

        #FST.plot_FST(xs, mutations_in_sim, bipartitions, char_cuts, pop_splits)

        #print(matrices)
        print("matrices done.")
        print("matrices:", matrices)

        admixture_plot.admixture_like_plot(matrices, pop_membership, agreement, seed,
                                           data_generation_mode,
                                           plot_ADMIXTURE=plot_ADMIXTURE,
                                           ADMIXTURE_file_name=ADMIXTURE_filename,
                                           cost_fct=cost)

        print("admixture like plot done.")
        # with open('saved_soft_matrices.pkl', 'wb') as f:
        #     pickle.dump(matrices, f)

if __name__ == '__main__':
    n = 40 # 800 #40      #15     # anzahl individuen
    # rho=int for constant theta in rep simulations, rho='rand' for random theta in (0,100) in every simulation:
    rho = 55# 100 55 0.5   #1      # recombination
    # theta=int for constant theta in rep simulations, theta='rand' for random theta in (0,100) in every simulation:
    theta = 55  # 100 55      # mutationsrate
    agreement = 5
    seed = 42 #42   #17
    noise = 0
    data_already_simulated = True # True or False, states if data object should be
    # simulated or loaded
    data_generation_mode = 'readVCF' # readVCF  out_of_africa sim

    # new parameters that need to be set to load/simulate appropriate data set
    rep = 1  # number of repetitions during simulation
    save_G = True  # set True to save genotype matrix during simulation, False otherwise
    print_ts = False  # (set small for large n) set True if ts should be printed during
    # simulation, this is only possible if rep==1. For large data sets, this step slows down the program noticeably.
    save_ts = True  # set True to save the tree sequence during simulation, False otherwise
    filepath = "data/with_demography/"  # filepath to the folder where the data is to be
    # saved/loaded.

    if data_generation_mode == 'out_of_africa':
        rho = -1
        theta = -1
        data = benchmark_data.SimulateOutOfAfrica(
            n, seed, save_G=save_G, print_ts=print_ts, save_ts=save_ts,
            filepath=filepath)
        if data_already_simulated == False:
            data.sim_data()
            print("Data has been simulated.")
        else:
            data.load_data()
            print("Data has been loaded.")

    elif data_generation_mode == 'readVCF':
        rho = -1
        theta = -1
        data = benchmark_data.ReadVCF('n_40_rep_1_rho_55_theta_55_seed_42.vcf', #'gen0_chr22_train.vcf',
                                     'admixture/data/')
        data.load_data()

    else:
        ## This generates the data object and either simulates or loads the data sets
        data = simulate_with_demography_diploid.Simulated_Data_With_Demography_Diploid(n,
                                                                                 rep, theta, rho, seed,
                                                                   save_G=save_G,
                                                                   print_ts=print_ts,
                                                                   save_ts=save_ts,
                                                                   filepath=filepath)
        if data_already_simulated == False:
            data.sim_data()
            print("Data has been simulated.")
        else:
            data.load_data()
            print("Data has been loaded.")

    cost_precomputed = True
    plot_ADMIXTURE = True
    ADMIXTURE_filename = data.admixture_filename

    output_directory = Path('output_tangles_in_pop_gen')
    plot = True

    # indv_pop_diploid_indices = np.arange(n//2) * 2
    # pop_membership = data.indv_pop[indv_pop_diploid_indices]
    # print("pop membership:", pop_membership)

    tangles_in_pop_gen(data, rho, theta, agreement, seed, data.indv_pop,
                       data_generation_mode, cost_precomputed=cost_precomputed,
                       output_directory=output_directory, plot=True,
                       plot_ADMIXTURE=plot_ADMIXTURE,
                       ADMIXTURE_filename=ADMIXTURE_filename)

    print("all done.")
