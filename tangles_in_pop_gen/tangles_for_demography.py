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
import networkx as nx

from src.tree_tangles import ContractedTangleTree, tangle_computation, \
    compute_soft_predictions_children  # , mut_props_per_terminal_node, get_terminal_node_properties
from src.utils import compute_hard_predictions, compute_mindset_prediciton, merge_doubles

import simulate_with_demography
import simulate_with_demography_diploid
import benchmark_data
import admixture_plot
import compute_kNN
from src import outsourced_cost_computation
import FST
import pickle
import warnings
import time
import psutil
from scipy.sparse import csr_matrix

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
                       data_generation_mode, cost_fct_name, cost_precomputed=False,
                       output_directory='', plot=True,
                       plot_ADMIXTURE=False,
                       ADMIXTURE_filename = ""):
    xs = np.transpose(sim_data.G[0])  # diploid genotype matrix
    if data_generation_mode == 'out_of_africa':
        mutations_in_sim = sim_data.mutation_id
    else:
        mutations_in_sim = np.arange(xs.shape[1])

    if data_generation_mode == 'readVCF':
        print("resort data via populations.")
        panel_file = ('/home/klara/ML_in_pop_gen_in_process/tangles_in_pop_gen'
                      '/tangles_in_pop_gen/admixture/data/integrated_call_samples_v3.20130502.ALL.panel')
        panel_df = pd.read_csv(panel_file, delimiter='\t')
        sample_pop = panel_df['pop']
        sample_super_pop = panel_df['super_pop']
        pop_array = np.array(sample_pop)
        super_pop_array = np.array(sample_super_pop)
        sort_criteria = super_pop_array + pop_array
        sorted_indices = np.argsort(sort_criteria)
        pop_membership = super_pop_array[sorted_indices]
        print("pop_membership after resorting:", pop_membership)
        xs = xs[sorted_indices]

    # data preprocessing:
    print("mutations before deletion:", len(mutations_in_sim))  # ), mutations_in_sim)
    print("number of mutations before zero column deletion:", xs.shape[1])
    num_zero_mut = 0
    num_n_mut = 0
    num_multiallelic = 0
    num_low_freq = 0
    columns_to_delete_0 = []
    columns_to_delete_n = []
    columns_to_delete_multiallelic = []
    columns_to_delete_low_freq = []
    for m in range(0, xs.shape[1]):
        if np.sum(xs[:, m]) == 0:
            columns_to_delete_0.append(m)
            num_zero_mut = num_zero_mut + 1
    xs = np.delete(xs, columns_to_delete_0, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_0)

    for m in range(0, xs.shape[1]):
        if np.all(xs[:, m] > 0):
            columns_to_delete_n.append(m)
            num_n_mut = num_n_mut + 1
    xs = np.delete(xs, columns_to_delete_n, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_n)

    for m in range(0, xs.shape[1]):
        if np.any(xs[:, m] > 2):
            columns_to_delete_multiallelic.append(m)
            num_multiallelic = num_multiallelic + 1
    xs = np.delete(xs, columns_to_delete_multiallelic, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_multiallelic)

    for m in range(0, xs.shape[1]):
        if np.count_nonzero(xs[:, m]) == 0:
            columns_to_delete_low_freq.append(m)
            num_low_freq = num_low_freq + 1
    xs = np.delete(xs, columns_to_delete_low_freq, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_low_freq)

    print("mutations after deletion:", len(mutations_in_sim))  # , mutations_in_sim)
    print("num mutations deleted (mutations carried by no indv.):", num_zero_mut)
    print("num mutations deleted (mutations carried by all indv.):", num_n_mut)
    print("multiallelic sites deleted:", num_multiallelic)
    print("low freq. sites deleted:", num_low_freq)
    count_larger_2 = np.count_nonzero(xs > 2)
    print("count larger 2:", count_larger_2)

    # # subsampling sites
    # sample_size = 5000
    # np.random.seed(seed)
    # print("number of sites after subsampling:", sample_size)
    # random_indices = np.random.choice(xs.shape[1], sample_size, replace=False)
    # print("random indices:", random_indices)
    # # downsize xs to selected sites
    # xs = xs[:, random_indices]

    n = xs.shape[0]  # diploid number of individuals
    nb_mut = xs.shape[1]
    print("n diploid:", n)
    print("number of mutations after mutation deletion and subsampling:", nb_mut)

    data = data_types.Data(xs=xs)


    ## kNN
    kNN_precomputed = False
    k = 20
    kNN_filename = (str(data_generation_mode) + "_n_" + str(n) + "_sites_" + str(
        nb_mut) + "_" + "_seed_" + str(seed) + "_k_" + str(k))
    if kNN_precomputed == False:
        kNN = compute_kNN.KNearestNeighbours(xs, k, filename=kNN_filename,
                                             filepath="data/saved_kNN/")
        kNN.compute_kNN()
    else:
        kNN = compute_kNN.KNearestNeighbours(xs, k, filename=kNN_filename,
                                             filepath="data/saved_kNN/")
        kNN.load_kNN()
    print("kNN.kNN", kNN.kNN)

    # pickle kNN-Matrix for cost function:
    with open("data/saved_kNN/kNN", 'wb') as outp:  # overwrites existing file.
        pickle.dump(kNN, outp, pickle.HIGHEST_PROTOCOL)

    # check if kNN Graph is connected:
    G = nx.from_numpy_array(kNN.kNN)
    print("knn Graph is connected:", nx.is_connected(G))

    check_kNN_within_pop = True
    if check_kNN_within_pop == True:
        # check if nearest neighbors within populations:
        pop_sizes = np.bincount(pop_membership)
        #pop_boundaries = [0]
        #pop_boundaries.append(np.cumsum(pop_sizes))
        pop_start = 0
        kNN_outside_pop_count = []
        indv_with_neighbours_outside_pop = []
        for i in range(0, len(pop_sizes)):
            c = 0
            for j in range(0, pop_sizes[i]):
                # get indices of neighbors
                neighbors = np.where(kNN.kNN[pop_start + j] == 1)[0]
                # check if neighbors lie outside of population
                neighbors_only_in_pop = True
                for idx in neighbors:
                    if idx < pop_start or idx >= pop_start + pop_sizes[i]:
                        indv_with_neighbours_outside_pop.append(pop_start + j)
                        if neighbors_only_in_pop == True:
                            c += 1
                            neighbors_only_in_pop = False
            kNN_outside_pop_count.append(c)
            pop_start += pop_sizes[i]

        print("per pop number of indv with neighbors outside of own pop:",
              kNN_outside_pop_count)
        print("indv with nearest neighbor outside of own pop:",
              np.unique(indv_with_neighbours_outside_pop))



    # calculate bipartitions
    print("\tGenerating set of bipartitions", flush=True)
    bipartitions = data_types.Cuts(values=(data.xs > 0).T,
                                   names=np.array(list(range(0, data.xs.shape[1]))))

    # Current memory usage
    print(f"Current memory usage 1: {psutil.virtual_memory().percent}%")

    print("\tFound {} bipartitions".format(len(bipartitions.values)), flush=True)
    print("\tCalculating costs if bipartitions", flush=True)
    #bipartitions = csr_matrix(bipartitions.values, dtype=bool)

    # Speed up bei paarweiser Kostenfuntkinon
    # bipartitions = utils.precompute_cost_and_order_cuts(bipartitions,
    #                                                     partial(cost_functions.normalized_mean_distances,
    #                                                             cost_functions.all_pairs_manhattan_distance(xs))
    #                                                     )

    cost_function = getattr(cost_functions, cost_fct_name)
    saved_costs_filename = (str(data_generation_mode) + "_n_" + str(n) +
                                   "_sites_" +
                                   str(nb_mut) + "_" +
                                   str(cost_fct_name) + "_seed_" +
                                   str(
                                       seed))

    start = time.time()
    print("time started")
    if cost_precomputed == False:
        # print("Precompute costs of bipartitions.")
        bipartitions = outsourced_cost_computation.compute_cost_and_order_cuts(
            bipartitions,
            partial(
                cost_function,
                data.xs, None))

        with open('../tangles_in_pop_gen/data/saved_costs/' + str(
                saved_costs_filename),
                  'wb') as handle:
            pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)


    else:
        print("Load costs of bipartitions.")
        with open('../tangles_in_pop_gen/data/saved_costs/' + str(
                saved_costs_filename), 'rb') as handle:
            bipartitions = pickle.load(handle)
    end = time.time()
    print("time needed:", end - start)

    print("bipartitions.names:", bipartitions.names)
    print("type(bipartitions.names):", type(bipartitions.names))
    print("bipartitions.costs:", bipartitions.costs)
    print("type(bipartitions.costs):", type(bipartitions.costs))

    # plot cost of bipartitions vs mutation frequency:
    mut_freq = np.sum(bipartitions.values, axis=1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(bipartitions.costs, mut_freq, alpha=0.5, s=5)
    ax.set_xlabel('costs')
    ax.set_ylabel('mutation frequency')
    with PdfPages('plots/cost_vs_mut_freq_sites_' + str(nb_mut) + "_seed_" + str(
            seed) + "_seed_" + str(cost_fct_name) + '.pdf') as pdf:
        pdf.savefig()
    plt.show()


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

    print("bipartitions.names:", bipartitions.names)
    print("type(bipartitions.names):", type(bipartitions.names))
    print("bipartitions.costs:", bipartitions.costs)
    print("type(bipartitions.costs):", type(bipartitions.costs))

    print("Tangle algorithm", flush=True)
    # calculate the tangle search tree
    print("\tBuilding the tangle search tree", flush=True)
    start_tangle_tree = time.time()
    tangles_tree = tangle_computation(cuts=bipartitions,
                                      agreement=agreement,
                                      verbose=3,
                                      prune_first_path=False)#,  # print everything
                                      # max_clusters=3)

    if tangles_tree.__class__ == np.float64:
        print("I am in if statement prune_first_path.")
        bip_idx = np.where(bipartitions.costs <= tangles_tree)[0]
        bipartitions = bipartitions[bip_idx]
        tangles_tree = tangle_computation(cuts=bipartitions,
                                          agreement=agreement,
                                          verbose=3)

    end_tangle_tree = time.time()
    print("tangle tree computation completed in ", end_tangle_tree -
          start_tangle_tree, " sec.")

    #plot_cuts_in_one(data, bipartitions, Path('tmp'))

    #typ_genome_per_pop = tangles_tree._get_path_to_leaf(tangles_tree,
    #                                                    tangles_tree.root, [], n)
    #print(typ_genome_per_pop)

    print("Built tree has {} leaves".format(len(tangles_tree.maximals)), flush=True)
    # postprocess tree
    print("Postprocessing the tree.", flush=True)
    # contract to binary tree
    print("\tContracting to binary tree", flush=True)
    contracted_tree = ContractedTangleTree(tangles_tree)
    contracted_tree.plot_tree("plots/tree_before_pruning")


    # prune short paths
    # print("\tPruning short paths (length at most 1)", flush=True)
    contracted_tree.prune(0)

    contracted_tree.plot_tree("plots/tree_after_pruning")

    # calculate
    print("\tcalculating set of characterizing bipartitions", flush=True)
    contracted_tree.calculate_setP()
    # print("caracterizing cuts:", contracted_tree.root)
    # print("test print nodes:", tangles_tree.root.right_child.right_child.right_child)
    # compute soft predictions
    # assign weight/ importance to bipartitions
    #weight = np.exp(-utils.normalize(bipartitions.costs)) * np.array([name.count(",
    # ") + 1 for name in bipartitions.names])
    weight = (1/bipartitions.costs)*np.sum(bipartitions.values, axis=1)*np.array([name.count(",") + 1 for name in bipartitions.names])
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
                                           data_generation_mode, sorting_level="all",
                                           plot_ADMIXTURE=plot_ADMIXTURE,
                                           ADMIXTURE_file_name=ADMIXTURE_filename,
                                           cost_fct=cost_fct_name)

        print("admixture like plot done.")
        # with open('saved_soft_matrices.pkl', 'wb') as f:
        #     pickle.dump(matrices, f)

if __name__ == '__main__':
    n = 800  # 800 #40      #15     # anzahl individuen
    # rho=int for constant theta in rep simulations, rho='rand' for random theta in (0,100) in every simulation:
    rho = 100  # 100 55 0.5   #1      # recombination
    # theta=int for constant theta in rep simulations, theta='rand' for random theta in (0,100) in every simulation:
    theta = 100  # 100 55      # mutationsrate
    agreement = 35
    seed = 42  # 42   #17
    noise = 0
    data_already_simulated = False  # True or False, states if data object should be
    # simulated or loaded
    data_generation_mode = 'sim'  # readVCF  out_of_africa sim
    cost_fct_name = "k_nearest_neighbours"  # FST, HWE or FST_normalized
    cost_precomputed = False
    plot_ADMIXTURE = False

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
        data = benchmark_data.ReadVCF(n, '1000G_chr22.vcf',  # 'gen0_chr22_train.vcf',
                                      # 1000G_chr22.vcf
                                      'admixture/data/')
        # data.read_vcf()
        data.load_vcf()
        print("pop membership:", data.indv_pop)
        print("len pop membership:", len(data.indv_pop))

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

    ADMIXTURE_filename = data.admixture_filename
    output_directory = Path('output_tangles_in_pop_gen')
    plot = True

    # indv_pop_diploid_indices = np.arange(n//2) * 2
    # pop_membership = data.indv_pop[indv_pop_diploid_indices]
    # print("pop membership:", pop_membership)

    tangles_in_pop_gen(data, rho, theta, agreement, seed, data.indv_pop,
                       data_generation_mode, cost_fct_name,
                       cost_precomputed=cost_precomputed,
                       output_directory=output_directory, plot=True,
                       plot_ADMIXTURE=plot_ADMIXTURE,
                       ADMIXTURE_filename=ADMIXTURE_filename)

    print("all done.")
