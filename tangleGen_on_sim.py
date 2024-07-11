import sys
from functools import partial
from pathlib import Path
sys.path.append('..')
from src import cost_functions, data_types
from src import utils
import numpy as np
from src.tree_tangles import ContractedTangleTree, tangle_computation, \
    compute_soft_predictions_children_popgen
from src.utils import merge_doubles
import simulate_with_demography
import plot_soft_clustering
import compute_kNN
from src import outsourced_cost_computation
import pickle
import time
import reliability_factor
from sklearn.metrics import silhouette_score

"""
Simple script for use of the tangle framework on simulated data (simulation with 
well-differentiated populations). The execution is divided in the following steps

    1. Load/simulate the dataset
    2. Find the cuts and compute the costs
    3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
    4. Postprocess in soft and hard clustering
    5. plot soft clustering
"""


def tangles_in_pop_gen(sim_data, agreement, seed, k, pruning, pop_membership,
                       data_generation_mode, cost_fct_name, cost_precomputed=False,
                       output_directory='', plot=True, plot_ADMIXTURE=False,
                       ADMIXTURE_filename=""):
    # get genotype matrix xs and mutation idx
    xs = np.transpose(sim_data.G[0])  # diploid genotype matrix
    mutations_in_sim = np.arange(xs.shape[1])

    ## pre-processing of the data: deletion of multiallelicity sites and SNPs which
    # for all individuals either have no ancestral alleles at all or only ancestral alleles
    print("number of sites before deletion:", len(mutations_in_sim))
    num_zero_mut = 0
    num_n_mut = 0
    num_multiallelic = 0
    columns_to_delete_0 = []
    columns_to_delete_n = []
    columns_to_delete_multiallelic = []
    # delete SNPs for which all individuals are homozygous for the ancestral allele:
    for m in range(0, xs.shape[1]):
        if np.sum(xs[:, m]) == 0:
            columns_to_delete_0.append(m)
            num_zero_mut = num_zero_mut + 1
    xs = np.delete(xs, columns_to_delete_0, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_0)
    # delete SNPs where the ancestral allele is not carried by any individual:
    for m in range(0, xs.shape[1]):
        if np.all(xs[:, m] > 0):
            columns_to_delete_n.append(m)
            num_n_mut = num_n_mut + 1
    xs = np.delete(xs, columns_to_delete_n, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_n)
    # delete multi-allelic sites:
    for m in range(0, xs.shape[1]):
        if np.any(xs[:, m] > 2):
            columns_to_delete_multiallelic.append(m)
            num_multiallelic = num_multiallelic + 1
    xs = np.delete(xs, columns_to_delete_multiallelic, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_multiallelic)
    print("number of sites after deletion:", len(mutations_in_sim))

    # specify number of diploid individuals and number of sites (or mutations):
    n = xs.shape[0]
    nb_mut = xs.shape[1]
    data = data_types.Data(xs=xs)
    start_tangles_1 = time.time()  # start timekeeping

    ## compute kNN for cost computation:
    kNN_precomputed = False  # specify if kNN already pre-computed or not
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
    # pickle kNN-matrix for cost function to load:
    with open("data/saved_kNN/kNN", 'wb') as outp:  # overwrites existing file.
        pickle.dump(kNN, outp, pickle.HIGHEST_PROTOCOL)

    ## calculate bipartitions
    print("\tGenerating set of bipartitions", flush=True)
    bipartitions = data_types.Cuts(values=(data.xs > 0).T,
                                   names=np.array(list(range(0, data.xs.shape[1]))))
    # Current memory usage
    # print(f"Current memory usage 1: {psutil.virtual_memory().percent}%")
    print("\tFound {} bipartitions".format(len(bipartitions.values)), flush=True)
    print("\tCalculating costs if bipartitions", flush=True)

    ## compute costs for each cut:
    cost_function = getattr(cost_functions, cost_fct_name)
    saved_costs_filename = (
                str(data_generation_mode) + "_n_" + str(n) + "_sites_" + str(
            nb_mut) + "_" + str(cost_fct_name) + "_seed_" + str(seed))
    start = time.time()  # start time measurement
    if cost_precomputed == False:
        bipartitions = outsourced_cost_computation.compute_cost_and_order_cuts(
            bipartitions, partial(cost_function, data.xs, None))
        with open('data/saved_costs/' + str(saved_costs_filename), 'wb') as handle:
            pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Load costs of bipartitions.")
        with open('data/saved_costs/' + str(saved_costs_filename), 'rb') as handle:
            bipartitions = pickle.load(handle)
    end = time.time()
    print("time needed for cost computation:", end - start)
    print("bipartitions.names:", bipartitions.names)
    print("bipartitions.costs:", bipartitions.costs)

    ## merge duplicate bipartitions:
    print("Merging doublicate mutations.")
    bipartitions = merge_doubles(bipartitions)
    print("number of bipartitions after merging:", len(bipartitions.names))

    print("Tangle algorithm", flush=True)
    ## calculate the tangle search tree:
    print("\tBuilding the tangle search tree", flush=True)
    start_tangle_tree = time.time()
    tangles_tree = tangle_computation(cuts=bipartitions, agreement=agreement, verbose=3)
    end_tangle_tree = time.time()
    print("tangle tree computation completed in ", end_tangle_tree - start_tangle_tree,
          " sec.")
    print("Built tree has {} leaves".format(len(tangles_tree.maximals)), flush=True)
    # postprocess tree
    print("Postprocessing the tree.", flush=True)
    # contract to binary tree
    print("\tContracting to binary tree", flush=True)
    contracted_tree = ContractedTangleTree(tangles_tree)
    # contracted_tree.plot_tree("plots/tree_before_pruning")
    # calculate set of characterizing cuts:
    print("\tcalculating set of characterizing bipartitions", flush=True)
    contracted_tree.calculate_setP()
    # prune short paths:
    contracted_tree.prune(bipartitions, pruning)
    contracted_tree.calculate_setP()
    # assign weight/ importance to bipartitions. no weights used for soft clustering in
    # population genetics:
    weight = np.ones(len(bipartitions.names))
    # calculate the soft clustering, i.e. soft predictions:
    print("Calculating soft predictions", flush=True)
    compute_soft_predictions_children_popgen(node=contracted_tree.root,
                                             cuts=bipartitions, weight=weight,
                                             cuts_probs=reliability_factor.compute_reliability(
                                                 xs), verbose=3)
    contracted_tree.processed_soft_prediction = True
    print("Calculating hard predictions", flush=True)
    ys_predicted, _ = utils.compute_hard_predictions(contracted_tree, cuts=bipartitions)
    print(ys_predicted)
    if plot:
        print("Plotting the data.", flush=True)
        output_directory.mkdir(parents=True, exist_ok=True)
        # get matrix with hierarchical soft clustering along the tangles tree to plot
        # this inferred ancestry. char_cuts are the characteristic SNPs per split in
        # the tangles tree, posititions indicate the position of the split in the tree.
        matrices, char_cuts, positions = contracted_tree.to_matrix()
        print("char cuts:", char_cuts)

        # silhouette score
        # sihouette_score = silhouette_score(kNN.pairwise_distances, ys_predicted, metric="precomputed")
        # print("silhouette score:", sihouette_score, agreement, kNN.k)
        #
        # c_ij_precomputed = False
        # c_ij_filename = ("data/saved_kNN/c_ij_" + data_generation_mode +
        #                  "_migration_a_" + str(agreement) + "_k_" + str(k) + "_p_" +
        #                  str(
        #             pruning) + "_b_0_05")
        # if c_ij_precomputed:
        #     with open(c_ij_filename, 'rb') as inp:
        #         c_ij = pickle.load(inp)
        #     print("c_ij loaded")
        # else:
        #     c_ij = contracted_tree.C_ij()
        #     with open(c_ij_filename, 'wb') as outp:
        #         pickle.dump(c_ij, outp, pickle.HIGHEST_PROTOCOL)
        #     print("c_ij calculation done.")
        #
        # # calculate Dasgupta's measure D:
        # w_ij = 1 - kNN.pairwise_distances/np.max(kNN.pairwise_distances)
        # # set entries of w_ij of individuals of same population to zero:
        # for i in range(n):
        #     for j in range(i, n):
        #         if pop_membership[i] == pop_membership[j]:
        #             w_ij[i, j] = 0
        #             w_ij[j, i] = 0
        # nb_pairs = (n*(n-1)/2) - 8*(100*(100-1)/2)
        # D = np.sum(np.triu(np.multiply(w_ij,c_ij), 1))/nb_pairs
        # print("Dasgupta's measure:", D, c_ij_filename)

        # get number of characterizing SNPs per split (necessary as bipartitions have
        # been merged):
        num_char_cuts_per_split = []
        for k in range(1, len(list(char_cuts.keys())) + 1):
            num_char_cuts_per_split.append(np.sum(np.array(
                [name.count(",") + 1 for name in
                 bipartitions.names[list(char_cuts[k].keys())]])))
        num_char_cuts = dict(zip(char_cuts.keys(), num_char_cuts_per_split))
        print("num_char_cuts_per_split:", num_char_cuts_per_split)
        # print("positions:", positions)
        end_tangles_1 = time.time()
        print("time needed for tangles:", end_tangles_1 - start_tangles_1)

        # plot inferred ancestry and if specified also ADMIXTURE (seed is seed for
        # ADMIXTURE):
        plot_soft_clustering.plot_inferred_ancestry(matrices, pop_membership, agreement,
                                                    data_generation_mode, 19,
                                                    char_cuts,
                                                    num_char_cuts,
                                                    sorting_level="lowest",
                                                    plot_ADMIXTURE=plot_ADMIXTURE,
                                                    ADMIXTURE_file_name=ADMIXTURE_filename,
                                                    cost_fct=cost_fct_name)


if __name__ == '__main__':
    n = 800  # number of diploid individuals
    rho = 100  # recombination rate in the sim, when using vcf this parameter is irrelevant
    theta = 100  # mutation rate in sim, when using vcf this parameter s irrelevant
    agreement = 50  # agreement parameter
    k = 40  # number of neighbours for k-nearest neighbour
    pruning = 0  # pruning parameter
    seed = 42  # seed for simulation
    data_generation_mode = 'sim'
    # specify if data can be loaded or needs to be simulated:
    data_already_simulated = False
    # specify cost function: FST_kNN for FST-based cost function, HWE_kNN for
    # Hardy-Weinberg equilibrium based cost function:
    cost_fct_name = "FST_kNN"
    cost_precomputed = False  # cost pre-computed or not
    plot_ADMIXTURE = False  # compare tangles to ADMXITURE or not
    filepath = "data/with_demography/"  # filepath to the folder where the data is to be
    # saved/loaded.
    data = simulate_with_demography.Simulated_Data_With_Demography(n, theta, rho, seed,
                                                                   filepath=filepath)
    if data_already_simulated == False:
        data.sim_data()
        print("Data has been simulated.")
    else:
        data.load_data()
        print("Data has been loaded.")

    # extract vcf filename for ADMIXTURE:
    ADMIXTURE_filename = data.vcf_filename
    output_directory = Path('output_tangles_in_pop_gen')

    tangles_in_pop_gen(data, agreement, seed, k, pruning, data.indv_pop,
                       data_generation_mode, cost_fct_name,
                       cost_precomputed=cost_precomputed,
                       output_directory=output_directory, plot=True,
                       plot_ADMIXTURE=plot_ADMIXTURE,
                       ADMIXTURE_filename=ADMIXTURE_filename)

    print("all done.")
