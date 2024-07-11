import pickle
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

import compute_kNN
import plot_soft_clustering
import read_vcf
import reliability_factor
from src import cost_functions, data_types
from src import outsourced_cost_computation
from src import utils
from src.tree_tangles import ContractedTangleTree, tangle_computation, \
    compute_soft_predictions_children_popgen
from src.utils import merge_doubles

sys.path.append('..')

"""
Simple script for use of tangleGen on 1kG project phase 3, chr22. The 
execution is divided in the following steps

    1. Load/simulate the dataset
    2. Find the cuts and compute the costs
    3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
    4. Postprocess in soft and hard clustering
    5. plot soft clustering
    
NOTE: vcf file containing chr22 has to be downloaded first from the 1000 Genomes 
Project Phase 3:
http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
Save vcf file in data/vcf/
"""


def tangles_in_pop_gen(sim_data, agreement, seed, k, pruning, pop_membership,
                       data_generation_mode, cost_fct_name, cost_precomputed=False,
                       output_directory='', plot=True, plot_ADMIXTURE=False,
                       ADMIXTURE_filename=""):
    # get genotype matrix xs and mutation idx
    xs = np.transpose(sim_data.G[0])  # diploid genotype matrix
    mutations_in_sim = np.arange(xs.shape[1])

    # get population membership for each individual (individuals in vcf file are also
    # sorted by populations, i.e. same sorting):
    panel_file = ('data/vcf/integrated_call_samples_v3.20130502.ALL.panel')
    panel_df = pd.read_csv(panel_file, delimiter='\t')
    sample_pop = panel_df['pop']
    pop_array = np.array(sample_pop)
    custom_order = {'YRI': 0, 'LWK': 1, 'GWD': 2, 'MSL': 3, 'ESN': 4, 'ASW': 5,
                    'ACB': 6, 'FIN': 7, 'CEU': 8, 'GBR': 9, 'TSI': 10, 'IBS': 11,
                    'GIH': 12, 'PJL': 13, 'BEB': 14, 'STU': 15, 'ITU': 16, 'CHB': 17,
                    'JPT': 18, 'CHS': 19, 'CDX': 20, 'KHV': 21, 'MXL': 22, 'PUR': 23,
                    'CLM': 24, 'PEL': 25}
    custom_sort_criteria = np.array([custom_order[pop] for pop in pop_array])
    sorted_indices = np.argsort(custom_sort_criteria)
    pop_membership = pop_array[sorted_indices]
    print("pop_membership after resorting:", pop_membership)
    xs = xs[sorted_indices]
    # exlude AMR:
    xs = xs[:2157]
    pop_membership = pop_membership[:2157]
    print("pop_membership after resorting without AMR:", pop_membership)

    ## pre-processing of the data: deletion of multiallelicity sites and SNPs which
    # for all individuals either have no ancestral alleles at all or only ancestral
    # alleles. in addition, delete SNPs with very low and very high allele frequencies.
    print("number of sites before deletion:", len(mutations_in_sim))
    num_zero_mut = 0
    num_n_mut = 0
    num_multiallelic = 0
    num_low_freq = 0
    num_high_freq = 0
    columns_to_delete_0 = []
    columns_to_delete_n = []
    columns_to_delete_low_freq = []
    columns_to_delete_high_freq = []
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
    # delete low-frequency alleles:
    for m in range(0, xs.shape[1]):
        if np.count_nonzero(xs[:, m]) < 100:
            columns_to_delete_low_freq.append(m)
            num_low_freq = num_low_freq + 1
    xs = np.delete(xs, columns_to_delete_low_freq, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_low_freq)
    print("low freq. sites deleted:", num_low_freq)
    # delete high-frequency alleles:
    for m in range(0, xs.shape[1]):
        if np.count_nonzero(xs[:, m]) > xs.shape[0] - 100:
            columns_to_delete_high_freq.append(m)
            num_high_freq = num_high_freq + 1
    xs = np.delete(xs, columns_to_delete_high_freq, axis=1)
    mutations_in_sim = np.delete(mutations_in_sim, columns_to_delete_high_freq)
    print("number of sites after deletion:", len(mutations_in_sim))

    # specify number of diploid individuals and number of sites (or mutations):
    n = xs.shape[0]
    nb_mut = xs.shape[1]
    data = data_types.Data(xs=xs)

    ## compute kNN for cost computation:
    kNN_precomputed = True  # specify if kNN already pre-computed or not
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
    # pickle kNN-Matrix for cost function to load:
    with open("data/saved_kNN/kNN", 'wb') as outp:  # overwrites existing file.
        pickle.dump(kNN, outp, pickle.HIGHEST_PROTOCOL)

    ## calculate bipartitions:
    print("\tGenerating set of bipartitions", flush=True)
    bipartitions = data_types.Cuts(values=(data.xs > 0).T,
                                   names=np.array(list(range(0, data.xs.shape[1]))))
    print("\tFound {} bipartitions".format(len(bipartitions.values)), flush=True)
    print("\tCalculating costs if bipartitions", flush=True)
    # current memory usage:
    print(f"Current memory usage 1: {psutil.virtual_memory().percent}%")
    print("\tFound {} bipartitions".format(len(bipartitions.values)), flush=True)
    print("\tCalculating costs if bipartitions", flush=True)

    ## compute costs for each cut:
    cost_function = getattr(cost_functions, cost_fct_name)
    saved_costs_filename = (
                str(data_generation_mode) + "_n_" + str(n) + "_sites_" + str(
            nb_mut) + "_" + str(cost_fct_name) + "_1kG_chr22")
    start = time.time()
    print("time started")
    if cost_precomputed == False:
        bipartitions = outsourced_cost_computation.compute_cost_and_order_cuts(
            bipartitions, partial(cost_function, data.xs, None))
        with open('../tangles_in_pop_gen/data/saved_costs/' + str(saved_costs_filename),
                  'wb') as handle:
            pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Load costs of bipartitions.")
        with open('../tangles_in_pop_gen/data/saved_costs/' + str(saved_costs_filename),
                  'rb') as handle:
            bipartitions = pickle.load(handle)
    end = time.time()
    print("time needed:", end - start)
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
    tangles_tree = tangle_computation(cuts=bipartitions, agreement=agreement, verbose=3,
                                      max_clusters=10)
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

        # plot inferred ancestry and if specified also ADMIXTURE (seed is seed for
        # ADMIXTURE):
        plot_soft_clustering.plot_inferred_ancestry(matrices, pop_membership, agreement,
                                                    data_generation_mode, seed,
                                                    char_cuts, num_char_cuts,
                                                    sorting_level="lowest",
                                                    plot_ADMIXTURE=plot_ADMIXTURE,
                                                    ADMIXTURE_file_name=ADMIXTURE_filename,
                                                    cost_fct=cost_fct_name)


if __name__ == '__main__':
    n = 2157  # number of diploid individuals
    rho = 100  # recombination rate in the sim, when using vcf this parameter is irrelevant
    theta = 100  # mutation rate in sim, when using vcf this parameter s irrelevant
    agreement = 200  # agreement parameter
    k = 40  # number of neighbours for k-nearest neighbour
    pruning = 0  # pruning parameter
    seed = 40  # seed for ADMIXTURE
    data_generation_mode = 'readVCF'
    data_set = 'chr22'
    # specify if data can be loaded or needs to be initially processed:
    data_already_processed = False
    # specify cost function: FST_kNN for FST-based cost function, HWE_kNN for
    # Hardy-Weinberg equilibrium based cost function:
    cost_fct_name = "HWE_kNN"
    cost_precomputed = True  # cost pre-computed or not
    plot_ADMIXTURE = False  # compare tangles to ADMIXTURE or not
    filepath = "data/with_demography/"  # filepath to the folder where the data is to be
    # saved/loaded.

    # load vcf file and process:
    rho = -1
    theta = -1
    data = read_vcf.ReadVCF(n, data_set, '1000G_chr22', 'data/vcf/')
    if data_already_processed:
        data.load_vcf()
    else:
        data.read_vcf()

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