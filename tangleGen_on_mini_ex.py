import sys
from functools import partial
from pathlib import Path

sys.path.append('..')
from src import cost_functions, data_types
from src import utils
import numpy as np
from src.tree_tangles import (ContractedTangleTree, tangle_computation,
                              compute_soft_predictions_children_popgen)
from src.utils import merge_doubles
import compute_kNN
import plot_soft_clustering
import pickle
import reliability_factor

"""
Simple script for use of tangleGen on the minimal example from the 
publication. The execution is divided in the following steps
    1. Specify genotype matrix
    2. Find the cuts and compute the costs
    3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
    4. Postprocess in soft and hard clustering
    5. plot soft clustering
"""


def tangles_in_pop_gen(agreement, k, pruning,
                       data_generation_mode, cost_fct_name, cost_precomputed=False,
                       output_directory='', plot=True):
    ## specify genotype matrix from minimal example:
    xs = np.array(
        [[0, 1, 0, 0, 0, 2], [0, 2, 0, 0, 0, 2], [0, 0, 0, 0, 0, 2], [1, 1, 0, 1, 0, 2],
         [0, 0, 1, 0, 0, 0], [0, 0, 2, 1, 0, 0], [0, 0, 2, 0, 2, 0], [0, 0, 2, 0, 2, 0],
         [0, 0, 1, 0, 2, 0]])
    # population membership per individual:
    pop_membership = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    # specify number of diploid individuals and number of sites (or mutations):
    n = xs.shape[0]
    nb_mut = xs.shape[1]
    data = data_types.Data(xs=xs)

    ## compute kNN for cost computation:
    kNN_precomputed = False  # specify if kNN already pre-computed or not
    kNN_filename = (str(data_generation_mode) + "_n_" + str(n) + "_sites_" + str(
        nb_mut) + "_k_" + str(k))
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

    ## calculate bipartitions
    bipartitions = data_types.Cuts(values=(data.xs > 0).T,
                                   names=np.array(list(range(0, data.xs.shape[1]))))
    print("\tFound {} bipartitions".format(len(bipartitions.values)), flush=True)
    print("\tCalculating costs if bipartitions", flush=True)

    ## compute costs for each cut:
    cost_function = getattr(cost_functions, cost_fct_name)
    saved_bipartitions_filename = (
                str(data_generation_mode) + "_n_" + str(n) + "_sites_" + str(
            nb_mut) + "_" + str(cost_fct_name))
    if cost_precomputed == False:
        bipartitions = utils.compute_cost_and_order_cuts(bipartitions,
                                                         partial(cost_function, data.xs,
                                                             None))
        with open('../tangles_in_pop_gen/data/saved_costs/' + str(
                saved_bipartitions_filename), 'wb') as handle:
            pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Load costs of bipartitions.")
        with open('../tangles_in_pop_gen/data/saved_costs/' + str(
                saved_bipartitions_filename), 'rb') as handle:
            bipartitions = pickle.load(handle)
    print("bipartitions.names:", bipartitions.names)
    print("bipartitions.costs:", bipartitions.costs)

    ## merge duplicate bipartitions:
    print("Merging doublicate mutations.")
    bipartitions = merge_doubles(bipartitions)

    print("Tangle algorithm", flush=True)
    # calculate the tangle search tree
    print("\tBuilding the tangle search tree", flush=True)
    tangles_tree = tangle_computation(cuts=bipartitions, agreement=agreement, verbose=3)
    print("Built tree has {} leaves".format(len(tangles_tree.maximals)), flush=True)
    # postprocess tree
    print("Postprocessing the tree.", flush=True)
    # contract to binary tree
    print("\tContracting to binary tree", flush=True)
    contracted_tree = ContractedTangleTree(tangles_tree)
    # calculate set of characterizing cuts:
    print("\tcalculating set of characterizing bipartitions", flush=True)
    contracted_tree.calculate_setP()
    # prune short paths:
    contracted_tree.prune(bipartitions, pruning)
    contracted_tree.calculate_setP()
    # assign weight/ importance to bipartitions. no weights used for soft clustering in
    # population genetics:
    weight = np.ones(len(bipartitions.names))
    # calculate soft clustering:
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

        # plot inferred ancestry:
        plot_soft_clustering.plot_inferred_ancestry(matrices, pop_membership, agreement,
                                                    data_generation_mode,
                                                    sorting_level="lowest",
                                                    cost_fct=cost_fct_name)


if __name__ == '__main__':
    agreement = 2   # agreement parameter
    k = 2           # number of neighbours for k-nearest neighbour
    pruning = 0     # pruning parameter
    data_generation_mode = 'sim'
    # specify cost function: FST_kNN for FST-based cost function, HWE_kNN for
    # Hardy-Weinberg equilibrium based cost function:
    cost_fct_name = "FST_kNN"
    cost_precomputed = False  # cost pre-computed or not
    filepath = "data/with_demography/"  # filepath to the folder where the data is to be
    output_directory = Path('output_tangles_in_pop_gen')

    tangles_in_pop_gen(agreement, k, pruning,
                       data_generation_mode, cost_fct_name,
                       cost_precomputed=cost_precomputed,
                       output_directory=output_directory, plot=True)

    print("all done.")