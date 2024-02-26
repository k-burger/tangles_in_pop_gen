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
import simulate_with_demography_mini_ex
import compute_kNN
import plot_soft_clustering
import pickle
import reliability_factor

"""
Simple script for use of the tangle framework on the minimal example from the 
publication. The execution is divided in the following steps
    1. Specify genotype matrix
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

    ## calculate bipartitions
    bipartitions = data_types.Cuts(values=(data.xs > 0).T,
                                   names=np.array(list(range(0, data.xs.shape[1]))))
    print("\tFound {} bipartitions".format(len(bipartitions.values)), flush=True)
    print("\tCalculating costs if bipartitions", flush=True)

    ## compute costs for each cut:
    cost_function = getattr(cost_functions, cost_fct_name)
    saved_bipartitions_filename = (
                str(data_generation_mode) + "_n_" + str(n) + "_sites_" + str(
            nb_mut) + "_" + str(cost_fct_name) + "_seed_" + str(seed))
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

        ## create vcf file with same properties as specified genotype matrix from
        # minimal example:
        mini_ex_vcf_path = "data/vcf/mini_ex_" + ADMIXTURE_filename + ".vcf"
        with (open("data/vcf/" + ADMIXTURE_filename + ".vcf", 'r') as infile, open(
            mini_ex_vcf_path, 'w') as outfile):
            for line in infile:
                # print(line)
                if line.startswith(
                        '#'):  # Schreibe Header-Zeilen direkt in die reduzierte Datei
                    outfile.write(line)
                else:
                    parts = line.split('\t')  # Trenne die Zeile in Teile
                    if parts[2] == str(0):
                        parts[9] = '0|0'
                        parts[10] = '0|0'
                        parts[11] = '0|0'
                        parts[12] = '1|0'
                        parts[13] = '0|0'
                        parts[14] = '0|0'
                        parts[15] = '0|0'
                        parts[16] = '0|0'
                        parts[17] = '0|0'
                        updated_line = '\t'.join(parts)
                        outfile.write(updated_line + '\n')
                    elif parts[2] == str(1):
                        parts[9] = '0|1'
                        parts[10] = '1|1'
                        parts[11] = '0|0'
                        parts[12] = '1|0'
                        parts[13] = '0|0'
                        parts[14] = '0|0'
                        parts[15] = '0|0'
                        parts[16] = '0|0'
                        parts[17] = '0|0'
                        updated_line = '\t'.join(parts)
                        outfile.write(updated_line + '\n')
                    elif parts[2] == str(2):
                        parts[9] = '0|0'
                        parts[10] = '0|0'
                        parts[11] = '0|0'
                        parts[12] = '0|0'
                        parts[13] = '0|1'
                        parts[14] = '1|1'
                        parts[15] = '1|1'
                        parts[16] = '1|1'
                        parts[17] = '1|0'
                        updated_line = '\t'.join(parts)
                        outfile.write(updated_line + '\n')
                    elif parts[2] == str(3):
                        parts[9] = '0|0'
                        parts[10] = '0|0'
                        parts[11] = '0|0'
                        parts[12] = '1|0'
                        parts[13] = '0|0'
                        parts[14] = '0|1'
                        parts[15] = '0|0'
                        parts[16] = '0|0'
                        parts[17] = '0|0'
                        updated_line = '\t'.join(parts)
                        outfile.write(updated_line + '\n')
                    elif parts[2] == str(4):
                        parts[9] = '0|0'
                        parts[10] = '0|0'
                        parts[11] = '0|0'
                        parts[12] = '0|0'
                        parts[13] = '0|0'
                        parts[14] = '0|0'
                        parts[15] = '1|1'
                        parts[16] = '1|1'
                        parts[17] = '1|1'
                        updated_line = '\t'.join(parts)
                        outfile.write(updated_line + '\n')
                    elif parts[2] == str(5):
                        parts[9] = '1|1'
                        parts[10] = '1|1'
                        parts[11] = '1|1'
                        parts[12] = '1|1'
                        parts[13] = '0|0'
                        parts[14] = '0|0'
                        parts[15] = '0|0'
                        parts[16] = '0|0'
                        parts[17] = '0|0'
                        updated_line = '\t'.join(parts)
                        outfile.write(updated_line + '\n')

        # plot inferred ancestry and if specified also ADMIXTURE (seed is seed for
        # ADMIXTURE):
        plot_soft_clustering.plot_inferred_ancestry(matrices, pop_membership, agreement,
                                                    data_generation_mode, 4, char_cuts,
                                                    num_char_cuts,
                                                    sorting_level="lowest",
                                                    plot_ADMIXTURE=plot_ADMIXTURE,
                                                    ADMIXTURE_file_name=ADMIXTURE_filename,
                                                    cost_fct=cost_fct_name)


if __name__ == '__main__':
    n = 9  # number of diploid individuals
    rho = 0  # recombination rate in the sim, when using vcf this parameter is irrelevant
    theta = 0.75  # mutation rate in sim, when using vcf this parameter s irrelevant
    agreement = 2  # agreement parameter
    k = 2  # number of neighbours for k-nearest neighbour
    pruning = 0  # pruning parameter
    seed = 4  # seed for simulation
    data_generation_mode = 'sim'
    # specify if data can be loaded or needs to be simulated:
    data_already_simulated = True
    # specify cost function: FST_kNN for FST-based cost function, HWE_kNN for
    # Hardy-Weinberg equilibrium based cost function:
    cost_fct_name = "FST_kNN"
    cost_precomputed = True  # cost pre-computed or not
    plot_ADMIXTURE = True  # compare tangles to ADMXITURE or not
    filepath = "data/with_demography/"  # filepath to the folder where the data is to be
    # saved/loaded.
    data = simulate_with_demography_mini_ex.Simulated_Data_With_Demography(n, theta,
                                                                           rho, seed,
                                                                           filepath=filepath)

    if data_already_simulated == False:
        data.sim_data()
        print("Data has been simulated.")
    else:
        data.load_data()
        print("Data has been loaded.")

    ADMIXTURE_filename = data.vcf_filename
    output_directory = Path('output_tangles_in_pop_gen')

    tangles_in_pop_gen(data, agreement, seed, k, pruning, data.indv_pop,
                       data_generation_mode, cost_fct_name,
                       cost_precomputed=cost_precomputed,
                       output_directory=output_directory, plot=True,
                       plot_ADMIXTURE=plot_ADMIXTURE,
                       ADMIXTURE_filename=ADMIXTURE_filename)

    print("all done.")
