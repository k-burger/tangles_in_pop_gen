from functools import partial
from pathlib import Path

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
from src.utils import compute_hard_predictions, compute_mindset_prediciton

import simulate_with_demography

"""
Simple script for exemplary use of the tangle framework.
The execution is divided in the following steps

    1. Load datasets
    2. Find the cuts and compute the costs
    3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
    4. Postprocess in soft and hard clustering
"""


def tangles_in_pop_gen(sim_data, rho, theta, agreement, seed,
                       output_directory='', plot=True):
    xs = np.transpose(sim_data.G[0])
    n = xs.shape[0]
    nb_mut = xs.shape[1]
    print("n:", n)
    print("number of mutations:", nb_mut)
    mut_pos = sim_data.ts[0].tables.sites.position
    #print("mut pos:", mut_pos)
    #print("G:", xs)
    data = data_types.Data(xs=xs)

    # calculate your bipartitions we use the binary questions/features directly as bipartitions
    # print("\tGenerating set of bipartitions", flush=True)
    bipartitions = data_types.Cuts(values=(data.xs == True).T,
                                   names=np.array(list(range(0, data.xs.shape[1]))))

    print("\tFound {} unique bipartitions".format(len(bipartitions.values)), flush=True)
    print("\tCalculating costs if bipartitions", flush=True)
    bipartitions = utils.compute_cost_and_order_cuts(bipartitions,
                                                     partial(
                                                         cost_functions.mean_manhattan_distance,
                                                         data.xs, None))
    # bipartitions = utils.compute_cost_and_order_cuts(bipartitions,
    #                                                  partial(
    #                                                      cost_functions.mean_manhattan_distance_weighted_mut_pos,
    #                                                      data.xs, None, mut_pos,
    #                                                      lambda x: 0.07))

    np.save("data/mutation_labels", bipartitions.names)
    loaded = np.load("data/mutation_labels.npy", allow_pickle=True)
    print("Tangle algorithm", flush=True)
    # calculate the tangle search tree
    print("\tBuilding the tangle search tree", flush=True)
    tangles_tree = tangle_computation(cuts=bipartitions,
                                      agreement=agreement,
                                      verbose=3  # print everything
                                      )

    plot_cuts_in_one(data, bipartitions, Path('tmp'))

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
    weight = np.exp(-utils.normalize(bipartitions.costs))

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
        filename1 = "tree_n_" + str(n) + "_rho_" + str(rho) + "_theta_" + str(
            theta) + "_seed_" + str(
            seed) + "_agreement_" + str(agreement) + "_noise_" + str(noise) + ".svg"
        filename2 = "contracted_n_" + str(n) + "_rho_" + str(rho) + "_theta_" + str(
            theta) + "_seed_" + str(
            seed) + "_agreement_" + str(agreement) + "_noise_" + str(noise) + ".svg"
        tangles_tree.plot_tree(path=output_directory / filename1)
        ## plot contracted tree
        contracted_tree.plot_tree(path=output_directory / filename2)

        # plot tree summary
        tangles_tree.print_tangles_tree_summary_hard_predictions(nb_mut,
                                                                 bipartitions.names,
                                                                 ys_predicted)

        contracted_tree.print_summary(contracted_tree.root)

        # plot soft predictions
        plotting.plot_soft_predictions(data=data,
                                       contracted_tree=contracted_tree,
                                       eq_cuts=bipartitions.equations,
                                       path=output_directory / 'soft_clustering')


n = 15  # 4 15 10 # anzahl individuen (wenn n hoch dann theta auch hoch, rho eher
# runter)
# rho=int for constant theta in rep simulations, rho='rand' for random theta in (0,100) in every simulation:
rho = 1  # 1 3 1 recombination
# theta=int for constant theta in rep simulations, theta='rand' for random theta in (0,100) in every simulation:
theta = 17  # 30 #30 50 30 # mutationsrate
agreement = 3
seed = 17  # 42 #90 42 42
noise = 0
data_already_simulated = False  # True or False, states if data object should be
# simulated or loaded

# new parameters that need to be set to load/simulate appropriate data set
rep = 1  # number of repetitions during simulation
save_G = True  # set True to save genotype matrix during simulation, False otherwise
print_ts = True  # (set small for large n) set True if ts should be printed during simulation, this is only possible if rep==1. For large data sets, this step slows down the program noticeably.
save_ts = True  # set True to save the tree sequence during simulation, False otherwise
filepath = "data/with_demography/"  # filepath to the folder where the data is to be
# saved/loaded.

## This generates the data object and either simulates or loads the data sets
data = simulate_with_demography.Simulated_Data_With_Demography(n, rep, theta, rho, seed,
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

output_directory = Path('output_tangles_in_pop_gen')
plot = True
tangles_in_pop_gen(data, rho, theta, agreement, seed, output_directory, plot=True)