#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate.py: Function to simulate genealogical trees via msprime.
In particular, it simulates SFS, genotype matrix, tree lengths...
Created on Mon Dec 14 14:16:36 2020

@author: Franz Baumdicker, Klara Burger
"""
import numpy
import msprime
import collections
import random
import pickle
import warnings
from IPython.display import SVG


class Simulated_Data:
    def __init__(self, n, rep, theta, rho, seed, save_G=True, print_ts=False, save_ts=False, filepath="", SFS=[], total_tree_length=[], G=[], true_thetas=[], true_rhos=[], ts=[]):
        # parameters that need to be set and are not a result of the simulation:
        self.n = n
        self.rep = rep
        self.theta = theta
        self.rho = rho
        self.seed = seed
        self.save_G = save_G
        self.save_ts = save_ts
        self.print_ts = print_ts
        self.filepath = filepath

        # properties that are results of the simulation:
        self.SFS = SFS
        self.total_tree_length = total_tree_length
        self.G = G
        self.true_thetas = true_thetas
        self.true_rhos = true_rhos
        self.ts = ts

    def sim_data(self, n, rep, theta, rho, seed, save_G=True, print_ts=False, save_ts=False, filepath=""):
        multi_SFS = []  # list to save the SFS
        multi_total_length = []  # list to save the total tree lengths
        multi_G = []  # list to save the genotype matrices
        multi_theta = []  # list to save theta used for simulating
        multi_rho = []  # list to save rho used for simulating
        multi_ts = []  # list to save the tree sequences

        num_mutations = []
        tree_length = []
        num_trees = []
        seed_vec = list(range(int(seed+2), int(seed+rep+2)))


        # check if dataset is simulated for training:
        if theta == 'rand':
            train = True
            theta_str = "random-100"
        else:
            train = False
            theta_str = str(theta)


        if rho == 'rand':
            rho_train = True
            rho_str = "random-50"
        else:
            rho_train = False
            rho_str = str(rho)

        # if training data, take in each iteration new theta to simulate
        if train:
            numpy.random.seed(seed - 1)
            theta = numpy.array(numpy.random.uniform(0, 100, rep))
            multi_theta = theta
        else:
            theta = theta*numpy.ones(rep)
            multi_theta = theta

        if rho_train:
            numpy.random.seed(seed - 2)
            rho = numpy.array(numpy.random.uniform(0, 50, rep))
            multi_rho = rho
        else:
            rho = rho*numpy.ones(rep)
            multi_rho = rho

        # simulate a datasets of size rep
        for i in range(0, rep):
            # set seed
            rng = numpy.random.default_rng(seed_vec[i])
            seeds = rng.integers(1, 2 ** 31 - 1, size=2)

            ts = msprime.sim_ancestry(n, sequence_length=1, discrete_genome=False, population_size=0.5,
                                      recombination_rate=rho[i], random_seed=seeds[0], ploidy=1)
            tree_sequence = msprime.sim_mutations(ts, rate=theta[i], random_seed=seeds[1], discrete_genome=False)


            if save_ts:
                multi_ts.append(tree_sequence)

            num_mutations.append(tree_sequence.num_mutations)

            m = 0
            for tree in tree_sequence.trees():
                m = m + 1
                tree_length.append(tree.total_branch_length)

            num_trees.append(m)

            # get mean total tree length and save as entry of multi_total_length
            mean_tot_branch_length = 0
            for tree in tree_sequence.trees():
                mean_tot_branch_length += tree.total_branch_length * (
                        tree.interval[1] - tree.interval[0]
                )
            multi_total_length.append(mean_tot_branch_length)

            # get genotype matrix
            G = tree_sequence.genotype_matrix()
            # potentially save the genotype matrix:
            if save_G:
                multi_G.append(G)
            assert G.shape[1] == n

            # calculate site frequency spectrum from genotype matrix
            # sum over columns of the genotype matrix
            a = G.sum(axis=1)
            # site frequency spectrum
            S = numpy.zeros((n - 1,), dtype=int)
            for j in range(0, n - 1):
                S[j] = collections.Counter(a)[j + 1]

            # save the SFS and the theta used for simulation
            multi_SFS.append(S)

        # for i in range(0,rep):
        #    multi_G[i] = numpy.concatenate((multi_G[i], numpy.zeros(((271 - multi_G[i].shape[0]), num_sample))))
        # G_filled = numpy.stack([multi_G[l] for l in range(0, rep)])

        # print properties
        #print("mean number mutations:", numpy.mean(num_mutations))
        #print("mean tree length:", numpy.mean(tree_length))
        #print("mean number of trees:", numpy.mean(num_trees))

        # assign properties
        self.seed = seed_vec
        self.true_thetas = multi_theta
        self.G = multi_G
        self.SFS = multi_SFS
        self.total_tree_length = multi_total_length
        self.true_rhos = multi_rho
        self.ts = multi_ts

        # save object
        filename = (filepath + "sim_n_" + str(n) + "_rep_" + str(
            rep) + "_rho_" + rho_str + "_theta_" + theta_str + "_seed_" + str(seed))
        if print_ts and rep==1:
            filename_svg = filename + "_svg"
            SVG(tree_sequence.draw_svg(path=filename_svg, x_axis=True, y_axis=True,
                                     symbol_size=5, y_label=[]))
        elif print_ts and rep>1:
            warnings.warn(
                'you try to print the ts but rep>1. only print svg if save_svg==True and rep==1.',
                stacklevel=1)
        with open(filename, 'wb') as outp:  # overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


    def load_data(self, n, rep, theta, rho, seed, filepath):
        # covert theta and rho_scenario into str to load correct data
        if theta == 'rand':
            train = True
            theta_str = "random-100"
        else:
            train = False
            theta_str = str(theta)

        if rho == 'rand':
            rho_train = True
            rho_str = "random-50"
        else:
            rho_train = False
            rho_str = str(rho)

        # load saved data
        filename = (filepath + "sim_n_" + str(n) + "_rep_" + str(rep) + "_rho_" + rho_str + "_theta_" + theta_str + "_seed_" + str(seed))
        with open(filename, 'rb') as inp:
            loaded_data = pickle.load(inp)  # could be changed to self=pickle.load(inp) or similar to save memory.

        # assign properties
        self.seed = loaded_data.seed
        self.true_thetas = loaded_data.true_thetas
        self.G = loaded_data.G
        self.SFS = loaded_data.SFS
        self.total_tree_length = loaded_data.total_tree_length
        self.true_rhos = loaded_data.true_rhos
        self.ts = loaded_data.ts

        # check if genotype matrix and ts have been saved during simulated:
        if self.G == []:
            warnings.warn(
                'the genotype matrix was not saved during simulation, this can lead to further problems.\n if necessary, simulate the data again with the corresponding seed.',
                stacklevel=1)
        if self.ts == []:
            warnings.warn(
                'tree sequence was not saved during simulation, this can lead to further problems.\n if necessary, simulate the data again with the corresponding seed.',
                stacklevel=1)




use_this_script_for_sim = True
if use_this_script_for_sim == True:
    ## This is the infomation needed in any script that wants to use the data object class:
    n = 10              # sample size
    rep = 1             # number of repetitions during simulation
    theta = 17          # theta=int for constant theta in rep simulations, theta='rand' for random theta in (0,100) in every simulation
    rho = 1             # rho=int for constant theta in rep simulations, rho='rand' for random theta in (0,100) in every simulation
    seed = 17           # starting seed for simulation (based on this seed, multiple seeds will be generated)
    save_G = True       # set True to save genotype matrix during simulation, False otherwise
    print_ts = True     # set True if ts should be printed during simulation, this is only possible if rep==1. For large data sets, this step slows down the program noticeably.
    save_ts = True      # set True to save the tree sequence during simulation, False otherwise
    filepath = "tangles_in_pop_gen/data/"

    data_already_simulated = False  # True or False, states if data object should be simulated or loaded

    ## This generates the data object and either simulates the properties or loads if it already exists.
    data = Simulated_Data(n, rep, theta, rho, seed)
    if data_already_simulated == False:
        data.sim_data(n, rep, theta, rho, seed, save_G=save_G, print_ts=print_ts, save_ts=save_ts, filepath=filepath)
    else:
        data.load_data(n, rep, theta, rho, seed, filepath=filepath)


print("simulation done.")