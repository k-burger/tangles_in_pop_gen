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
# import tskit
#import demesdraw
#import matplotlib.pyplot as plt


class Simulated_Data_With_Demography_Diploid:
    def __init__(self, n_diploid, rep, theta, rho, seed, save_G=True, print_ts=False,
                 save_ts=False, filepath="", SFS=[], total_tree_length=[], G=[],
                 true_thetas=[], true_rhos=[], ts=[], indv_pop=[],
                 admixture_filename=[]):
        # parameters that need to be set and are not a result of the simulation:
        self.n = n_diploid
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
        self.indv_pop = indv_pop
        self.admixture_filename = admixture_filename

    def sim_data(self):
        #n_haploid = 2*self.n    # get haploid sample size
        multi_SFS = []  # list to save the SFS
        multi_total_length = []  # list to save the total tree lengths
        multi_G = []  # list to save the genotype matrices
        multi_theta = []  # list to save theta used for simulating
        multi_rho = []  # list to save rho used for simulating
        multi_ts = []  # list to save the tree sequences

        num_mutations = []
        tree_length = []
        num_trees = []
        seed_vec = list(range(int(self.seed+2), int(self.seed+self.rep+2)))


        # check if dataset is simulated for training:
        if self.theta == 'rand':
            train = True
            theta_str = "random-100"
        else:
            train = False
            theta_str = str(self.theta)


        if self.rho == 'rand':
            rho_train = True
            rho_str = "random-50"
        else:
            rho_train = False
            rho_str = str(self.rho)

        # if training data, take in each iteration new theta to simulate
        if train:
            numpy.random.seed(self.seed - 1)
            theta = numpy.array(numpy.random.uniform(0, 100, rep))
            multi_theta = theta
        else:
            theta = self.theta*numpy.ones(self.rep)
            multi_theta = theta

        if rho_train:
            numpy.random.seed(self.seed - 2)
            rho = numpy.array(numpy.random.uniform(0, 50, self.rep))
            multi_rho = rho
        else:
            rho = self.rho*numpy.ones(self.rep)
            multi_rho = rho

        # define demography for 3 populations
        # demography = msprime.Demography()
        # demography.add_population(name="A", initial_size=0.5)
        # demography.add_population(name="B", initial_size=0.5)
        # demography.add_population(name="C", initial_size=0.5)
        # demography.add_population(name="AB", initial_size=0.5)
        # demography.add_population(name="ABC", initial_size=0.5)
        # demography.add_population_split(time=0.5, derived=["A", "B"], ancestral="AB")
        # demography.add_population_split(time=1, derived=["AB", "C"], ancestral="ABC")
        # demography.set_symmetric_migration_rate(["B", "C"], 0.5)

        # define demography for 8 populations
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=0.25)
        demography.add_population(name="B", initial_size=0.25)
        demography.add_population(name="C", initial_size=0.25)
        demography.add_population(name="D", initial_size=0.25)
        demography.add_population(name="E", initial_size=0.25)
        demography.add_population(name="F", initial_size=0.25)
        demography.add_population(name="G", initial_size=0.25)
        demography.add_population(name="H", initial_size=0.25)
        demography.add_population(name="AB", initial_size=0.25)
        demography.add_population(name="CD", initial_size=0.25)
        demography.add_population(name="ABCD", initial_size=0.25)
        demography.add_population(name="EF", initial_size=0.25)
        demography.add_population(name="EFG", initial_size=0.25)
        demography.add_population(name="EFGH", initial_size=0.25)
        demography.add_population(name="ABCDEFGH", initial_size=0.25)
        demography.add_population_split(time=2/7, derived=["A", "B"], ancestral="AB")
        demography.add_population_split(time=4 / 7, derived=["E", "F"], ancestral="EF")
        demography.add_population_split(time=6/7, derived=["C", "D"], ancestral="CD")
        demography.add_population_split(time=8 / 7, derived=["EF", "G"],
                                        ancestral="EFG")
        demography.add_population_split(time=10/7, derived=["AB", "CD"],
                                        ancestral="ABCD")
        demography.add_population_split(time=12/7, derived=["EFG", "H"],
                                        ancestral="EFGH")
        demography.add_population_split(time=2, derived=["ABCD", "EFGH"],
                                        ancestral="ABCDEFGH")
        #demography.set_symmetric_migration_rate(["A", "B", "C", "D", "E", "F", "G",
        #                                         "H"],
        #                                        0.5)
        demography.set_symmetric_migration_rate(["A", "E"],0.5)

        size = self.n // 8   # constant population size for 8 simulated populations

        # graph = msprime.Demography.to_demes(demography)
        # fig, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
        # demesdraw.tubes(graph, ax=ax, seed=1)
        # filename_short = (self.filepath + "demographic_structure_n_" + str(self.n))
        # plt.savefig(filename_short + '.pdf')
        # plt.show()

        # simulate a datasets of size rep
        for i in range(0, self.rep):
            # set seed
            rng = numpy.random.default_rng(seed_vec[i])
            seeds = rng.integers(1, 2 ** 31 - 1, size=2)

            ts = msprime.sim_ancestry(samples={"A": size, "B": size, "C": size, "D": size,
                                               "E": size, "F": size, "G": size, "H": size},
                                      sequence_length=1,
                                      discrete_genome=False,# population_size=0.5,
                                      recombination_rate=rho[i]/1, random_seed=seeds[
                    0], demography=demography, ploidy=2)

            #for p in ts.provenances():
            #    print(p)

            tree_sequence = msprime.sim_mutations(ts, rate=theta[i]/1,
                                                  random_seed=seeds[1],
                                                  discrete_genome=False)
            #print("actual seed ts simulation:", seeds[0])
            #print("actual seed mutation simulation:", seeds[1])
            #ts.dump("data/ts_n_10_rho_1_theta_17")
            #tree_sequence.dump("data/ts_with_mutation_n_10_rho_1_theta_17")


            print("ts.individuals_population:", tree_sequence.individuals_population)
            print("num_migration:", tree_sequence.num_migrations)
            indv_names = [f"{i}indv" for i in range(self.n)]
            print("indv_names:", indv_names)
            admixture_filename = ("n_" +str(self.n) +"_rep_" + str(self.rep) + "_rho_" + rho_str +
                                  "_theta_" + theta_str + "_seed_" + str(self.seed))
            with open("admixture/data/"+admixture_filename+".vcf", "w") as vcf_file:
                tree_sequence.write_vcf(vcf_file, individual_names=indv_names)

            if self.save_ts:
                multi_ts.append(tree_sequence)

            num_mutations.append(tree_sequence.num_mutations)
            print("num_mutations:", num_mutations)

            m = 0
            for tree in tree_sequence.trees():
                m = m + 1
                tree_length.append(tree.total_branch_length)

            num_trees.append(m)
            print("num trees:", num_trees)
            #print("tree length:", tree_length)

            # get mean total tree length and save as entry of multi_total_length
            mean_tot_branch_length = 0
            for tree in tree_sequence.trees():
                mean_tot_branch_length += tree.total_branch_length * (
                        tree.interval[1] - tree.interval[0]
                )
            multi_total_length.append(mean_tot_branch_length)

            print("mean total branch length:", mean_tot_branch_length)

            # get genotype matrix
            G_haploid = tree_sequence.genotype_matrix()
            print("n diploid in sim:", self.n)
            # array with indices of rows to sum
            column_indices = numpy.arange(self.n) * 2
            # sum rows and create new numpy ndarray
            G = G_haploid[:,column_indices] + G_haploid[:,column_indices + 1]
            print("G_haploid:\n", G_haploid)
            print("G:\n", G)
            print("G.shape[1]:", G.shape[1])
            # potentially save the genotype matrix:
            if self.save_G:
                multi_G.append(G)
            assert G.shape[1] == self.n

            # calculate site frequency spectrum from genotype matrix
            # sum over columns of the genotype matrix
            a = G.sum(axis=1)
            # site frequency spectrum
            S = numpy.bincount(a, None, self.n)[1:self.n]

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
        self.true_thetas = multi_theta
        self.G = multi_G
        self.SFS = multi_SFS
        self.total_tree_length = multi_total_length
        self.true_rhos = multi_rho
        self.ts = multi_ts
        self.indv_pop = tree_sequence.individuals_population
        self.admixture_filename = admixture_filename

        # save object
        filename = (self.filepath + "sim_with_demography_n_" + str(self.n) + "_rep_" + str(
            self.rep) + "_rho_" + rho_str + "_theta_" + theta_str + "_seed_" + str(
            self.seed))
        if self.print_ts and self.rep==1:
            filename_svg = filename + ".svg"
            #SVG(tree_sequence.draw_svg(path=filename_svg, x_axis=True, y_axis=True, symbol_size=5, y_label=[]))
            # for larger n, use the following instead of the SVG print command above:
            wide_fmt = (3000, 250) #this sets the format of the plot, for smaller/larger n decrease/increase first entry (x-axis).
            SVG(tree_sequence.draw_svg(path=filename_svg, x_axis=True, y_axis=True, time_scale="rank",  #used time_scale="rank" for better visualization of large tree sequences
                                     symbol_size=5, y_label=[], size=wide_fmt))
        elif self.print_ts and self.rep>1:
            warnings.warn(
                'you try to print the ts but rep>1. only print svg if save_svg==True and rep==1.',
                stacklevel=1)
        with open(filename, 'wb') as outp:  # overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)



    def load_data(self):
        # covert theta and rho_scenario into str to load correct data
        if self.theta == 'rand':
            train = True
            theta_str = "random-100"
        else:
            train = False
            theta_str = str(self.theta)

        if self.rho == 'rand':
            rho_train = True
            rho_str = "random-50"
        else:
            rho_train = False
            rho_str = str(self.rho)

        # load saved data
        filename = (self.filepath + "sim_with_demography_n_" + str(self.n) + "_rep_" + str(
            self.rep) + "_rho_" + rho_str + "_theta_" + theta_str + "_seed_" + str(
            self.seed))
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
        self.indv_pop = loaded_data.indv_pop
        self.admixture_filename = loaded_data.admixture_filename

        # check if genotype matrix and ts have been saved during simulated:
        if self.G == []:
            warnings.warn(
                'the genotype matrix was not saved during simulation, this can lead to further problems.\n if necessary, simulate the data again with the corresponding seed.',
                stacklevel=1)
        if self.ts == []:
            warnings.warn(
                'tree sequence was not saved during simulation, this can lead to further problems.\n if necessary, simulate the data again with the corresponding seed.',
                stacklevel=1)


use_this_script_for_sim = False
if use_this_script_for_sim == True:
    ## This is the infomation needed in any script that wants to use the data object class:
    n = 8              # sample size
    rep = 1             # number of repetitions during simulation
    theta = 0.25          # theta=int for constant theta in rep simulations,
    # theta='rand' for random theta in (0,100) in every simulation
    rho = 0            # rho=int for constant theta in rep simulations, rho='rand'
    # for random theta in (0,100) in every simulation
    seed = 42           # starting seed for simulation (based on this seed, multiple
    # seeds will be generated)
    save_G = True       # set True to save genotype matrix during simulation, False otherwise
    print_ts = False     # set True if ts should be printed during simulation, this is
    # only possible if rep==1. For large data sets, this step slows down the program noticeably.
    save_ts = True      # set True to save the tree sequence during simulation, False otherwise
    filepath = "data/with_demography/"
    data_already_simulated = False  # True or False, states if data object should be simulated or loaded

    ## This generates the data object and either simulates the properties or loads if it already exists.
    data = Simulated_Data_With_Demography_Diploid(n, rep, theta, rho, seed, save_G=save_G,
                          print_ts=print_ts, save_ts=save_ts, filepath=filepath)
    if data_already_simulated == False:
        data.sim_data()
    else:
        data.load_data()

    print("G:", data.G)


    print("simulation done.")