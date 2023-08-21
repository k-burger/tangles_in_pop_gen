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
import stdpopsim
from IPython.display import SVG
import allel
# import tskit
import demesdraw
import matplotlib.pyplot as plt


class SimulateOutOfAfrica:
    def __init__(self, n_diploid, seed, save_G=True, print_ts=False,
                 save_ts=False, filepath="", SFS=[], total_tree_length=[], G=[],
                 true_thetas=[], true_rhos=[], ts=[], indv_pop=[],
                 admixture_filename=[]):
        # parameters that need to be set and are not a result of the simulation:
        self.n = n_diploid
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

        # simulate a datasets of size rep
        species = stdpopsim.get_species("HomSap")
        model = species.get_demographic_model("OutOfAfrica_3G09")

        # graph = msprime.Demography.to_demes(model)
        # fig, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
        # demesdraw.tubes(graph, ax=ax, seed=1)
        # filename_short = (self.filepath + "out_of_africa_n_" + str(self.n) + "_seed_"
        #                   + str(self.seed))
        # plt.savefig(filename_short + '.pdf')
        # plt.show()
        print(model.num_populations)
        print(model.num_sampling_populations)
        print([pop.name for pop in model.populations])
        contig = species.get_contig("chr22", mutation_rate=model.mutation_rate)
        # default is a flat genetic map
        print("mean recombination rate:",
              f"{contig.recombination_map.mean_rate:.3}")
        print("mean mutation rate:", contig.mutation_rate)
        print("model mutation rate:", model.mutation_rate)
        size = self.n // 3
        samples = {"YRI": size, "CHB": size, "CEU": size}
        engine = stdpopsim.get_engine("msprime")
        tree_sequence = engine.simulate(model, contig, samples, seed=self.seed)
        print("num sites:", tree_sequence.num_sites)


        print("ts.individuals_population:", tree_sequence.individuals_population)
        print("num_migration:", tree_sequence.num_migrations)
        indv_names = [f"{i}indv" for i in range(self.n)]
        print("indv_names:", indv_names)
        admixture_filename = ("out_of_africa_n_" + str(self.n)+ "_seed_" + str(
            self.seed))
        with open("admixture/data/" + admixture_filename + ".vcf", "w") as vcf_file:
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
        # print("tree length:", tree_length)

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
        G = G_haploid[:, column_indices] + G_haploid[:, column_indices + 1]
        print("G_haploid:\n", G_haploid)
        print("G.shape[1] haploid:", G_haploid.shape[1])
        count_larger_1 = numpy.count_nonzero(G_haploid == 4)
        print("count_larger_1:", count_larger_1)
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
        self.G = multi_G
        self.SFS = multi_SFS
        self.total_tree_length = multi_total_length
        self.ts = multi_ts
        self.indv_pop = tree_sequence.individuals_population
        self.admixture_filename = admixture_filename

        # save object
        filename = (self.filepath + "out_of_africa_n_" +str(self.n) + "_seed_" + str(
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
        # load saved data
        filename = (self.filepath + "out_of_africa_n_" +str(self.n)+ "_seed_" + str(
            self.seed))
        with open(filename, 'rb') as inp:
            loaded_data = pickle.load(inp)  # could be changed to self=pickle.load(inp) or similar to save memory.

        # assign properties
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


class ReadVCF:
    def __init__(self, vcf_filename, filepath="", n = [], sites = [], G=[], indv_pop=[],
                 admixture_filename=[]):
        # parameters that need to be set and are not a result of the simulation:
        self.vcf_filename = vcf_filename
        self.filepath = filepath
        self.admixture_filename = vcf_filename

        # properties that are results of data processing:
        self.n = n
        self.sites = sites
        self.G = G
        self.indv_pop = indv_pop

    def load_data(self):
        # load vcf file
        vcf = allel.read_vcf(self.filepath + self.vcf_filename)
        G_haploid = vcf['calldata/GT']
        G = []

        # Reshape the array to combine the inner 2D arrays
        G_haploid = numpy.concatenate(G_haploid, axis=1)

        column_indices = numpy.arange(int(G_haploid.shape[1] / 2)) * 2
        #print("column indices:", column_indices)
        # sum rows and create new numpy ndarray
        G_diploid = G_haploid[:, column_indices] + G_haploid[:, column_indices + 1]
        G_diploid = G_diploid.transpose()
        self.n = G_diploid.shape[1]
        self.sites = G_diploid.shape[0]
        print("n:", self.n)
        print("num sites:", self.sites)

        # assign properties
        G.append(G_diploid)
        self.G = G
        print("G:", self.G)
        self.indv_pop = numpy.zeros(self.n)
        print("indv pop:", self.indv_pop)





use_this_script_for_sim = False
if use_this_script_for_sim == True:
    ## This is the infomation needed in any script that wants to use the data object class:
    n = 15              # sample size
    seed = 42           # starting seed for simulation (based on this seed, multiple
    # seeds will be generated)
    save_G = True       # set True to save genotype matrix during simulation, False otherwise
    print_ts = False     # set True if ts should be printed during simulation, this is
    # only possible if rep==1. For large data sets, this step slows down the program noticeably.
    save_ts = True      # set True to save the tree sequence during simulation, False otherwise
    filepath = "data/with_demography/"
    data_already_simulated = False  # True or False, states if data object should be simulated or loaded

    ## This generates the data object and either simulates the properties or loads if it already exists.
    #data = SimulateOutOfAfrica(n, seed, save_G=save_G,
    #                      print_ts=print_ts, save_ts=save_ts, filepath=filepath)
    # if data_already_simulated == False:
    #     data.sim_data()
    # else:
    #     data.load_data()

    data = ReadVCF('out_of_africa_n_15_seed_42.vcf', 'admixture/data/')
    data.load_data()


    print("final G:", data.G)
    print("final G[0]:", data.G[0][0])
    print("final G shape 0:", data.G[0].shape[0])
    print("final G shape 1:", data.G[0].shape[1])


    print("simulation done.")

