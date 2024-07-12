import pickle
import warnings
from src import customized_demesdraw
import matplotlib.pyplot as plt
import msprime
import numpy
from IPython.display import SVG

"""
Script to simulate data with an underlying demographic structure (8 populations) as 
shown in the publication. The resulting genotype matrix is saved in an vcf file. 
"""


class Simulated_Data_With_Demography:
    def __init__(self, n_diploid, theta, rho, migration_rate_multi, seed,
                 save_G=True, print_ts=False, save_ts=True, filepath="", SFS=[],
                 total_tree_length=[], G=[], true_thetas=[], true_rhos=[],
                 ts=[], indv_pop=[], vcf_filename=[]):
        # parameters that need to be set and are not a result of the simulation:
        self.n = n_diploid
        self.rep = 1
        self.theta = theta
        self.rho = rho
        self.migration_rate_multi = migration_rate_multi
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
        self.vcf_filename = vcf_filename

    # function to simulate data with underlying demographic structure:
    def sim_data(self):
        # n_haploid = 2*self.n    # get haploid sample size
        multi_SFS = []  # list to save the SFS
        multi_total_length = []  # list to save the total tree lengths
        multi_G = []  # list to save the genotype matrices
        multi_theta = []  # list to save theta used for simulating
        multi_rho = []  # list to save rho used for simulating
        multi_ts = []  # list to save the tree sequences

        num_mutations = []
        tree_length = []
        num_trees = []
        seed_vec = list(range(int(self.seed + 2), int(self.seed + self.rep + 2)))

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
            theta = numpy.array(numpy.random.uniform(0, 100, self.rep))
            multi_theta = theta
        else:
            theta = self.theta * numpy.ones(self.rep)
            multi_theta = theta

        if rho_train:
            numpy.random.seed(self.seed - 2)
            rho = numpy.array(numpy.random.uniform(0, 50, self.rep))
            multi_rho = rho
        else:
            rho = self.rho * numpy.ones(self.rep)
            multi_rho = rho

        # when using with migration between all population A to H, set scale to 8:
        scale = 8

        ## define demography for 8 populations:
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=0.25*scale)
        demography.add_population(name="B", initial_size=0.25*scale)
        demography.add_population(name="C", initial_size=0.25*scale)
        demography.add_population(name="D", initial_size=0.25*scale)
        demography.add_population(name="E", initial_size=0.25*scale)
        demography.add_population(name="F", initial_size=0.25*scale)
        demography.add_population(name="G", initial_size=0.25*scale)
        demography.add_population(name="H", initial_size=0.25*scale)
        demography.add_population(name="AB", initial_size=0.25*scale)
        demography.add_population(name="CD", initial_size=0.25*scale)
        demography.add_population(name="ABCD", initial_size=0.25*scale)
        demography.add_population(name="EF", initial_size=0.25*scale)
        demography.add_population(name="EFG", initial_size=0.25*scale)
        demography.add_population(name="EFGH", initial_size=0.25*scale)
        demography.add_population(name="ABCDEFGH", initial_size=0.25*scale)

        # this parameter rescales the coalescent times: c=7 to achieve
        # well-differentiated populations, c=70 for significant incomplete lineage
        # sorting:
        c = 7/scale

        # define population splits and times:
        demography.add_population_split(time=2 / c, derived=["A", "B"], ancestral="AB")
        demography.add_population_split(time=4 / c, derived=["E", "F"], ancestral="EF")
        demography.add_population_split(time=6 / c, derived=["C", "D"], ancestral="CD")
        demography.add_population_split(time=8 / c, derived=["EF", "G"],
                                        ancestral="EFG")
        demography.add_population_split(time=10 / c, derived=["AB", "CD"],
                                        ancestral="ABCD")
        demography.add_population_split(time=12 / c, derived=["EFG", "H"],
                                        ancestral="EFGH")
        demography.add_population_split(time=28 / (2 * c), derived=["ABCD", "EFGH"],
                                        ancestral="ABCDEFGH")

        # set migration between populations A ato H:
        demography.set_symmetric_migration_rate(["A", "E"],
                                                self.migration_rate_multi*0.5/scale)
        demography.set_symmetric_migration_rate(["A", "B"],
                                                self.migration_rate_multi*0.5/scale)
        demography.set_symmetric_migration_rate(["D", "C"],
                                                self.migration_rate_multi*0.5/scale)
        demography.set_symmetric_migration_rate(["C", "B"],
                                                self.migration_rate_multi*0.5/scale)
        demography.set_symmetric_migration_rate(["E", "F"],
                                                self.migration_rate_multi*0.5/scale)
        demography.set_symmetric_migration_rate(["F", "G"],
                                                self.migration_rate_multi*0.5/scale)
        demography.set_symmetric_migration_rate(["G", "H"],
                                                self.migration_rate_multi*0.5/scale)

        ## plot demographic structure via demesdraw.tubes with customized colors and
        # demes positions:
        graph = msprime.Demography.to_demes(demography)
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = ['#029e73', '#0173b2', '#de8f05', '#d55e00', '#56b4e9', '#949494',
                '#cc78bc', '#fbafe4', '#ece133', '#ca9161', '#004949', '#920000',
                '#924900', '#490092', '#b66dff']
        cmap2 = ['#4dbb9d', '#99d8c7', '#4d9dc9', '#e7b050', '#e18e4c', '#88caef', '#b2d5e7']
        demes_colors = {'A': cmap[6], 'B': cmap[3], 'C': cmap[5],
                        'D': cmap[0], 'E': cmap[7], 'F': cmap[4],
                        'G': cmap[2], 'H': cmap[1], 'AB': cmap2[4],
                        'CD': cmap2[0], 'EF': cmap2[5],
                        'EFG': cmap2[3], 'ABCD': cmap2[1],
                        'EFGH': cmap2[2], 'ABCDEFGH': cmap2[6]}
        positions = {'ABCDEFGH': 11 / 5, 'ABCD': 5.5 / 5, 'EFGH': 19.375 / 5,
                     'H': 22 / 5, 'EFG': 16.75 / 5, 'CD': 2.5 / 5, 'AB': 8.5 / 5,
                     'G': 19 / 5, 'EF': 14.5 / 5, 'B': 7 / 5, 'A': 10 / 5, 'E': 13 / 5,
                     'F': 16 / 5, 'D': 1 / 5, 'C': 4 / 5}
        # set number of migration lines between populations with migraion between
        # populations A to H:
        migration_lines = numpy.array([4,4,3,3,2,2,2,2,1,1,1,1,1,1])

        customized_demesdraw.tubes(graph, positions=positions,
                                   num_lines_per_migration=migration_lines,
                                   seed=19,
                                   colours=demes_colors, fill=True)
        filename_short = ("plots/demographic_structure_migration_A_H")
        plt.tick_params(axis='y', labelsize=15)
        plt.tick_params(axis='x', labelsize=15)
        #plt.axis('off')
        plt.gca().spines['left'].set_visible(False)
        plt.yticks([])  # y-Achsenticks entfernen
        plt.ylabel('')

        # plt.ylabel('time ago (generations)', fontsize=14)
        plt.savefig(filename_short + ".jpeg", format='jpeg', dpi=300)
        plt.close()
        # show saved image:
        image = plt.imread(filename_short + ".jpeg")
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        # set constant population size for 8 simulated populations:
        size = self.n // 8

        # simulate data with the specified demographic structure:
        for i in range(0, self.rep):
            # set seed
            rng = numpy.random.default_rng(seed_vec[i])
            seeds = rng.integers(1, 2 ** 31 - 1, size=2)
            # simulate tree sequence:
            ts = msprime.sim_ancestry(
                samples={"A": size, "B": size, "C": size, "D": size, "E": size,
                         "F": size, "G": size, "H": size}, sequence_length=1,
                discrete_genome=False, recombination_rate=rho[i] / scale,
                random_seed=seeds[0], demography=demography, ploidy=2)
            # simulate mutations:
            tree_sequence = msprime.sim_mutations(ts, rate=theta[i] / scale,
                                                  random_seed=seeds[1],
                                                  discrete_genome=False)
            # print("ts.individuals_population:", tree_sequence.individuals_population)
            # print("num_migration:", tree_sequence.num_migrations)

            ## safe as vcf file:
            # change indv_names to work with vcf file:
            indv_names = [f"{i}indv" for i in range(self.n)]
            # print("indv_names:", indv_names)
            # safe tree_sequence in vfc file:
            vcf_filename = ("n_" + str(self.n) + "_rep_" + str(
                self.rep) + "_rho_" + rho_str + "_theta_" + theta_str + "_seed_" + str(
                self.seed))
            with open("data/vcf/" + vcf_filename + ".vcf", "w") as vcf_file:
                tree_sequence.write_vcf(vcf_file, individual_names=indv_names)

            # save tree_sequence if specified:
            if self.save_ts:
                multi_ts.append(tree_sequence)

            ## compute properties of simulation as genotype matrix, SFS,... and save
            # them:
            num_mutations.append(tree_sequence.num_mutations)  # nb of mutations
            print("num_mutations:", num_mutations)
            # tree length and number of trees:
            m = 0
            for tree in tree_sequence.trees():
                m = m + 1
                tree_length.append(tree.total_branch_length)
            num_trees.append(m)
            # print("num trees:", num_trees)
            # print("tree length:", tree_length)

            # get mean total tree length and save as entry of multi_total_length
            mean_tot_branch_length = 0
            for tree in tree_sequence.trees():
                mean_tot_branch_length += tree.total_branch_length * (
                        tree.interval[1] - tree.interval[0])
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

        # assign properties
        self.true_thetas = multi_theta
        self.G = multi_G
        self.SFS = multi_SFS
        self.total_tree_length = multi_total_length
        self.true_rhos = multi_rho
        self.ts = multi_ts
        self.indv_pop = tree_sequence.individuals_population
        self.vcf_filename = vcf_filename

        # save object
        filename = (self.filepath + "sim_with_demography_n_" + str(
            self.n) + "_rep_" + str(
            self.rep) + "_rho_" + rho_str + "_theta_" + theta_str + "_seed_" + str(
            self.seed))
        with open(filename, 'wb') as outp:  # overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

        # print tree sequence:
        if self.print_ts and self.rep == 1:
            filename_svg = filename + ".svg"
            # SVG(tree_sequence.draw_svg(path=filename_svg, x_axis=True, y_axis=True, symbol_size=5, y_label=[]))
            # for larger n, use the following instead of the SVG print command above:
            wide_fmt = (3000,
                        250)  # this sets the format of the plot, for smaller/larger n decrease/increase first entry (x-axis).
            SVG(tree_sequence.draw_svg(path=filename_svg, x_axis=True, y_axis=True,
                                       time_scale="rank",
                                       # used time_scale="rank" for better visualization of large tree sequences
                                       symbol_size=5, y_label=[], size=wide_fmt))
        elif self.print_ts and self.rep > 1:
            warnings.warn(
                'you try to print the ts but rep>1. only print svg if save_svg==True and rep==1.',
                stacklevel=1)

    # function to load data with underlying demographic structure that has been
    # simulated already:
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
        filename = (self.filepath + "sim_with_demography_n_" + str(
            self.n) + "_rep_" + str(
            self.rep) + "_rho_" + rho_str + "_theta_" + theta_str + "_seed_" + str(
            self.seed))
        with open(filename, 'rb') as inp:
            loaded_data = pickle.load(
                inp)  # could be changed to self=pickle.load(inp) or similar to save memory.

        # assign properties
        self.seed = loaded_data.seed
        self.true_thetas = loaded_data.true_thetas
        self.G = loaded_data.G
        self.SFS = loaded_data.SFS
        self.total_tree_length = loaded_data.total_tree_length
        self.true_rhos = loaded_data.true_rhos
        self.ts = loaded_data.ts
        self.indv_pop = loaded_data.indv_pop
        self.vcf_filename = loaded_data.vcf_filename

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
    n = 800  # sample size
    theta = 100  # theta=int for constant theta in rep simulations,
    # theta='rand' for random theta in (0,100) in every simulation
    rho = 100  # rho=int for constant theta in rep simulations, rho='rand'
    # sclaling factore for migration rate:
    migration_rate_multi = 4
    # for random theta in (0,100) in every simulation
    seed = 42  # starting seed for simulation (based on this seed, multiple
    # seeds will be generated)
    filepath = "data/with_demography/"
    data_already_simulated = False  # True or False, states if data object should be simulated or loaded

    ## This generates the data object and either simulates the properties or loads if it already exists.
    data = Simulated_Data_With_Demography(n, theta, rho, migration_rate_multi, seed,
                                          filepath=filepath)
    if data_already_simulated == False:
        data.sim_data()
    else:
        data.load_data()

    #print("G:", data.G)

    # print("simulation done.")