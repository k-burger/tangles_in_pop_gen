import os
import pickle
import allel
import numpy

"""
Script for processing vcf files. First reads the vcf file, preprocesses it and saves 
it in a pickle file. This step can take some time with large data sets. Afterwards, 
the pre-processed vcf file can be loaded directly.
"""


class ReadVCF:
    def __init__(
        self,
        n,
        data_set,
        vcf_filename,
        filepath="",
        sites=[],
        G=[],
        indv_pop=[],
        mutation_id=[],
        admixture_filename=[],
    ):
        # parameters that need to be set and are not a result of the simulation:
        self.n = n  # number of diploid individuals.
        self.data_set = data_set  # type of data_set: 'chr22' or 'AIMs'
        self.vcf_filename = vcf_filename  # filename of considered vcf
        self.filepath = filepath  # filepath to vcf_filename
        # for ADMIXTURE, remove '.vcf'
        self.admixture_filename = os.path.splitext(vcf_filename)[0]

        # properties that are results of data processing:
        self.sites = sites  # number of SNPs
        self.G = G  # genotype matrix
        self.indv_pop = indv_pop  # population membership per individual
        self.mutation_id = mutation_id  # mutation id

    # read and preprocess data:
    def read_vcf(self):
        # read vcf:
        vcf = allel.read_vcf(self.filepath + self.vcf_filename + ".vcf")
        # get haploid genotype matrix:
        G_haploid = vcf["calldata/GT"]
        G = []

        # Reshape the array to combine the inner 2D arrays
        G_haploid = numpy.concatenate(G_haploid, axis=1)

        # sum rows and create new numpy ndarray to create diploid genotype matrix:
        column_indices = numpy.arange(int(G_haploid.shape[1] / 2)) * 2
        G_diploid = G_haploid[:, column_indices] + G_haploid[:, column_indices + 1]
        G_diploid = G_diploid.transpose()

        # assign properties
        self.n = G_diploid.shape[1]
        self.sites = G_diploid.shape[0]
        G.append(G_diploid)
        self.G = G
        self.indv_pop = numpy.zeros(self.n)
        self.mutation_id = vcf["samples"]

        # save object
        filename = (
            self.filepath + "processed_vcf/1000G_n_" + str(self.n) + "_" + self.data_set
        )
        with open(filename, "wb") as outp:  # overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    # load preprocessed vcf file:
    def load_vcf(self):
        # load vcf file
        filename = (
            self.filepath + "processed_vcf/1000G_n_" + str(self.n) + "_" + self.data_set
        )
        with open(filename, "rb") as inp:
            loaded_data = pickle.load(inp)

        # assign properties
        self.n = loaded_data.n
        self.sites = loaded_data.sites
        self.G = loaded_data.G
        self.indv_pop = loaded_data.indv_pop
        self.mutation_id = loaded_data.mutation_id
