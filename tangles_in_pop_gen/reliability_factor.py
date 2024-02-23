import numpy as np

"""
Script to compute reliability factors for each cut which are used in the soft 
clustering.
"""


def compute_reliability(xs):
    # get number of individuals and mutations of genotype matrix xs:
    nb_indv, nb_mut = xs.shape
    reliability = np.zeros((nb_mut, 2))

    # iterate through cuts:
    for i in range(0, nb_mut):
        N0 = np.count_nonzero(xs[:, i] == 0)  # nb of indv with 2 ancestral alleles
        N1 = np.count_nonzero(xs[:, i] == 1)  # nb of heterozygous indv
        N2 = np.count_nonzero(xs[:, i] == 2)  # nb of indv with 2 derived alleles

        # initial estimates of the allele frequencies:
        p1 = (1 / (2 * (N1 + N2))) * (N1 + 2 * N2)
        p2 = (1 / (2 * (N0 + N1))) * N1
        iteration_count_1 = 0  # init count of updates for p1
        iteration_count_2 = 0
        # stopping criterion: if the change in p1 or p2 is less than tol, then sop:
        tol = 0.00001

        if p1 == 1:  # reliable cut as every indv is homozygous for corresponding SNP
            reliability_T = 1
            reliability_F = 1
        else:
            p1_iteration_before = 0  # to enter first update loop
            p2_iteration_before = 0  # to enter first update loop
            # iterative procedure to approximate true underlying allele frequency p1,
            # stop if estimates of allele frequency change less than tol:
            while np.absolute(p1 - p1_iteration_before) > tol:
                p1_iteration_before = p1
                # comp expected number of zeros in regard of Hardy-Weinberg eq.:
                exp_0 = np.power((1 - p1), 2) * ((N1 + N2) / (1 - np.power(1 - p1, 2)))
                # take them into account to get new estimate of the allele frequency:
                p1 = (1 / (2 * (N1 + N2 + exp_0))) * (N1 + 2 * N2)
                iteration_count_1 += 1  # iteration count

            # iterative procedure to approximate true underlying allele frequency p2,
            # stop if estimates of allele frequency change less than tol:
            while np.absolute(p2 - p2_iteration_before) > tol:
                p2_iteration_before = p2
                # comp expected number of ones & twos in regard of Hardy-Weinberg eq.:
                exp_1 = 2 * p2 * (1 - p2) * N0 / np.power(1 - p2, 2)
                exp_2 = np.power(p2, 2) * N0 / (np.power(1 - p2, 2))
                # take them into account to get new estimate of the allele frequency:
                p2 = (1 / (2 * (N0 + exp_1 + exp_2))) * (exp_1 + 2 * exp_2)
                iteration_count_2 += 1  # iteration count

            # compute reliability factor for both sides of the cut:
            reliability_T = np.maximum(
                1 - ((np.minimum(exp_1, N1) + np.minimum(exp_2, N2)) / (N1 + N2)), 0)
            reliability_F = np.maximum(1 - (exp_0 / N0), 0)

        # save computed reliability factor for each cut:
        reliability[i, 0] = reliability_T
        reliability[i, 1] = reliability_F
    return reliability
