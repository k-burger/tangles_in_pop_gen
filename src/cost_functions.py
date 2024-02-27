import numpy as np
from sklearn.neighbors import DistanceMetric
import pickle
import sys
sys.path.append('..')

"""
Script with 3 different cost functions, all specific to population genetics.

    1. k_nearest_neighbours: can be used as cost function, but is used in the 
    publication as a function for calculating the kNN normalization factor and is 
    combined with the two other cost functions
    2. FST_kNN: cost function based on mean FST values combined with kNN
    3. HWE_kNN: cost function based on the divergence of Hardy-Weinberg equilibrium 
    and combined with kNN.
"""


# this function computes for each cut the fraction of neighbours that lie on the other
# side of the cut. This serves as a normalization factor as cuts which cut through
# closely related individuals will be penalized.
def k_nearest_neighbours(xs, n_sample, cut):
    # compute number of individuals on both sides of the cut:
    len_in_cut = np.count_nonzero(cut == True)
    len_out_cut = np.count_nonzero(cut == False)

    # lead kNN matrix:
    with open("data/saved_kNN/kNN", 'rb') as inp:
        kNN = pickle.load(inp)

    # Find indices of individuals on each side of the cut
    mutation_indices_False = np.where(cut == False)[0]
    mutation_indices_True = np.where(cut == True)[0]

    # Find neighbors for individuals that do not carry the mutation responsible for
    # the cut
    neighbors_False = np.where(kNN.kNN[mutation_indices_False])[1]
    # Find neighbors for individuals that do carry the mutation responsible for the cut
    neighbors_True = np.where(kNN.kNN[mutation_indices_True])[1]
    # Count neighbors with opposite mutation status
    kNN_on_opposite_side_False = np.sum(cut[neighbors_False] != False)
    kNN_on_opposite_side_True = np.sum(cut[neighbors_True] != True)
    # compute fraction of neighbours that lie on the other side of the cut:
    knn_breaches = (1 + ((kNN_on_opposite_side_False) /
                           (len_out_cut * kNN.k) + (
                                       kNN_on_opposite_side_True) / (
                                       len_in_cut * kNN.k)))
    return knn_breaches

# cost function based on mean FST-values:
def FST_kNN(xs, n_samples, cut):
    # compute number of individuals on both sides of the cut:
    len_in_cut = np.count_nonzero(cut == True)
    len_out_cut = np.count_nonzero(cut == False)
    len_xs = len_in_cut + len_out_cut  # total number of individuals

    # allele frequencies on both sides of the cut:
    p_in = (1 / (2 * len_in_cut)) * np.sum(xs * cut[:, np.newaxis], axis=0)
    p_out = (1 / (2 * len_out_cut)) * np.sum(xs * (~cut)[:, np.newaxis], axis=0)
    # global allele frequency:
    p = ((len_in_cut / len_xs) * p_in) + ((len_out_cut / len_xs) * p_out)

    # mean FST value on both sides of the cut:
    F_in = np.abs(1 - ((p_in * (1 - p_in)) / (p * (1 - p))))
    F_out = np.abs(1 - ((p_out * (1 - p_out)) / (p * (1 - p))))

    # mean FST_value:
    FST = 0.5 * ((np.sum(F_in) / xs.shape[1]) + (np.sum(F_out) / xs.shape[1]))

    # normalization factor that penalizes unbalanced cuts:
    norm_balanced_cuts = np.power((len_xs / len_in_cut) + (len_xs / len_out_cut), 0.05)
    # load pre-computed kNN penalization:
    kNN = k_nearest_neighbours(xs, n_samples, cut)

    # penalize unbalanced cuts and cuts cutting through similar individuals. Take the
    # reciprocal of FST as cuts with a low cost are prioritized and higher FST values
    # indicate a better cut:
    normalized_FST = (1 / FST) * np.power(kNN, 1) * np.power(norm_balanced_cuts, 1)
    return normalized_FST


# cost function based on Hardy-Weinberg equilibrium:
def HWE_kNN(xs, n_samples, cut):
    # compute number of individuals on both sides of the cut:
    len_in_cut = np.count_nonzero(cut == True)
    len_out_cut = np.count_nonzero(cut == False)
    len_xs = len_in_cut + len_out_cut  # total number of individuals

    # allele frequencies on both sides of the cut:
    p_in = (1 / (2 * len_in_cut)) * np.sum(xs * cut[:, np.newaxis], axis=0)
    p_out = (1 / (2 * len_out_cut)) * np.sum(xs * (~cut)[:, np.newaxis], axis=0)

    # expected heterozygosity on both sides of the cut:
    expected_H_in = 2 * p_in * (1 - p_in)
    expected_H_out = 2 * p_out * (1 - p_out)

    # observed heterozygotes on both sides of the cut:
    x_in = (1 / len_in_cut) * np.count_nonzero(xs * cut[:, np.newaxis] == 1, axis=0)
    x_out = (1 / len_out_cut) * np.count_nonzero(xs * (~cut)[:, np.newaxis] == 1,
                                                 axis=0)

    # penalize random cuts:
    normalization = np.sum(np.sqrt(np.abs(p_in - p_out)))

    # mean divergence from Hardy-Weinberg equilibrium over all SNPs:
    F_in = np.abs(x_in - expected_H_in)
    F_out = np.abs(x_out - expected_H_out)
    HWE_div = ((1 + np.minimum(np.sum(F_in), np.sum(F_out))) / (normalization))

    # normalization factor that penalizes unbalanced cuts:
    norm_balanced_cuts = np.power((len_xs / len_in_cut) + (len_xs / len_out_cut), 0.1)
    # load pre-computed kNN penalization:
    kNN = k_nearest_neighbours(xs, n_samples, cut)

    # normalized HWE:
    normalized_HWE = HWE_div * np.power(kNN, 2) * np.power(norm_balanced_cuts, 1)

    return normalized_HWE


class BipartitionSimilarity():
    def __init__(self, all_bipartitions: np.ndarray) -> None:
        """
        Computes the cost cost of a bipartition according to Klepper et. al 2020,
        "Clustering With Tangles Algorithmic Framework"; p.9
        https://arxiv.org/abs/2006.14444

        This class is intended for repeated cost calculation, which is done in an efficient manner
        by precomputing all distances between bipartitions
        in a first step and then computing the cost of a bipartition by summing the respective
        distances.

        all_bipartitions: np.ndarray of shape (datapoints, questions),
            containing all possible bipartitions.
        """
        metric = DistanceMetric.get_metric('manhattan')
        self.dists = metric.pairwise(all_bipartitions)

    def __call__(self, bipartition: np.ndarray):
        """
        bipartitions: np.ndarray of shape (datapoints,) consisting of booleans.
            In the questionnaire scenario, this is corresponding to one
            column (question), filled out by all participants
        """
        if np.all(bipartition) or np.all(~bipartition):
            # Should this be 0 or inf?
            return np.inf
        in_cut = np.where(bipartition)[0]
        out_cut = np.where(~bipartition)[0]

        distance = self.dists[np.ix_(in_cut, out_cut)]
        similarity = 1. / (distance / np.max(distance))
        expected_similarity = np.mean(similarity)
        return expected_similarity