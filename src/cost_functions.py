import numpy as np
from sklearn.metrics import pairwise_distances

from sklearn.neighbors import DistanceMetric


def gauss_kernel_distance(xs, n_samples, cut, decay=1):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space
    cut: array of shape [n_points]
        The cut that we are considering
    n_samples: int, optional (default=None)
        The maximums number of points to take per orientation in the Monte Carlo approximation of the order

    Returns
    -------
    expected_order, int
        The average order for the cut
    """

    _, n_features = xs.shape
    if np.all(cut) or np.all(~cut):
        return 0

    if not n_samples:

        in_cut = xs[cut, :]
        out_cut = xs[~cut, :]

    else:

        idx = np.arange(len(xs))

        if n_samples <= len(idx[cut]):
            idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
            in_cut = xs[idx_in, :]
        else:
            in_cut = xs[cut, :]

        if n_samples <= len(idx[~cut]):
            idx_out = np.random.choice(
                idx[~cut], size=n_samples, replace=False)
            out_cut = xs[idx_out, :]
        else:
            out_cut = xs[~cut, :]

    distance = pairwise_distances(in_cut, out_cut, metric='euclidean')
    similarity = np.exp(-distance)
    expected_similarity = np.sum(similarity)

    return expected_similarity


def euclidean_distance(xs, n_samples, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space
    cut: array of shape [n_points]
        The cut that we are considering
    n_samples: int, optional (default=None)
        The maximums number of points to take per orientation in the Monte Carlo approximation of the order

    Returns
    -------
    expected_order, int
        The average order for the cut
    """

    _, n_features = xs.shape
    if np.all(cut) or np.all(~cut):
        return 0

    if not n_samples:

        in_cut = xs[cut, :]
        out_cut = xs[~cut, :]

    else:

        idx = np.arange(len(xs))

        if n_samples <= len(idx[cut]):
            idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
            in_cut = xs[idx_in, :]
        else:
            in_cut = xs[cut, :]

        if n_samples <= len(idx[~cut]):
            idx_out = np.random.choice(
                idx[~cut], size=n_samples, replace=False)
            out_cut = xs[idx_out, :]
        else:
            out_cut = xs[~cut, :]

    distance = -pairwise_distances(in_cut, out_cut, metric='euclidean')
    similarity = distance
    expected_similarity = np.sum(similarity)

    return expected_similarity


def mean_euclidean_distance(xs, n_samples, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space
    cut: array of shape [n_points]
        The cut that we are considering
    n_samples: int, optional (default=None)
        The maximums number of points to take per orientation in the Monte Carlo approximation of the order

    Returns
    -------
    expected_order, int
        The average order for the cut
    """

    _, n_features = xs.shape
    if np.all(cut) or np.all(~cut):
        return 0

    if not n_samples:
        in_cut = xs[cut, :]
        out_cut = xs[~cut, :]

    else:

        idx = np.arange(len(xs))

        if n_samples <= len(idx[cut]):
            idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
            in_cut = xs[idx_in, :]
        else:
            in_cut = xs[cut, :]

        if n_samples <= len(idx[~cut]):
            idx_out = np.random.choice(
                idx[~cut], size=n_samples, replace=False)
            out_cut = xs[idx_out, :]
        else:
            out_cut = xs[~cut, :]

    distance = -pairwise_distances(in_cut, out_cut, metric='euclidean')
    similarity = distance
    expected_similarity = np.mean(similarity)

    return expected_similarity


def manhattan_distance(xs, n_samples, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space
    cut: array of shape [n_points]
        The cut that we are considering
    n_samples: int, optional (default=None)
        The maximums number of points to take per orientation in the Monte Carlo approximation of the order

    Returns
    -------
    expected_order, int
        The average order for the cut
    """

    _, n_features = xs.shape
    if np.all(cut) or np.all(~cut):
        return 0

    if not n_samples:

        in_cut = xs[cut, :]
        out_cut = xs[~cut, :]

    else:

        idx = np.arange(len(xs))

        if n_samples <= len(idx[cut]):
            idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
            in_cut = xs[idx_in, :]
        else:
            in_cut = xs[cut, :]

        if n_samples <= len(idx[~cut]):
            idx_out = np.random.choice(
                idx[~cut], size=n_samples, replace=False)
            out_cut = xs[idx_out, :]
        else:
            out_cut = xs[~cut, :]

    metric = DistanceMetric.get_metric('manhattan')

    distance = metric.pairwise(in_cut, out_cut)
    similarity = 1. / (distance / np.max(distance))
    expected_similarity = np.sum(similarity)

    return expected_similarity


def mean_manhattan_distance(xs, n_samples, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space
    cut: array of shape [n_points]
        The cut that we are considering
    n_samples: int, optional (default=None)
        The maximums number of points to take per orientation in the Monte Carlo approximation of the order

    Returns
    -------
    expected_order, int
        The average order for the cut
    """

    _, n_features = xs.shape
    if np.all(cut) or np.all(~cut):
        return 0

    if not n_samples:
        in_cut = xs[cut, :]
        out_cut = xs[~cut, :]

    else:

        idx = np.arange(len(xs))

        if n_samples <= len(idx[cut]):
            idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
            in_cut = xs[idx_in, :]
        else:
            in_cut = xs[cut, :]

        if n_samples <= len(idx[~cut]):
            idx_out = np.random.choice(
                idx[~cut], size=n_samples, replace=False)
            out_cut = xs[idx_out, :]
        else:
            out_cut = xs[~cut, :]

    metric = DistanceMetric.get_metric('manhattan')
    print("len incut:", len(in_cut))
    print("len incut[0]:", len(in_cut[0]))
    #print("shape incut:", in_cut.shape())
    print("len outcut:", len(out_cut))
    print("len outcut[0]:", len(out_cut[0]))
    #print("shape out:", out_cut.shape())
    distance = metric.pairwise(in_cut, out_cut)
    print("len distance:", len(distance))
    print("len distance[0]:",len(distance[0]))
    similarity = 1. / (distance / np.max(distance))
    expected_similarity = np.mean(similarity)
    return expected_similarity

def mean_fst_distance(xs, n_samples, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space
    cut: array of shape [n_points]
        The cut that we are considering
    n_samples: int, optional (default=None)
        The maximums number of points to take per orientation in the Monte Carlo approximation of the order

    Returns
    -------
    expected_order, int
        The average order for the cut
    """

    _, n_features = xs.shape
    if np.all(cut) or np.all(~cut):
        return 0

    if not n_samples:
        in_cut = xs[cut, :]
        out_cut = xs[~cut, :]

    else:

        idx = np.arange(len(xs))

        if n_samples <= len(idx[cut]):
            idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
            in_cut = xs[idx_in, :]
        else:
            in_cut = xs[cut, :]

        if n_samples <= len(idx[~cut]):
            idx_out = np.random.choice(
                idx[~cut], size=n_samples, replace=False)
            out_cut = xs[idx_out, :]
        else:
            out_cut = xs[~cut, :]

    metric = DistanceMetric.get_metric('manhattan')
    #print("len incut:", len(in_cut))
    #print("len incut[0]:", len(in_cut[0]))
    #print("shape incut:", in_cut.shape())
    #print("len outcut:", len(out_cut))
    #print("len outcut[0]:", len(out_cut[0]))
    #print("shape out:", out_cut.shape())
    distance_in_out = np.mean(metric.pairwise(in_cut, out_cut))
    distance_in = np.mean(metric.pairwise(in_cut, in_cut))
    distance_out = np.mean(metric.pairwise(out_cut, out_cut))
    #total_dist = np.mean(metric.pairwise(xs, xs))
    print("total dist done.")
    #print("len distance:", distance_in)
    #print("len distance[0]:",len(distance_in[0]))
    fst_1 = (distance_in_out - distance_in)/distance_in_out
    #print("fst_1:", fst_1)
    fst_2 = (distance_in_out - distance_out)/distance_in_out
    #print("I am here.")
    #print("fst_2:", fst_2)
    #print("max fst:", np.max([fst_1, fst_2]))
    fst = 1. / (np.sum([fst_1, fst_2]))
    #print("fst:", fst)
    # if fst < 1:
    #     print("fst:", fst)
    #     print("distance in:", distance_in)
    #     print("len in cut:", len(in_cut))
    #     print("len in cut:", len(in_cut[0]))
    #     print("distance out:", distance_out)
    #     print("total dist:", total_dist)
    #     print("fst_1:", fst_1)
    #     print("fst_2:", fst_2)
    #
    # if fst == 1:
    #     print("fst:", fst)
    #     print("distance in:", distance_in)
    #     print("len in cut:", len(in_cut))
    #     print("len in cut:", len(in_cut[0]))
    #     print("distance out:", distance_out)
    #     print("total dist:", total_dist)
    #     print("fst_1:", fst_1)
    #     print("fst_2:", fst_2)
    #similarity = 1. / (distance / np.max(distance))
    #expected_similarity = np.mean(similarity)
    return fst #expected_similarity

def mean_manhattan_distance_weighted_mut_pos(xs, n_samples, mut_pos, f, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space
    cut: array of shape [n_points]
        The cut that we are considering
    n_samples: int, optional (default=None)
        The maximums number of points to take per orientation in the Monte Carlo approximation of the order

    Returns
    -------
    expected_order, int
        The average order for the cut
    """

    _, n_features = xs.shape
    if np.all(cut) or np.all(~cut):
        return 0

    if not n_samples:

        in_cut = xs[cut, :]
        out_cut = xs[~cut, :]

    else:

        idx = np.arange(len(xs))

        if n_samples <= len(idx[cut]):
            idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
            in_cut = xs[idx_in, :]
        else:
            in_cut = xs[cut, :]

        if n_samples <= len(idx[~cut]):
            idx_out = np.random.choice(idx[~cut], size=n_samples, replace=False)
            out_cut = xs[idx_out, :]
        else:
            out_cut = xs[~cut, :]

    metric = DistanceMetric.get_metric('manhattan')
    distance = metric.pairwise(in_cut, out_cut)

    n = len(cut)
    sigma = f(n)

    #print("cut:", cut)
    #print("non cut:", ~cut)
    #print("mut position:", mut_pos)

    in_cut_pos = mut_pos[cut]
    #print("in_cut_idx:", in_cut_pos)
    out_cut_pos = mut_pos[~cut]
    #print("out_cut_idx:", out_cut_pos)

    distance_weights = np.exp(- np.array([[np.abs(i - j) for i in out_cut_pos] for j in in_cut_pos]) / (2*sigma))

    #distance_weights = 1
    distance = np.multiply(distance, distance_weights)

    similarity = 1. / (distance / np.max(distance))
    expected_similarity = np.mean(similarity)

    return expected_similarity



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


def edges_cut_cost(A, n_samples, cut):
    """
    Compute the value of a graph cut, i.e. the number of vertex that are cut by the bipartition

    Parameters
    ----------
    n_samples
    A: array of shape [nb_vertices, nb_vertices]
        Adjacency matrix for our graph
    cut: array of shape [n_points]
        The cut that we are considering

    Returns
    -------
    order: int
        order of the cut
    """

    partition = np.where(cut == True)[0]
    comp = np.where(cut == False)[0]

    values = A[np.ix_(partition, comp)].reshape(-1)

    if not n_samples:
        order = np.sum(values)
    else:
        if len(values) > int(n_samples):
            values = np.random.choice(values, n_samples)
        order = np.sum(values)

    return order


def mean_edges_cut_cost(A, n_samples, cut):
    """
    Compute the value of a graph cut, i.e. the number of vertex that are cut by the bipartition

    Parameters
    ----------
    n_samples
    A: array of shape [nb_vertices, nb_vertices]
        Adjacency matrix for our graph
    cut: array of shape [n_points]
        The cut that we are considering

    Returns
    -------
    order: int
        order of the cut
    """

    partition = np.where(cut == True)[0]
    comp = np.where(cut == False)[0]

    values = A[np.ix_(partition, comp)].reshape(-1)

    if not n_samples:
        order = np.mean(values)
    else:
        if len(values) > int(n_samples):
            values = np.random.choice(values, n_samples)
        order = np.mean(values)

    return order
