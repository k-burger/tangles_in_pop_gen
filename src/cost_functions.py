import itertools

import numpy as np
from sklearn.metrics import pairwise_distances

from sklearn.neighbors import DistanceMetric
import time


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


def HWE_divergence(xs, n_samples, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space

    Returns
    -------
    distances, [double, double]
        All pairwise distances
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
    #print("xs:", xs)
    #print("len(xs):", len(xs))
    #print("in cut:", in_cut)
    #print("len(in):", len(in_cut))
    #print("out cut:", out_cut)
    #print("len(out):", len(out_cut))
    #print("num mutations:", xs.shape[1])

    F_in = []
    F_out = []
    normalization = 0
    for m in range(0, xs.shape[1]):
        #print("########## mutation ", m, " ##########")
        #print("in cut:", in_cut)
        #print("out cut:", out_cut)
        p_in = (1 / (2 * len(in_cut))) * np.sum(in_cut[:,m])
        p_out = (1 / (2 * len(out_cut))) * np.sum(out_cut[:,m])
        #print("p_in:", p_in)
        #print("p_out:", p_out)
        expected_H_in = 2*p_in*(1-p_in)
        expected_H_out = 2 * p_out * (1 - p_out)
        #print("expected_H_in:", expected_H_in)
        #print("expected_H_out:", expected_H_out)
        x_in = (1 /len(in_cut)) * np.count_nonzero(in_cut[:,m]==1)
        x_out = (1 /len(out_cut)) * np.count_nonzero(out_cut[:,m]==1)
        #print("x_in:", x_in)
        #print("x_out:", x_out)
        F_in.append(pow(x_in - expected_H_in, 2))
        F_out.append(pow(x_out - expected_H_out, 2))
        #print("F_in:", F_in)
        #print("F_out:", F_out)
        normalization += pow(p_in -p_out, 2)
        #print("normalization:", normalization)
    print("normalization:", normalization)
    normalization = normalization/xs.shape[1]
    print("normalization mean:", normalization)
    #print("final normalization:", normalization)
    HWE_div = (sum(F_in) + sum(F_out))/(normalization*xs.shape[1])
    print("HWE_div:", HWE_div)
    return HWE_div


def HWE(xs, n_samples, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space

    Returns
    -------
    distances, [double, double]
        All pairwise distances
    """

    len_in_cut = np.count_nonzero(cut == True)
    len_out_cut = np.count_nonzero(cut == False)

    p_in = (1 / (2 * len_in_cut)) * np.sum(xs * cut[:, np.newaxis], axis=0)
    p_out = (1 / (2 * len_out_cut)) * np.sum(xs * (~cut)[:, np.newaxis], axis=0)
    expected_H_in = 2 * p_in * (1 - p_in)
    expected_H_out = 2 * p_out * (1 - p_out)
    x_in = (1 / len_in_cut) * np.count_nonzero(xs * cut[:, np.newaxis] == 1, axis=0)
    x_out = (1 / len_out_cut) * np.count_nonzero(xs * (~cut)[:, np.newaxis] == 1, axis=0)


    F_in = np.power(x_in - expected_H_in, 2)
    F_out = np.power(x_out - expected_H_out, 2)
    normalization = np.sum(np.power(p_in - p_out, 2))
    HWE_div = (np.sum(F_in) + np.sum(F_out)) / normalization
    return HWE_div






def normalized_mean_distances(distances, cut):
    idx = np.arange(len(cut))
    in_cut = idx[cut]
    out_cut = idx[~cut]

    idxs = list(itertools.product(in_cut, out_cut))

    distance = np.array([distances[idx] for idx in idxs])

    similarity = 1. / (distance / np.max(distance))
    return np.mean(similarity)


def all_pairs_manhattan_distance(xs):
    """
    This function computes all pairwise distances.
    """

    return pairwise_distances(xs, xs, metric='manhattan', n_jobs=-1)


def FST_observed(xs, n_samples, cut):
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

    F_in = []
    F_out = []
    #print("len(xs):", len(xs))
    count = 0
    for m in range(0, xs.shape[1]):
        #print("########## mutation ", m, " ##########")
        x = (1 / len(xs)) * np.count_nonzero(xs[:, m] == 1)
        x_in = (1 /len(in_cut)) * np.count_nonzero(in_cut[:,m]==1)
        x_out = (1 /len(out_cut)) * np.count_nonzero(out_cut[:,m]==1)
        #print("x:", x)
        #print("x_in:", x_in)
        #print("x_out:", x_out)
        if x == 0:
            F_in.append(1)
            F_out.append(1)
            count += 1
        else:
            F_in.append(x_in/x)
            F_out.append(x_out/x)
        #print("F_in:", F_in)
        #print("F_out:", F_out)
    FST_obs = 1/(0.5*(np.abs((1-(sum(F_in)/xs.shape[1]))) + np.abs((1-(sum(
        F_out)/xs.shape[
        1])))))
    print("FST_obs:", FST_obs)
    if count >0:
        print("observed x=0:", count)
    return FST_obs

def FST_expected(xs, n_samples, cut):
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
    #_, n_features = xs.shape
    #idx = np.arange(len(cut))
    in_cut = xs[cut, :]
    out_cut = xs[~cut, :]

    F_in = []
    F_out = []
    for m in range(0, xs.shape[1]):
        # print("########## mutation ", m, " ##########")
        p_in = (1 / (2 * len(in_cut))) * np.sum(in_cut[:, m])
        p_out = (1 / (2 * len(out_cut))) * np.sum(out_cut[:, m])
        p = ((len(in_cut) / len(xs)) * p_in) + ((len(out_cut) / len(xs)) * p_out)
        # if p == 0:
        #     print("FAIL!!!")
        # if p_in == 0:
        #     print("p_in FAIL!!!")
        # if p_out == 0:
        #     print("p_out FAIL!!!")
        F_in.append(np.abs(1 - ((p_in * (1-p_in))/(p*(1-p)))))
        # if np.isnan(np.abs(1 - ((p_in * (1-p_in))/(p*(1-p))))):
        #     print("p_in Fail:", p_in)
        #     print("p_out:", p_out)
        #     print(xs[:,m])
        #     print("p Fail:", p)
        F_out.append(np.abs(1 - ((p_out * (1-p_out))/(p*(1-p)))))
        # if np.isnan(np.abs(1 - ((p_out * (1-p_out))/(p*(1-p))))):
        #     print("p_out Fail:", p_out)
        #     print("p_in :", p_in)
        #     print(xs[:, m])
        #     print("p Fail:", p)
        #print("F_in:", F_in)
        #print("F_out:", F_out)

    FST_exp = 0.5*((np.sum(F_in)/xs.shape[1]) + (np.sum(F_out)/xs.shape[1]))
    # print("FST_exp:", FST_exp)
    # p_in = (1 / (2 * len(in_cut) * xs.shape[1])) * np.sum(in_cut)
    # p_out = (1 / (2 * len(out_cut) * xs.shape[1])) * np.sum(out_cut)
    # p = ((len(in_cut) / len(xs)) * p_in) + ((len(out_cut) / len(xs)) * p_out)
    # print("p_in:", p_in)
    # print("p_out:", p_out)
    # print("p:", p)
    # F_in = 1 - ((p_in * (1-p_in))/(p*(1-p)))
    # F_out = 1 - ((p_out * (1-p_out))/(p*(1-p)))
    # FST_exp = np.maximum(np.abs(F_in),np.abs(F_out))
    # print("F_in:", F_in)
    # print("F_out:", F_out)
    #print("FST:", FST_exp)
    #print("FST_exp:", 1/FST_exp)
    return 1/FST_exp

def calculate_in_cut_out_cut_parallel(xs, cut):
    in_cut = xs[cut, :]
    out_cut = xs[~cut, :]
    return in_cut, out_cut

def FST(xs, n_samples, cut):
    #print("time needed for seperation:", inter_time - time1 ) #time1 - start
    #print("in_cut rows:", in_cut.shape[0])
    #print("in_cut columns:", in_cut.shape[1])
    #print("out_cut rows:", out_cut.shape[0])
    #print("out_cut columns:", out_cut.shape[1])
    #print("in_cut:", in_cut)
    #print("xs in cost:", xs)
    start_time = time.time()

    len_in_cut = np.count_nonzero(cut == True)
    len_out_cut = np.count_nonzero(cut == False)
    len_xs = len_in_cut + len_out_cut
    #print("True 2:", len_in_cut)
    #print("False 2:", len_out_cut)

    p_in = (1 / (2 * len_in_cut)) * np.sum(xs * cut[:, np.newaxis], axis=0)
    p_out = (1 / (2 * len_out_cut)) * np.sum(xs * (~cut)[:, np.newaxis], axis=0)
    p = ((len_in_cut / len_xs) * p_in) + ((len_out_cut / len_xs) * p_out)

    #print("p_in:", p_in, ((p_in * (1 - p_in))))
    #print("p_out:", p_out, (p_out * (1 - p_out)))
    #print("p:", p, (p * (1 - p)))

    F_in = np.abs(1 - ((p_in * (1 - p_in)) / (p * (1 - p))))
    F_out = np.abs(1 - ((p_out * (1 - p_out)) / (p * (1 - p))))

    FST_exp = 0.5 * ((np.sum(F_in) / xs.shape[1]) + (np.sum(F_out) / xs.shape[1]))
    #print("FST exp:", FST_exp)
    end_time = time.time()

    #print("time needed for cost calculation:", end_time - start_time)
    return 1/FST_exp

def FST_expected_fast_old(xs, n_samples, cut):
    #print("cut:", cut)
    #print("len cut cost fct:", len(cut))
    # start = time.time()


    time1 = time.time()
    in_cut = xs[cut, :]
    out_cut = xs[~cut, :]

    #cut_indices = np.where(cut)[0]
    #in_cut = xs[cut_indices]
    #out_cut = np.delete(xs, cut_indices, axis=0)
    inter_time = time.time()
    print("time needed for seperation:", inter_time - time1 ) #time1 - start
    #print("in_cut rows:", in_cut.shape[0])
    #print("in_cut columns:", in_cut.shape[1])
    #print("out_cut rows:", out_cut.shape[0])
    #print("out_cut columns:", out_cut.shape[1])
    #print("in_cut:", in_cut)
    #print("xs in cost:", xs)

    len_xs = len(xs)
    len_in_cut = len(in_cut)
    len_out_cut = len(out_cut)
    print("True:", len_in_cut)
    print("False:", len_out_cut)
    if len_out_cut == 0:
        print("FAIL! len out cut zero")
        print(len_out_cut)
        print(len_in_cut)

    p_in = (1 / (2 * len_in_cut)) * np.sum(in_cut, axis=0)
    p_out = (1 / (2 * len_out_cut)) * np.sum(out_cut, axis=0)

    p = ((len_in_cut / len_xs) * p_in) + ((len_out_cut / len_xs) * p_out)

    F_in = np.abs(1 - ((p_in * (1 - p_in)) / (p * (1 - p))))
    F_out = np.abs(1 - ((p_out * (1 - p_out)) / (p * (1 - p))))

    FST_exp = 0.5 * ((np.sum(F_in) / xs.shape[1]) + (np.sum(F_out) / xs.shape[1]))
    #print("FST exp:", FST_exp)
    end_time = time.time()
    print("time needed for rest:", end_time - inter_time)
    return 1 / FST_exp

def FST_wikipedia_fast(xs, n_samples, cut):
    #print("cut:", cut)
    #print("len cut cost fct:", len(cut))
    in_cut = xs[cut, :]
    out_cut = xs[~cut, :]
    #print("in_cut rows:", in_cut.shape[0])
    #print("in_cut columns:", in_cut.shape[1])
    #print("out_cut rows:", out_cut.shape[0])
    #print("out_cut columns:", out_cut.shape[1])
    #print("in_cut:", in_cut)
    #print("xs in cost:", xs)

    n = len(xs)
    print("n:", n)
    n_in = len(in_cut)
    print("n_in:", n_in)
    n_out = len(out_cut)
    print("n_out:", n_out)
    p_in = (1 / (2 * n_in)) * np.sum(in_cut, axis=0)
    p_out = (1 / (2 * n_out)) * np.sum(out_cut, axis=0)
    p = ((n_in / n) * p_in) + ((n_out / n) * p_out)
    print("p_in:", p_in)
    print("p_out:", p_out)
    print("p:", p)

    FST = (p * (1 - p) - (n_in/n)*(p_in * (1 - p_in)) - (n_out/n)*(p_out * (
            1-p_out)))/(p * (1 - p))
    print("len FST:", len(FST))

    FST_exp = np.sum(FST) / xs.shape[1]
    print("FST_exp:", FST_exp)
    #print("FST exp:", FST_exp)
    return 1 / FST_exp

def HWE_FST_exp(xs, n_samples, cut):
    HWE = HWE_divergence(xs, n_samples, cut)
    FST = FST_expected(xs, n_samples, cut)
    print("HWE:", HWE)
    print("FST:", FST)
    return 0.5*(HWE + FST)

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
