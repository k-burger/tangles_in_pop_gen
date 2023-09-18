from __future__ import annotations
import hashlib
import json
import multiprocessing

import numpy as np
from sklearn.manifold import TSNE
from typing import Union, Optional

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import DistanceMetric
from tqdm import tqdm

from src.data_types import Cuts
import pickle
import time

def order_cuts(bipartitions: Cuts, cost_bipartitions: np.ndarray):
    """
    Orders cuts based on the cost of the cuts.

    bipartitions: Cuts,
    where values contains an ndarray of shape (n_questions, n_datapoints).
    cost_bipartitions: ndarray,
    where values contains an ndarray of shape (n_datapoints). Contains
    the cost of each cut as value.
    """
    idx = np.argsort(cost_bipartitions)

    bipartitions.values = bipartitions.values[idx]
    bipartitions.costs = cost_bipartitions[idx]
    if bipartitions.names is not None:
        bipartitions.names = bipartitions.names[idx]
    #if bipartitions.equations is not None:
    #    bipartitions.equations = bipartitions.equations[idx]

    bipartitions.order = np.argsort(idx)

    with (open('../tangles_in_pop_gen/data/saved_costs/test_load_bipartitions', 'wb') as
          handle):
        pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bipartitions

def compute_cost_and_order_cuts(bipartitions, cost_functions, verbose=True):
    #bipartitions.values = bipartitions.values.astype(np.uint8)
    costs = compute_cost_splitted(bipartitions, cost_functions, verbose=verbose)
    return order_cuts(bipartitions, costs)

def compute_cost_splitted(bipartitions, cost_function, verbose=True):
    """
    Compute the cost of a series of cuts and returns a cost array.

    Parameters
    ----------
    cuts: Cuts
        where cuts.values has shape (n_questions, n_datapoints)
    cost_function: function
        callable that calculates the cost of a single cut, which is an ndarray of shape
        (n_datapoints)

    Returns
    -------
    cost: ndarray of shape (n_questions) containing the costs of each cut as entries
    """
    if verbose:
        print("Preomputing costs of cuts...")

    # set for testing to 2 (only one iteration)
    num_iterations = 2

    # for debugging load bipartitions
    #with (open('../tangles_in_pop_gen/data/saved_costs/debug_bipartitions', 'wb') as
    #      handle):
    #    pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../tangles_in_pop_gen/data/saved_costs/debug_bipartitions',
              'rb') as handle:
        bipartitions = pickle.load(handle)
        #bipartitions.values = bipartitions.values[:, :90]

    # for debugging slice sice set to 5041 as it is the number of mutations in loaded
    # test example
    # set slice size
    slice_size = len(bipartitions.values) // (num_iterations-1)
    slice_size = 5041

    # Initialize list for saving costs of bipartitions
    costs = []

    # Set start index for slicing
    start_idx = 0
    start_time = time.time()

    for i in range(0, num_iterations):
        # Set end index for slicing
        end_idx = start_idx + slice_size
        # make sure end index is larger than number of mutations
        end_idx = min(end_idx, len(bipartitions.values))

        # get slice of bipartitions
        slice = bipartitions.values[start_idx:end_idx].copy()

        # compute cost of slice with multiprocessing
        pool = multiprocessing.Pool()
        slice_costs = np.array(pool.map(cost_function, slice))
        pool.close()

        # add costs to list of costs
        costs.extend(slice_costs)

        # monitor progress in percent
        progress = (i + 1) / num_iterations * 100

        # print progress
        duration = time.time() - start_time
        print(f"Progress: {progress:.2f}% of cost computation done in {duration:.2f} "
              f"sec.")

        # update start index for slice computation in next iteration
        start_idx = end_idx

    # transform list of costs into numpy array
    final_costs = np.array(costs)

    return final_costs