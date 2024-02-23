from __future__ import annotations
import multiprocessing
import numpy as np
from src.data_types import Cuts
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

    bipartitions.order = np.argsort(idx)

    return bipartitions

def compute_cost_and_order_cuts(bipartitions, cost_functions, verbose=True):
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

    # compute costs with multiprocessing and monitor progress:
    num_iterations = 10     # progress is monitored in 10% steps

    # in each iteration, compute costs of 10% of cuts:
    slice_size = len(bipartitions.values) // (num_iterations-1)
    #print("slice size: ", slice_size, " bipartitions.")

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
        slice = bipartitions.values[start_idx:end_idx]

        # compute cost of slice with multiprocessing
        pool = multiprocessing.Pool(processes=15)
        slice_costs = np.array(pool.map(cost_function, slice))
        pool.close()
        pool.join()
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
    costs_np_array = np.array(costs)

    return costs_np_array