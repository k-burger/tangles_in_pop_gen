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
import cProfile
import psutil
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
    # Aktuellen Speicherverbrauch ausgeben
    print(f"Current memory usage 2: {psutil.virtual_memory().percent}%")
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
    # Start cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Aktuellen Speicherverbrauch ausgeben
    print(f"Current memory usage 3: {psutil.virtual_memory().percent}%")

    if verbose:
        print("Preomputing costs of cuts...")

    num_iterations = 2

    #with (open('../tangles_in_pop_gen/data/saved_costs/debug_bipartitions', 'wb') as
    #      handle):
    #    pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../tangles_in_pop_gen/data/saved_costs/debug_bipartitions',
              'rb') as handle:
        new_bipartitions = pickle.load(handle)
        #bipartitions.values = bipartitions.values[:, :90]

    # Berechnen Sie die Größe jeder Slice basierend auf der Zeilenanzahl und der Anzahl der Iterationen.
    slice_size = len(new_bipartitions.values) // (num_iterations-1)
    slice_size = 5041

    # Initialisieren eine leere Liste, um die Ergebnisse zu speichern.
    costs = []

    # Startindex für das Slicing
    start_idx = 0
    start_time = time.time()
    # Schleife über die Iterationen
    for i in range(0, num_iterations):
        # Berechnen Sie das Ende des aktuellen Slice
        end_idx = start_idx + slice_size
        # Sicherstellen, dass das Ende nicht über die Gesamtzahl der Zeilen hinausgeht
        end_idx = min(end_idx, len(new_bipartitions.values))
        # Slice erstellen
        slice = new_bipartitions.values[start_idx:end_idx].copy()
        # Aktuellen Speicherverbrauch ausgeben
        print(f"Current memory usage 4: {psutil.virtual_memory().percent}%")
        # Aktuellen Speicherverbrauch ausgeben
        print(f"Current memory usage 5: {psutil.virtual_memory().percent}%")
        # Berechnung mit cost durchführen
        pool = multiprocessing.Pool()
        slice_costs = np.array(pool.map(cost_function, slice))
        pool.close()
        # Aktuellen Speicherverbrauch ausgeben
        print(f"Current memory usage 6: {psutil.virtual_memory().percent}%")
        # Ergebnisse zur Ergebnisliste hinzufügen
        costs.extend(slice_costs)
        print("6")
        # Berechnen Sie den Fortschritt in Prozent
        progress = (i + 1) / num_iterations * 100
        print("7")
        # Fortschrittsausgabe
        duration = time.time() - start_time
        print(f"Progress: {progress:.2f}% of cost computation done in {duration:.2f} "
              f"sec.")
        #start_time = time.time()
        print("8")
        # Aktualisieren des Startindex für das nächste Slice
        start_idx = end_idx

    # Ergebnisse in einen ndarray konvertieren
    final_costs = np.array(costs)

    # Stop cProfile
    profiler.disable()
    profiler.print_stats(sort='cumulative')

    return final_costs

def compute_cost_splitted_old(bipartitions, cost_function, verbose=True):
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
    # Start cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Aktuellen Speicherverbrauch ausgeben
    print(f"Current memory usage 3: {psutil.virtual_memory().percent}%")

    if verbose:
        print("Preomputing costs of cuts...")

    num_iterations = 2

    #with (open('../tangles_in_pop_gen/data/saved_costs/debug_bipartitions', 'wb') as
    #      handle):
    #    pickle.dump(bipartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../tangles_in_pop_gen/data/saved_costs/debug_bipartitions',
              'rb') as handle:
        new_bipartitions = pickle.load(handle)
        #bipartitions.values = bipartitions.values[:, :90]

    # Berechnen Sie die Größe jeder Slice basierend auf der Zeilenanzahl und der Anzahl der Iterationen.
    slice_size = len(new_bipartitions.values) // (num_iterations-1)
    slice_size = 5041

    # Initialisieren eine leere Liste, um die Ergebnisse zu speichern.
    costs = []

    # Startindex für das Slicing
    start_idx = 0
    start_time = time.time()
    # Schleife über die Iterationen
    for i in range(0, num_iterations):
        # Berechnen Sie das Ende des aktuellen Slice
        print("1")
        end_idx = start_idx + slice_size
        print("2")
        # Sicherstellen, dass das Ende nicht über die Gesamtzahl der Zeilen hinausgeht
        end_idx = min(end_idx, len(new_bipartitions.values))
        print("3")
        # Slice erstellen
        slice = new_bipartitions.values[start_idx:end_idx].copy()
        # Aktuellen Speicherverbrauch ausgeben
        print(f"Current memory usage 4: {psutil.virtual_memory().percent}%")
        print("4", len(slice), slice.shape[0], slice.shape[1])
        print(slice)
        print(type(slice))
        # Aktuellen Speicherverbrauch ausgeben
        print(f"Current memory usage 5: {psutil.virtual_memory().percent}%")
        # Berechnung mit cost durchführen
        pool = multiprocessing.Pool()
        slice_costs = np.array(pool.map(cost_function, slice))
        pool.close()
        # Aktuellen Speicherverbrauch ausgeben
        print(f"Current memory usage 6: {psutil.virtual_memory().percent}%")
        print("5")
        # Ergebnisse zur Ergebnisliste hinzufügen
        costs.extend(slice_costs)
        print("6")
        # Berechnen Sie den Fortschritt in Prozent
        progress = (i + 1) / num_iterations * 100
        print("7")
        # Fortschrittsausgabe
        duration = time.time() - start_time
        print(f"Progress: {progress:.2f}% of cost computation done in {duration:.2f} "
              f"sec.")
        #start_time = time.time()
        print("8")
        # Aktualisieren des Startindex für das nächste Slice
        start_idx = end_idx

    # Ergebnisse in einen ndarray konvertieren
    final_costs = np.array(costs)

    # slice_size = len(bipartitions.values)/100
    # results =
    # for i in range(0, len(bipartitions.values), slice_size):
    #     # Slice erstellen, unter Berücksichtigung des verbleibenden Rests in der letzten Iteration
    #     slice = bipartitions.values[i:min(i + slice_size, len(bipartitions.values))]
    #     pool = multiprocessing.Pool()
    #     # Berechnung mit cost durchführen
    #     slice_results = np.array(pool.map(cost_function, slice))
    #     pool.close()
    #     # Ergebnisse zur Ergebnisliste hinzufügen
    #     results.extend(slice_results)
    #     # Fortschrittsausgabe
    #     print(
    #         f"Fortschritt: {i + min(slice_size, len(bipartitions.values))}/{len(bipartitions.values)} Zeilen verarbeitet")
    # Ergebnisse in einen ndarray konvertieren
    # final_result = np.array(results)
    # pool = multiprocessing.Pool()
    # cost_bipartitions = np.array(pool.map(cost_function, bipartitions.values))
    # pool.close()

    # cost_bipartitions = np.zeros(len(bipartitions.values), dtype=float)
    # for i_cut, cut in enumerate(tqdm(bipartitions.values, disable=not verbose)):
    #     cost_bipartitions[i_cut] = cost_function(cut)

    # Stop cProfile
    profiler.disable()
    profiler.print_stats(sort='cumulative')

    return final_costs