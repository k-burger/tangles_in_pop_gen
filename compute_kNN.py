import pickle

import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial import distance

"""
Script to compute k-nearest neighbours.
"""


class KNearestNeighbours:
    def __init__(self, G, k, filename="", filepath="", kNN=[], pairwise_distances=[]):
        self.G = G  # genotype matrix
        self.k = k  # number of neighbours considered for each individual
        self.filename = filename  # filename
        self.filepath = filepath  # filepath
        self.kNN = kNN  # kNN matrix
        self.pairwise_distances = pairwise_distances  # distances between individuals

    # compute kNN:
    def compute_kNN(self):
        # Calculate pairwise distances between individuals using Euclidean distance
        self.pairwise_distances = (
            distance.squareform(distance.pdist(self.G, "cityblock")) / 2
        )
        # Initialize an empty adjacency matrix
        adjacency_matrix = lil_matrix((self.G.shape[0], self.G.shape[0]), dtype=bool)

        # Loop through each individual to find its k-nearest neighbors
        for i in range(self.G.shape[0]):
            # Sort individuals by distance
            nearest_neighbors = np.argsort(self.pairwise_distances[i])
            # excluding individuals themselves (distance[i] = 0):
            nearest_neighbors = np.delete(
                nearest_neighbors, np.where(nearest_neighbors == i)
            )
            # Set adjacency to True for the k-nearest neighbors
            adjacency_matrix[i, nearest_neighbors[0 : self.k]] = True
        # Convert the adjacency matrix to a dense NumPy array if needed
        self.kNN = adjacency_matrix.toarray()

        # save kNN computation:
        with open(self.filepath + self.filename, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    # load kNN computation:
    def load_kNN(self):
        # load pre-computed kNN:
        with open(self.filepath + self.filename, "rb") as inp:
            loaded_data = pickle.load(inp)

        # assign properties
        self.G = loaded_data.G
        self.k = loaded_data.k
        self.kNN = loaded_data.kNN
        self.pairwise_distances = loaded_data.pairwise_distances
