import numpy as np
from scipy.spatial import distance
from scipy.sparse import lil_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class KNearestNeighbours:
    def __init__(self, G, k, filename="", filepath="", kNN = [], pairwise_distances =
    []):
        self.G = G
        self.k = k
        self.filename = filename
        self.filepath = filepath
        self.kNN = kNN
        self.pairwise_distances = pairwise_distances

    def compute_kNN(self):
        # Calculate pairwise distances between individuals using Euclidean distance
        self.pairwise_distances = distance.squareform(distance.pdist(self.G,
                                                                     'cityblock'))/2
        #print("pairwise_distances", pairwise_distances)
        # Initialize an empty adjacency matrix
        adjacency_matrix = lil_matrix((self.G.shape[0], self.G.shape[0]), dtype=bool)
        #print("adjacency_matrix", adjacency_matrix)

        # Loop through each individual to find its k-nearest neighbors
        for i in range(self.G.shape[0]):
            # Sort individuals by distance
            nearest_neighbors = np.argsort(self.pairwise_distances[i])
            # excluding individuals themselves (distance[i] = 0):
            nearest_neighbors = np.delete(nearest_neighbors, np.where(nearest_neighbors==i))

            # Set adjacency to True for the k-nearest neighbors
            adjacency_matrix[i, nearest_neighbors[0:self.k ]] = True

        # Convert the adjacency matrix to a dense NumPy array if needed
        self.kNN = adjacency_matrix.toarray()
        #print("final adjacency_matrix", adjacency_matrix)

        with open(self.filepath + self.filename, 'wb') as outp:  # overwrites any existing
            # file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def load_kNN(self):
        with open(self.filepath + self.filename, 'rb') as inp:
            loaded_data = pickle.load(inp)

        # assign properties
        self.G = loaded_data.G
        self.k = loaded_data.k
        self.kNN = loaded_data.kNN
        self.pairwise_distances = loaded_data.pairwise_distances




# xs = np.array([[0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1],
#                    [1, 1, 0, 1, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0],
#                    [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0]])
# filename = "test_save_kNN"
# filepath = "data/"
# kNN = KNearestNeighbours(xs,2, filename=filename, filepath=filepath)
# kNN.compute_kNN()
# print(kNN.kNN)
#
# # pickle kNN-Matrix for cost function:
# with open("data/saved_kNN/kNN", 'wb') as outp:  # overwrites existing file.
#     pickle.dump(kNN, outp, pickle.HIGHEST_PROTOCOL)
#
#
#
# # Beispiel-Daten (bitte mit deinen echten Daten ersetzen)
# pop_membership = np.array([0,0,0,0,1,1,2,2,2])
# pop_sizes = np.array([4,2,3])

# # Plot-Einstellungen
# plt.figure(figsize=(10, 8))
# cmap = plt.get_cmap('viridis', 8)  # Hier kannst du die Farbpalette anpassen
#
# # Plot der paarweisen Distanzen
# plt.imshow(kNN.pairwise_distances, cmap='viridis', origin='upper', interpolation='none')
#
# # Farbliche Kennzeichnung der Populationen
# start = 0
# for size in pop_sizes:
#     plt.axhline(start + size, color='white', linewidth=2)
#     plt.axvline(start + size, color='white', linewidth=2)
#     start += size
#
# # Farbliche Kennzeichnung der Populationen in der Legende
# legend_colors = [cmap(i) for i in range(8)]
# legend_labels = [f'Population {i+1}' for i in range(8)]
# plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=label,
#                                 markerfacecolor=color, markersize=10)
#                     for label, color in zip(legend_labels, legend_colors)])
# plt.show()

# # Erstelle ein DataFrame mit den Distanzen und den Populationen
# df = pd.DataFrame(kNN.pairwise_distances, columns=pop_membership, index=pop_membership)
#
# # Plot mit seaborn.clustermap
# sns.clustermap(df, cmap='viridis', method='average', col_cluster=False, row_cluster=False)
#
# plt.show()


