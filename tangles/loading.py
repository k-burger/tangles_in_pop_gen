import numpy as np

from sklearn.datasets import make_blobs

import networkx as nx


def load_GMM(blob_sizes, blob_centers, blob_variances, seed):

    xs, ys = make_blobs(n_samples=blob_sizes, centers=blob_centers, cluster_std=blob_variances, n_features=2,
                        random_state=seed, shuffle=False)

    return xs, ys


def load_SBM(block_sizes, p_in, p_out, seed):

    nb_nodes = np.sum(block_sizes)

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    G = nx.random_partition_graph(block_sizes, p_in, p_out, seed=seed)

    for node, ad in G.adjacency():
        A[node, list(ad.keys())] = True

    for cls, points in enumerate(G.graph["partition"]):
        ys[list(points)] = cls

    return A, ys, G


def make_mindsets(mindset_sizes, nb_questions, nb_useless, noise, seed):

    if seed is not None:
        np.random.seed(seed)

    nb_points = sum(mindset_sizes)
    nb_mindsets = len(mindset_sizes)

    xs, ys = [], []

    # create ground truth mindset
    mindsets = np.random.randint(2, size=(nb_mindsets, nb_questions))

    for idx_mindset, size_mindset in enumerate(mindset_sizes):

        # Points without noise
        xs_mindset = np.tile(mindsets[idx_mindset], (size_mindset, 1))
        ys_mindset = np.repeat(idx_mindset, repeats=size_mindset, axis=0)

        xs.append(xs_mindset)
        ys.append(ys_mindset)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)

    # Add noise
    noise_per_question = np.random.rand(nb_points, nb_questions)
    flip_question = noise_per_question < noise
    xs[flip_question] = np.logical_not(xs[flip_question])

    # add noise question like gender etc.
    if nb_useless is not None:
        mindsets = np.hstack((mindsets, np.full([nb_mindsets, nb_useless], 0.5)))
        useless = np.random.randint(2, size=[nb_points, nb_useless])
        xs = np.hstack((xs, useless))

    return xs, ys, mindsets


def make_likert_questionnaire(nb_samples, nb_features, nb_mindsets, centers, range_answers, seed=None): \
        # pragma: no cover

    if seed is not None:
        np.random.seed(seed)

    min_answer = range_answers[0]
    max_answer = range_answers[1]

    xs = np.zeros((nb_samples, nb_features), dtype=int)
    ys = np.zeros(nb_samples, dtype=int)

    idxs = np.array_split(np.arange(nb_samples), nb_mindsets)

    if not centers:
        centers = np.random.random_integers(low=min_answer, high=max_answer, size=(nb_mindsets, nb_features))
    else:
        raise NotImplementedError

    for i in np.arange(nb_mindsets):

        nb_points = len(idxs[i])
        answers_mindset = np.random.normal(loc=centers[i], size=(nb_points, nb_features))
        answers_mindset = np.rint(answers_mindset)
        answers_mindset[answers_mindset > max_answer] = max_answer
        answers_mindset[answers_mindset < min_answer] = min_answer

        xs[idxs[i]] = answers_mindset
        ys[idxs[i]] = i

    return xs, ys, centers
