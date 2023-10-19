import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import bitarray as ba
import numpy as np
from typing import Dict

from src.tangles import Tangle, core_algorithm
from src.utils import compute_hard_predictions, matching_items, Orientation, normalize
from src.cost_functions import BipartitionSimilarity
from src.data_types import Cuts

MAX_CLUSTERS = 50


class TangleNode(object):

    def __init__(self, parent, right_child, left_child, is_left_child, name, splitting,
                 did_split, last_cut_added_id, last_cut_added_orientation, tangle: Tangle):

        self.parent = parent
        self.right_child = right_child
        self.left_child = left_child
        self.is_left_child = is_left_child

        self.splitting = splitting

        self.did_split = did_split
        self.last_cut_added_id = last_cut_added_id
        self.last_cut_added_orientation = last_cut_added_orientation

        self.tangle = tangle
        self.name = name

    @property
    def last_cut_added(self) -> np.ndarray:
        cut = self.tangle.get_cuts().get_cut_at(self.last_cut_added_id)
        if self.last_cut_added_orientation:
            return cut
        else:
            return ~cut

    def __str__(self, height=0):    # pragma: no cover

        if self.parent is None:
            string = 'Root'
        else:
            padding = ' '
            string = '{}{} -> {}'.format(padding * height,
                                         self.last_cut_added_id, self.last_cut_added_orientation)

        if self.left_child is not None:
            string += '\n'
            string += self.left_child.__str__(height=height + 1)
        if self.right_child is not None:
            string += '\n'
            string += self.right_child.__str__(height=height + 1)

        return string

    def is_leaf(self):
        return self.left_child is None and self.right_child is None


class ContractedTangleNode(TangleNode):

    def __init__(self, parent, node):

        attributes = node.__dict__
        super().__init__(**attributes)

        self.parent = parent
        self.right_child = None
        self.left_child = None

        self.characterizing_cuts = None
        self.characterizing_cuts_left = None
        self.characterizing_cuts_right = None

        self.is_left_child_deleted = False
        self.is_right_child_deleted = False

        self.p = None

    def get_characterizing_cut_values(self, characterizing_cuts: Dict[int, Orientation]) -> Dict[int, np.ndarray]:
        """
        Returns the values of the cuts in the characterizing cuts dictionary, with the orientation. 
        IDs are changed to IDs of the unsorted cuts.
        """
        ret = {}
        own_cuts: Cuts = self.tangle._cuts
        for cut_id, orientation in characterizing_cuts.items():
            cut = orientation.orient_cut(
                own_cuts.get_cut_at(cut_id, from_unsorted=False))
            ret[own_cuts.unsorted_id(cut_id)] = cut
        return ret

    def __repr__(self) -> str:
        return "Node: " + self.__str__()

    def __str__(self) -> str:
        if self.parent is None:
            return "Root"
        else:
            orientation = 'T' if self.last_cut_added_orientation else 'F'
            return f"{self.last_cut_added_id} -> {orientation}"

    def to_string_tree_like(self, height: int = 0):
        string = ""

        if self.parent is None:
            string += 'Root\n'

        padding = '  '
        string_cuts = ['{} -> {}'.format(k, v) for k, v in self.characterizing_cuts_left.items()] \
            if self.characterizing_cuts_left is not None else ''
        string += '{}{} left: {}\n'.format(padding *
                                           height, self.last_cut_added_id, string_cuts)

        string_cuts = ['{} -> {}'.format(k, v) for k, v in self.characterizing_cuts_right.items()] \
            if self.characterizing_cuts_right is not None else ''
        string += '{}{} right: {}\n'.format(padding *
                                            height, self.last_cut_added_id, string_cuts)

        if self.left_child is not None:
            string += '\n'
            string += self.left_child.to_string_tree_like(height=height + 1)
        if self.right_child is not None:
            string += '\n'
            string += self.right_child.to_string_tree_like(height=height + 1)

        return string


# created new TangleNode and adds it as child to current node
def _add_new_child(current_node, tangle, name, last_cut_added_id, last_cut_added_orientation, did_split):
    new_node = TangleNode(parent=current_node,
                          right_child=None,
                          left_child=None,
                          name=name,
                          is_left_child=last_cut_added_orientation,
                          splitting=False,
                          did_split=did_split,
                          last_cut_added_id=last_cut_added_id,
                          last_cut_added_orientation=last_cut_added_orientation,
                          tangle=tangle)

    if new_node.is_left_child:
        current_node.left_child = new_node
    else:
        current_node.right_child = new_node

    return new_node


class TangleTree(object):

    def __init__(self, agreement, cuts, max_clusters=None, prune_first_path=False):

        self.root = TangleNode(parent=None,
                               right_child=None,
                               left_child=None,
                               name=None,
                               splitting=None,
                               is_left_child=None,
                               did_split=True,
                               last_cut_added_id=-1,
                               last_cut_added_orientation=None,
                               tangle=Tangle(cuts=cuts))
        self.prune_first_path = prune_first_path
        self.cuts = cuts
        self.max_clusters = max_clusters
        self.active = [self.root]
        self.maximals = []
        self.will_split = []
        self.is_empty = True
        self.agreement = agreement
        self.first_split = None

    def __str__(self):  # pragma: no cover
        return str(self.root)

    # function to add a single cut to the tree
    # function checks if tree is empty
    # --- stops if number of active leaves gets too large ! ---
    def add_cut(self, cut, name, cut_id):
        if self.max_clusters and len(self.active) >= self.max_clusters:
            print('Stopped since there are more then %s leaves already.'.format(self.max_clusters))
            return False

        current_active = self.active
        self.active = []

        could_add_one = False
        # Go through all nodes that are on the order of the preceding cut.
        # Check if we can add the current cut to them.
        for current_node in current_active:
            could_add_node, did_split, is_maximal = self._add_children_to_node(
                current_node, cut, name, cut_id)
            could_add_one = could_add_one or could_add_node

            if did_split:
                print("test")
                if not self.first_split and self.prune_first_path:
                    index_of_cut = np.where(np.sum(self.cuts.values == cut, axis=1) == self.cuts.values.shape[1])[0][0]
                    cuts = Cuts(values=self.cuts.values[index_of_cut:],
                                costs=self.cuts.costs[index_of_cut:],
                                names=self.cuts.names[index_of_cut:] if self.cuts.names is not None else None,
                                equations=self.cuts.equations[index_of_cut:] if self.cuts.equations is not None else None)
                    self.__init__(agreement=self.agreement, cuts=cuts, max_clusters=self.max_clusters)
                    self.first_split = True
                    could_add_node, did_split, is_maximal = self._add_children_to_node(
                        current_node, cut, name, cut_id)
                    could_add_one = could_add_one or could_add_node
                current_node.splitting = True
                self.will_split.append(current_node)
            elif is_maximal:
                self.maximals.append(current_node)

        if could_add_one:
            self.is_empty = False

        return could_add_one

    def _add_children_to_node(self, current_node, cut, name, cut_id):
        old_tangle = current_node.tangle

        if cut.dtype is not bool:
            cut = cut.astype(bool)

        # Tangle with the cut added in present orientation.
        new_tangle_true = old_tangle.add(new_cut=ba.bitarray(cut.tolist()),
                                         new_cut_id=cut_id,
                                         orientation=True,
                                         min_size=self.agreement)
        # Tangle with the cut added in opposite orientation.
        new_tangle_false = old_tangle.add(new_cut=ba.bitarray((~cut).tolist()),
                                          new_cut_id=cut_id,
                                          orientation=False,
                                          min_size=self.agreement)

        could_add_one = False

        # Case of a splitting tangle, we could add both orientations
        if new_tangle_true is not None and new_tangle_false is not None:
            did_split = True
        else:
            did_split = False

        # Cut could not be added in any orientation.
        if new_tangle_true is None and new_tangle_false is None:
            is_maximal = True
        else:
            is_maximal = False

        # Cut could be added in the original orientation.
        if new_tangle_true is not None:
            could_add_one = True
            new_node = _add_new_child(current_node=current_node,
                                      name=name,
                                      tangle=new_tangle_true,
                                      last_cut_added_id=cut_id,
                                      last_cut_added_orientation=True,
                                      did_split=did_split)
            self.active.append(new_node)

        # Cut could be added in the opposite orientation.
        if new_tangle_false is not None:
            could_add_one = True
            new_node = _add_new_child(current_node=current_node,
                                      name=name,
                                      tangle=new_tangle_false,
                                      last_cut_added_id=cut_id,
                                      last_cut_added_orientation=False,
                                      did_split=did_split)
            self.active.append(new_node)

        return could_add_one, did_split, is_maximal

    def plot_tree(self, path=None):    # pragma: no cover

        tree = nx.Graph()
        labels = self._add_node_to_nx(tree, self.root)

        pos = graphviz_layout(tree, prog='dot')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        nx.draw_networkx(tree, pos=pos, ax=ax, labels=labels, node_size=1500)
        plt.tight_layout()
        if path:
            plt.savefig(path, bbox_inches='tight')
        else:
            plt.show()

    def print_tangles_tree_summary_hard_predictions(self, n, idx_bipartitions, ys_predicted):
        tree = nx.Graph()
        path_to_leaf_init = []
        j = 0

        # vector describing the presence of the samples in every tangles branch from root to leaf
        path_to_leaf = self._get_path_to_leaf(tree, self.root, path_to_leaf_init, n)

        # get an index sorting to display the samples later in ascending order
        idx_sort = np.argsort(idx_bipartitions)
        print("idx bipartitions:", idx_bipartitions)
        print("sorted idx bipartitions:", idx_sort)

        # if the tangles branch from the root to the leaf does not have the length n (number of samples),
        # fill the vector indicating the presence of the samples from behind with -1 until it has a length of n.
        for i in range(0,len(path_to_leaf)):
            path_to_leaf[i] = np.array(path_to_leaf[i]).astype(int)
            if len(path_to_leaf[i]) != n:
                path_to_leaf[i] = np.append(path_to_leaf[i], (-1)*np.ones(n-len(path_to_leaf[i])))
            path_to_leaf[i] = np.array(path_to_leaf[i]).astype(int)[idx_sort]

        # get characteristics of mutations clustered into each leaf of the tangles tree
        mut_nb = []     # number of mutations
        mut_idx = []    # index of mutations (i.e. names of mutation)
        mut_mean_pos = []   # mean position of mutations (i.e. mean index of mutations)
        mut_var_pos = []    # variance of mutations (i.e. variance of index of mutations)
        for k in range(0,len(path_to_leaf)):
            mut_nb.append(np.count_nonzero(ys_predicted == k))
            mut_idx.append(np.sort(np.where(np.array(ys_predicted) == k)[0]))
            if np.count_nonzero(ys_predicted == k) != 0:
                mut_mean_pos.append(np.round(np.mean(np.where(np.array(ys_predicted) == k)[0]), 1))
                mut_var_pos.append(np.round(np.var((np.where(np.array(ys_predicted) == k)[0])), 1))
            else:
                mut_mean_pos.append(-1)     # no mutation clustered to this particular leaf
                mut_var_pos.append(-1)      # no mutation clustered to this particular leaf

        # print summary of the tangles tree (i.e. characteristics computed above)
        print("summary tangles tree:")
        for elem in path_to_leaf:
            print(j, ": ", elem, ", nb of mutations: ", mut_nb[j], ", mutation idx: ", mut_idx[j],
                  ", mean mutation position:", mut_mean_pos[j], ", var mutation position:", mut_var_pos[j])
            j = j+1

    def _get_path_to_leaf(self, tree, node, path_to_leaf, n):
        if node.left_child is None and node.right_child is None:
            path_to_leaf.append([node.tangle._specification[i] for i in range(0,len(node.tangle._specification))])

        if node.left_child is not None:
            left_subtree = self._get_path_to_leaf(tree, node.left_child, path_to_leaf, n)
        if node.right_child is not None:
            right_subtree = self._get_path_to_leaf(tree, node.right_child, path_to_leaf, n)
        return path_to_leaf

    def _add_node_to_nx(self, tree, node, parent_id=None, direction=None):  # pragma: no cover

        if node.parent is None:
            my_id = 'root'
            my_label = 'Root'

            tree.add_node(my_id)
        else:
            my_id = parent_id + direction
            str_o = 'T' if node.last_cut_added_orientation else 'F'
            my_label = '{} -> {}'.format(node.name, str_o)

            tree.add_node(my_id)
            tree.add_edge(my_id, parent_id)

        labels = {my_id: my_label}

        if node.left_child is not None:
            left_labels = self._add_node_to_nx(
                tree, node.left_child, parent_id=my_id, direction='left')
            labels = {**labels, **left_labels}
        if node.right_child is not None:
            right_labels = self._add_node_to_nx(
                tree, node.right_child, parent_id=my_id, direction='right')
            labels = {**labels, **right_labels}

        return labels


class ContractedTangleTree(TangleTree):

    # noinspection PyMissingConstructor
    def __init__(self, tree):

        self.is_empty = tree.is_empty
        self.processed_soft_prediction = False
        self.maximals = []
        self.splitting = []
        self.root = self._contract_subtree(parent=None, node=tree.root)

    def __str__(self):  # pragma: no cover
        return str(self.root)

    def prune(self, prune_depth=1, verbose=True):
        self._delete_noise_clusters(self.root, depth=prune_depth)
        if verbose:
            print("\t{} clusters after cutting out short paths.".format(
                len(self.maximals)))

    def _delete_noise_clusters(self, node, depth):
        if depth == 0:
            return

        if node.is_leaf():
            if node.parent is None:
                Warning(
                    "This node is a leaf and the root at the same time. This tree is empty!")
            else:
                node_id = node.last_cut_added_id
                parent_id = node.parent.last_cut_added_id

                diff = node_id - parent_id

                if diff <= depth:
                    self.maximals.remove(node)
                    node.parent.splitting = False
                    if node.is_left_child:
                        node.parent.left_child = None
                        node.parent.is_left_child_deleted = True
                    else:
                        node.parent.right_child = None
                        if node.parent.is_left_child_deleted:
                            self.maximals.append(node.parent)
                            self._delete_noise_clusters(node.parent, depth)

        else:
            self._delete_noise_clusters(node.left_child, depth)
            if not node.splitting:
                self.splitting.remove(node)

            self._delete_noise_clusters(node.right_child, depth)

            if not node.splitting:
                if node.parent is None:
                    if node.right_child is not None:
                        self.root = node.right_child
                        self.root.parent = None
                    elif node.left_child is not None:
                        self.root = node.left_child
                        self.root.parent = None
                else:
                    if node in self.splitting:
                        self.splitting.remove(node)
                        if node.is_left_child:
                            node.parent.left_child = node.left_child
                        else:
                            node.parent.right_child = node.left_child
                    else:
                        if node.right_child is not None:
                            if node.is_left_child:
                                node.parent.left_child = node.right_child
                            else:
                                node.parent.right_child = node.right_child

    def calculate_setP(self):
        self._calculate_characterizing_cuts(self.root)

    def _calculate_characterizing_cuts(self, node):

        if node.left_child is None and node.right_child is None:
            node.characterizing_cuts = dict()
            return
        else:
            if node.left_child is not None and node.right_child is not None:
                self._calculate_characterizing_cuts(node.left_child)
                self._calculate_characterizing_cuts(node.right_child)

                process_split(node)
                return

    # As python has no Tail Call Optimization, it is more beneficial to
    # use contract_subtree in an iterative fashion. Else we quickly
    # get in the territory of a stack explosion.
    def _contract_subtree_iterative(self, parent, node):
        current_node = node

        while True:
            if current_node.left_child is None and current_node.right_child is None:
                # is leaf so create new node
                contracted_node = ContractedTangleNode(
                    parent=parent, node=current_node)
                self.maximals.append(contracted_node)
                return contracted_node
            elif current_node.left_child is not None and current_node.right_child is not None:
                # is splitting so create new node
                contracted_node = ContractedTangleNode(
                    parent=parent, node=current_node)

                contracted_left_child = self._contract_subtree_iterative(
                    parent=contracted_node, node=current_node.left_child)
                contracted_node.left_child = contracted_left_child
                # let it know that it is a left child!
                contracted_node.left_child.is_left_child = True

                contracted_right_child = self._contract_subtree_iterative(
                    parent=contracted_node, node=current_node.right_child)
                contracted_node.right_child = contracted_right_child
                # let it know that it is a right child!
                contracted_node.right_child.is_left_child = False

                self.splitting.append(contracted_node)

                return contracted_node
            else:
                if current_node.left_child is not None:
                    current_node = current_node.left_child
                elif current_node.right_child is not None:
                    current_node = current_node.right_child

    def _contract_subtree(self, parent, node):
        return self._contract_subtree_iterative(parent, node)

    def print_summary(self, node, loc=''):
        if node:
            self.print_summary(node.left_child, loc + 'l')
            self.print_node(node, loc)
            self.print_summary(node.right_child, loc + 'r')
        else:
            return

    def print_node(self, node, loc):
        if node == self.root:
            return
        else:
            print(loc, ": ", np.argwhere(node.p > 0.5).flatten())

    def to_matrix(self):
        queue = [self.root]
        return self._write_to_mat(queue, 1, {}, {})

    def _write_to_mat(self, queue, index, matrices, char_cuts):
        try:
            node = queue[0]
            queue = queue[1:]
        except:
            return matrices, char_cuts

        if node.left_child is not None and node.right_child is not None:
            if index in matrices.keys():
                matrices[index] = np.concatenate([matrices[index], node.left_child.p.reshape(-1, 1)], axis=1)
                char_cuts[index] = np.concatenate([char_cuts[index], node.left_child.characterizing_cuts], axis=1)
            else:
                matrices[index] = node.left_child.p.reshape(-1, 1)
                char_cuts[index] = node.left_child.characterizing_cuts
            matrices[index] = np.concatenate([matrices[index], node.right_child.p.reshape(-1, 1)], axis=1)

            if node.left_child.last_cut_added_id < node.right_child.last_cut_added_id:
                queue.append(node.left_child)
                queue.append(node.right_child)
            else:
                queue.append(node.right_child)
                queue.append(node.left_child)

        matrices, char_cuts = self._write_to_mat(queue, index + 1, matrices, char_cuts)

        return matrices, char_cuts


def process_split(node):
    node_id = node.last_cut_added_id if node.last_cut_added_id else -1

    characterizing_cuts_left = node.left_child.characterizing_cuts
    characterizing_cuts_right = node.right_child.characterizing_cuts

    orientation_left = node.left_child.tangle.get_specification()
    orientation_right = node.right_child.tangle.get_specification()

    # add new relevant cuts
    for id_cut in range(node_id + 1, node.left_child.last_cut_added_id + 1):
        characterizing_cuts_left[id_cut] = Orientation(
            orientation_left[id_cut])

    for id_cut in range(node_id + 1, node.right_child.last_cut_added_id + 1):
        characterizing_cuts_right[id_cut] = Orientation(
            orientation_right[id_cut])

    id_not_in_both = (characterizing_cuts_left.keys() | characterizing_cuts_right.keys()) \
        .difference(characterizing_cuts_left.keys() & characterizing_cuts_right.keys())

    # if cuts are not oriented in both subtrees delete
    for id_cut in id_not_in_both:
        characterizing_cuts_left.pop(id_cut, None)
        characterizing_cuts_right.pop(id_cut, None)

    # characterizing cuts of the current node
    characterizing_cuts = {
        **characterizing_cuts_left, **characterizing_cuts_right}

    id_cuts_oriented_same_way = matching_items(
        characterizing_cuts_left, characterizing_cuts_right)

    # if they are oriented in the same way they are not relevant for distungishing but might be for 'higher' nodes
    # delete in the left and right parts but keep in the characteristics of the current node
    for id_cut in id_cuts_oriented_same_way:
        characterizing_cuts[id_cut] = characterizing_cuts_left[id_cut]
        characterizing_cuts_left.pop(id_cut)
        characterizing_cuts_right.pop(id_cut)

    id_cuts_oriented_both_ways = characterizing_cuts_left.keys(
    ) & characterizing_cuts_right.keys()

    # remove the cuts that are oriented in both trees but in different directions from the current node since they do
    # not affect higher nodes anymore
    for id_cut in id_cuts_oriented_both_ways:
        characterizing_cuts.pop(id_cut)

    node.characterizing_cuts_left = characterizing_cuts_left
    node.characterizing_cuts_right = characterizing_cuts_right
    node.characterizing_cuts = characterizing_cuts


def compute_soft_predictions_node(characterizing_cuts, cuts, weight):
    sum_p = np.zeros(len(cuts.values[0]))

    for i, o in characterizing_cuts.items():
        if o.direction == 'left':
            sum_p += np.array(cuts.values[i]) * weight[i]
        elif o.direction == 'right':
            sum_p += np.array(~cuts.values[i]) * weight[i]

    return sum_p


def compute_soft_predictions_children(node, cuts, weight, verbose=0):
    _, nb_points = cuts.values.shape

    if node.parent is None:
        node.p = np.ones(nb_points)

    if node.left_child is not None and node.right_child is not None:

        unnormalized_p_left = compute_soft_predictions_node(characterizing_cuts=node.characterizing_cuts_left,
                                                            cuts=cuts,
                                                            weight=weight)
        unnormalized_p_right = compute_soft_predictions_node(characterizing_cuts=node.characterizing_cuts_right,
                                                             cuts=cuts,
                                                             weight=weight)

        # normalize the ps
        total_p = unnormalized_p_left + unnormalized_p_right

        p_left = unnormalized_p_left / total_p
        p_right = unnormalized_p_right / total_p

        node.left_child.p = p_left * node.p
        node.right_child.p = p_right * node.p

        compute_soft_predictions_children(node=node.left_child,
                                          cuts=cuts,
                                          weight=weight,
                                          verbose=verbose)

        compute_soft_predictions_children(node=node.right_child,
                                          cuts=cuts,
                                          weight=weight,
                                          verbose=verbose)


def tangle_computation(cuts, agreement, verbose, max_clusters=None, prune_first_path=False):
    """

    Parameters
    ----------
    cuts: cuts
    agreement: int
        The agreement parameter
    verbose:
        verbosity level
    Returns
    -------
    tangles_tree: TangleTree
        The tangle search tree
    """

    if verbose >= 2:
        print("Using agreement = {} \n".format(agreement))
        print("Start tangle computation", flush=True)

    tangles_tree = TangleTree(agreement=agreement, cuts=cuts, max_clusters=max_clusters, prune_first_path=prune_first_path)
    old_order = None

    unique_orders = np.unique(cuts.costs)

    for order in unique_orders:

        if old_order is None:
            idx_cuts_order_i = np.where(cuts.costs <= order)[0]
        else:
            idx_cuts_order_i = np.where(np.all([cuts.costs > old_order,
                                                cuts.costs <= order], axis=0))[0]

        if len(idx_cuts_order_i) > 0:

            if verbose >= 2:
                print("\tCompute tangles of order {} with {} new cuts".format(
                    order, len(idx_cuts_order_i)), flush=True)

            cuts_order_i = cuts.values[idx_cuts_order_i]
            cuts_names_i = cuts.names[idx_cuts_order_i] if cuts.names is not None else idx_cuts_order_i
            new_tree = core_algorithm(tree=tangles_tree,
                                      current_cuts=cuts_order_i,
                                      current_names=cuts_names_i,
                                      idx_current_cuts=idx_cuts_order_i)

            if new_tree is None:
                max_order = cuts.costs[-1]
                if verbose >= 2:
                    print('\t\tI could not add any new cuts due to inconsistency')
                    print('\n\tI stopped the computation at order {} instead of {}'.format(old_order, max_order),
                          flush=True)
                break
            else:
                tangles_tree = new_tree

                if verbose >= 2:
                    print("\t\tI found {} tangles of order less or equal {}".format(len(new_tree.active), order),
                          flush=True)

        old_order = order

    if tangles_tree is not None:
        tangles_tree.maximals += tangles_tree.active

    if verbose >= 1:
        print("\t{} leaves before cutting out short paths.".format(
            len(tangles_tree.maximals)))

    return tangles_tree


def get_hard_predictions(X: np.ndarray, agreement: int, verbose: int = 0):
    """
    Simple function to return hard predictions from a set of cuts X.

    Cuts X are column-wise e.g. each column is a cut.
    """
    verbose_bool = verbose > 0
    cuts = Cuts((X == 1).T)
    cost_function = BipartitionSimilarity(
        cuts.values.T)
    cuts.compute_cost_and_order_cuts(cost_function, verbose=verbose_bool)

    # Building the tree, contracting and calculating predictions
    tangles_tree = tangle_computation(cuts=cuts,
                                      agreement=agreement,
                                      # print nothing
                                      verbose=verbose_bool)

    contracted = ContractedTangleTree(tangles_tree)
    contracted.prune(1, verbose=verbose_bool)

    contracted.calculate_setP()

    # soft predictions
    weight = np.exp(-normalize(cuts.costs))

    compute_soft_predictions_children(
        node=contracted.root, cuts=cuts, weight=weight, verbose=verbose_bool)
    contracted.processed_soft_predictions = True

    ys_predicted, _ = compute_hard_predictions(
        contracted, cuts, verbose=verbose_bool)

    return ys_predicted
