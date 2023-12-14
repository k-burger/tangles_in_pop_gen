import numpy as np

def get_adaptive_cuts(xs, s, nb_iter, seed):
    # set seed:
    np.random.seed(seed)
    # get number of individuals and mutations of genotype matrix xs:
    nb_individuals, nb_mut = xs.shape
    # init list to store cuts (number of cuts is nb_mut//s)
    cuts = [2] * (nb_mut // s)
    mut_per_cut = [2] * (nb_mut // s)
    # print("init cuts:", cuts)

    # # list with all mutation indices to be shuffled:
    # mutation_indices = list(range(nb_mut))
    # # shuffle list mutation_indices to base cuts on randomly selected mutations:
    # np.random.shuffle(mutation_indices)
    # # print("mutation indices after shuffle:", mutation_indices)
    # # print("list for loop:", list(range(0, nb_mut, s)))

    # sort mutations according to their deviation from the Hardy-Weinberg equilibrium:
    mutation_indices = sort_mutations_HWE(xs)
    print("mutation indices:", mutation_indices)

    # for each iteration compute cuts based on selected mutations and compute
    # probability for each indv to be part of A and B and sort accordingly:
    for run in range(0, nb_iter):
        print("start run ", run)

        # init matrix to save probabilities of each indv to be on each side of each cut:
        P_n_in_A = np.zeros((nb_individuals, nb_mut // s))
        P_n_in_B = np.zeros((nb_individuals, nb_mut // s))

        # iterate through groups of randomly selected mutations of length s:
        for c in list(range(0, nb_mut-s, s)):
            #print("start c:", c // s)
            # get randomly selected mutation to base cut computation on:
            selected_mutations = mutation_indices[c:c+s]
            #print("selected mutations:", selected_mutations)

            # save selected mutations:
            mut_per_cut[c // s] = selected_mutations.tolist()


            # init allele frequencies of all selected mutations for A and B. Init
            # probabilities of observing allele counts of 0, 1, 2 in A and B:
            af_A = np.zeros(s)
            af_B = np.zeros(s)
            P_A = np.zeros((3, s))
            P_B = np.zeros((3, s))

            # initial split in A and B based on selected mutations:
            if run == 0:
                # # initial sets A and B s.t. A contains all indv that carry atleast
                # # one of the selected mutations not:
                # A = np.any(xs[:, selected_mutations] == 0, axis=1)
                # B = ~A

                # # random initial sets A and B:
                # A = np.random.choice([True, False], size=nb_individuals)
                # B = ~A

                # initial sets A and B s.t. A contains individuals that have more
                # copies of first selected mutation than second and B that have more
                # copies of second selected mutation. All othe mutations get
                # distributed randomly:
                random_assignment = np.random.choice([True, False], size=xs.shape[0])
                condition_A = xs[:, selected_mutations[0]] > xs[:,
                                                             selected_mutations[1]]
                condition_B = xs[:, selected_mutations[0]] < xs[:,
                                                             selected_mutations[1]]
                A = np.full(xs.shape[0], False)  # Alle False initialisieren
                B = np.full(xs.shape[0], False)  # Alle False initialisieren
                A[condition_A] = True
                B[condition_B] = True
                A[xs[:, selected_mutations[0]] == xs[:, selected_mutations[1]]] = \
                random_assignment[
                    xs[:, selected_mutations[0]] == xs[:, selected_mutations[1]]]
                B[xs[:, selected_mutations[0]] == xs[:, selected_mutations[1]]] = ~ \
                random_assignment[
                    xs[:, selected_mutations[0]] == xs[:, selected_mutations[1]]]

            # if cuts have already been computed, take them to base new computation on:
            elif np.any(cuts[c // s] != 2):
                B = cuts[c//s]
                A = ~B

            # cut splits indv into A and B with A or B empty. skip.
            else:
                continue

            # get number of indv in A and B:
            len_A = np.count_nonzero(A == True)
            len_B = np.count_nonzero(B == True)

            # if A or B contain no indv, skip cut update based on the currently
            # # selected mutations for this run:
            # if len_A == 0 or len_B == 0:
            #     # mark that
            #     # cuts[c // s] = 0
            #     continue

            # get proportion of indv in A and B:
            frac_A = 1/2 #len_A/(len_A + len_B)
            frac_B = 1/2 #len_B/(len_A + len_B)
            #print("len:", len_A, len_B)

            # compute for each selected mutation allele frequencies and probability
            # of allele counts in A and B:
            for m in range(s):
                af_A[m] = (1 / (2 * len_A + 2)) * (np.sum(xs[A, selected_mutations[
                    m]]) + 1)
                af_B[m] = (1 / (2 * len_B + 2)) * (np.sum(xs[B, selected_mutations[
                    m]]) + 1)
                P_A[0, m] = np.power((1-af_A[m]), 2)
                P_A[1, m] = af_A[m]*(1-af_A[m])
                P_A[2, m] = np.power(af_A[m], 2)
                P_B[0, m] = np.power((1 - af_B[m]), 2)
                P_B[1, m] = af_B[m] * (1 - af_B[m])
                P_B[2, m] = np.power(af_B[m], 2)

            # compute probability of each indv to belong to A and B based on selected
            # mutations:
            for n in range(nb_individuals):
                nominator_A = 1
                denominator = 0
                nominator_B = 1
                for m in range(s):
                    nominator_A = nominator_A * P_A[xs[n,selected_mutations[m]], m]
                    nominator_B = nominator_B * P_B[xs[n,selected_mutations[m]], m]
                    # denominator +=  frac_A * P_A[xs[n, selected_mutations[
                    #     m]], m] + frac_B*P_B[xs[n, selected_mutations[m]], m])
                P_n_in_A[n, c // s] = (nominator_A*frac_A)/(frac_A*nominator_A +
                                                            frac_B*nominator_B)
                P_n_in_B[n, c // s] = (nominator_B * frac_B) /((frac_A*nominator_A) +
                                                               (frac_B*nominator_B))


                # sort indv to more likely group A or B, if same probability,
                # don't change group:
                if (P_n_in_A[n, c // s] > P_n_in_B[n, c // s]) and (B[n] == True):
                    A[n] = True
                    B[n] = False
                    if run > 0:
                        print("indv", n, "changed for cut",  c // s, "into A in run",
                              run) #,
                          # ". New B:", B)
                elif (P_n_in_A[n, c // s] < P_n_in_B[n, c // s]) and (A[n] == True):
                    A[n] = False
                    B[n] = True
                    if run > 0:
                        print("indv", n, "changed for cut",  c // s, "into B in run",
                              run) #,
                          # ". New B:", B)

                # check im A and B update resulted in empty groups, if yes, skip
                if np.count_nonzero(A == True) == 0 or np.count_nonzero(B == True) == 0:
                    # skip this cut
                    print("empty!")
                    # continue
                else:
                    # save resulting cut (assignment of indv to A and B)
                    cuts[c // s] = B

        # print("cuts ", cuts, "in run ", run)

    #print("nb_skipped_cuts:", nb_skipped_cuts)
    # remove all empty cuts:
    # cuts = list(filter(lambda x: not np.all(x == 2), cuts))
    # cuts = list(filter(lambda x: not (np.all(x == True) or np.all(x == False)), cuts))
    mask = np.array(list(map(lambda x: not np.all(x == 2) and not (
                    np.all(x == True) or np.all(x == False)), cuts)))
    cuts = list(filter(lambda x: not np.all(x == 2) and not (
                    np.all(x == True) or np.all(x == False)), cuts))
    # remove cuts in P_n_in_A and P_n_in_B accordingly.
    # P_n_in_A = P_n_in_A[:, mask]
    P_n_in_B = P_n_in_B[:, mask]

    # remove selected mutations for cuts not considered:
    mut_per_cut = [mut_per_cut[i] for i in range(len(mut_per_cut)) if mask[i]]
    #print("mut_per_cut:", mut_per_cut)
    print(len(cuts), " out of ", len(list(range(0, nb_mut, s))))
    names = np.array(list(range(len(cuts))))

    for final_cut in cuts:
        if len(final_cut) != nb_individuals:
            raise ValueError("Invalid number of indv in cut: {}".format(
                len(final_cut)))
    return np.array(cuts), names, P_n_in_B, mut_per_cut


# function to sort the mutations according to their deviation from the Hardy-Weinberg
# equilibrium:
def sort_mutations_HWE(xs):
    # get number of individuals and mutations of genotype matrix xs:
    nb_indv, nb_mut = xs.shape

    # compute allele frequencies of all mutations
    af = (1 / (2 * nb_indv)) * np.sum(xs, axis=0)

    # compute expected and observed heterozygotes:
    expected_heterozygotes = 2 * af * (1 - af)
    observed_heterozygotes = (1 / nb_indv) * np.count_nonzero(xs==1, axis=0)

    # compute difference between observed and expected
    HWE = np.abs(observed_heterozygotes - expected_heterozygotes)
    print(HWE)

    # sort mutations according to HWE difference, starting with those having the
    # largest difference between observed and expected:
    sorted_mutations = np.argsort(HWE)[::-1]

    return sorted_mutations




# # minimal example:
# xs = np.array([[0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 0, 1, 1, 1],
#                [0, 0, 0, 0, 0, 1, 1,1], [1, 1, 0, 1, 0, 1, 1, 1],
#                [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0],
#                [0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0],
#                [0, 0, 1, 0, 1, 0, 0, 0]])
# s = 3
# cuts, names, P_n_in_B, mut_per_cut = get_adaptive_cuts(xs, s, 10, seed=42)
# print("cuts:", cuts)
# print(len(cuts), " out of ", len(list(range(0, xs.shape[1], s))))


