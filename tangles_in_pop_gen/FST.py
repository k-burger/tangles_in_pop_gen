import numpy as np
import matplotlib.pyplot as plt

def FST_old_A_B(A, B, m):
    n = len(A) + len(B)
    n_A = len(A)
    n_B = len(B)
    p_A = (1 / (2 * n_A)) * np.sum(A[:, m])
    p_B = (1 / (2 * n_B)) * np.sum(B[:, m])
    p_T = ((n_A / n) * p_A) + ((n_B / n) * p_B)
    FST_A = np.abs(1 - ((p_A * (1 - p_A)) / (p_T * (1 - p_T))))
    FST_B = np.abs(1 - ((p_B * (1 - p_B)) / (p_T * (1 - p_T))))
    FST = 0.5*(FST_A + FST_B)

    if np.abs(FST) > 50:
        print("FST:", FST)
        print("FST_A:", FST_A)
        print("FST_B:", FST_B)
        print("p_A:", p_A)
        print("p_B:", p_B)
        print("p_T:", p_T)

    return FST

def FST_old_S_T(G, S, m):
    n = len(G)
    n_S = len(S)
    p_S = (1 / (2 * n_S)) * np.sum(S[:, m])
    p_T = (1 / (2 * n)) * np.sum(G[:, m])
    FST = 1 - ((p_S * (1 - p_S)) / (p_T * (1 - p_T)))
    if np.abs(FST) > 50:
        print("FST:", FST)
        print("p_S:", p_S)
        print("p_T:", p_T)
    return np.abs(FST)

def FST(G, A, B, m):
    n = len(G)
    n_A = len(A)
    n_B = len(B)
    p_A = (1 / (2 * n_A)) * np.sum(A[:, m])
    p_B = (1 / (2 * n_B)) * np.sum(B[:, m])
    p_T = (1 / (2 * n)) * np.sum(G[:, m])
    FST = (p_T * (1 - p_T) - (n_A/n)*(p_A * (1 - p_A)) - (n_B/n)*(p_B * (1 -p_B))) / (p_T * (1 - p_T))
    return FST

def FST_values_tangles(G, bipartitions, characterizing_cuts):
    char_cuts_keys = list(characterizing_cuts.keys())
    print("len characterizing cuts:", len(characterizing_cuts))
    FST_all = [[] for l in range(len(characterizing_cuts))]
    for i in range(0, len(characterizing_cuts)):
        print("number of char_cuts for split ", i, ":", len(list(
            characterizing_cuts[char_cuts_keys[i]].keys())))
        print("real names:", bipartitions.names[list(characterizing_cuts[char_cuts_keys[
            i]].keys())])
        for mut in bipartitions.names[list(characterizing_cuts[char_cuts_keys[
            i]].keys())]:
            cut = bipartitions.get_cut_at(mut, True)
            A = G[cut, :]
            B = G[~cut, :]
            FST_all[i].append(FST(G, A, B, mut)) # 0.5*(FST(G, A, mut) + FST(G, B,
            # mut))
        #print("len FST_all:", len(FST_all))
        # print("len FST_all i:", len(FST_all[i]))
        # print("FST > 1:", (np.array(FST_all[i]) > 1).sum())
        # print("FST < 0:", (np.array(FST_all[i]) < 0).sum())
    return FST_all

def FST_values_sim(G, mutations, pop_splits):
    FST_all = [[] for l in range(len(pop_splits))]
    G = np.array(G)
    pop_splits = np.array(pop_splits)
    for i in range(0, len(pop_splits)):
        if len(pop_splits[i]) == 3:
            A = G[pop_splits[i][0]:pop_splits[i][1]]
            B = G[pop_splits[i][1]:pop_splits[i][2]]
        else:
            A = np.concatenate((G[pop_splits[i][0]:pop_splits[i][1]],
                               G[pop_splits[i][2]:pop_splits[i][3]]))
            B = G[pop_splits[i][1]:pop_splits[i][2]]

        for mut in mutations:
            FST_all[i].append(FST(G, A, B, mut)) # 0.5*(FST(G, A, mut) + FST(G, B,
            # mut))
        #    if 0.5*(FST(G, A, mut) + FST(G, B, mut)) < 0:
        #         print("FST < 0!")
        #         print("A:", FST(G, A, mut))
        #         print("B:", FST(G, B, mut))
        # print("len FST_all:", len(FST_all))
        # print("len FST_all i:", len(FST_all[i]))
        # print("FST > 1:", (np.array(FST_all[i]) > 1).sum())
        # print("FST < 0:", (np.array(FST_all[i]) < 0).sum())
    return FST_all


def plot_FST(G, mutations, bipartitions, characterizing_cuts, pop_splits):
    FST_sim = FST_values_sim(G, mutations, pop_splits)
    FST_tangles = FST_values_tangles(G, bipartitions, characterizing_cuts)
    char_cuts_keys = list(characterizing_cuts.keys())


    fig, axs = plt.subplots(len(pop_splits), figsize=(20, 40))
    #fig.suptitle('FST values', fontsize=20)
    #fig.tight_layout()
    # Create the grid
    for subplot in axs:
        subplot.set_facecolor('white')
        subplot.set_xlabel('mutations')
        subplot.set_ylabel('FST values')

    if len(pop_splits) <= len(characterizing_cuts):
        nb_plots = len(pop_splits)
        print("Tangles has at least as many splits as the demographic structure.")
    else:
        nb_plots = len(characterizing_cuts)
        print("Tangles has less splits as the demographic structure.")

    for i in range(0, nb_plots):
        mut_names_char_cuts = bipartitions.names[list(characterizing_cuts[char_cuts_keys[
            i]].keys())]
        axs[i].scatter(mut_names_char_cuts,
                       FST_tangles[i],
                       color='r',
                       label='characterizing cut', marker='x', s=10)
        axs[i].scatter(mutations, FST_sim[i], color='b', label='in simulation',
                       marker='s', s=1)

        plt.ylabel('split in ' + str(i) + ' pops')

    plt.legend()
    plt.savefig('plots/FST_values_sim_vs_tangles.pdf')
    plt.show()


pop_splits=[[0,400,800], [0,700,800], [0,200,800], [0,600,700,800], [0,200,300,800],
            [0,400,500,800], [0,100,200,800]]