import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import string
import subprocess


# This function creates an admixture-like plot with tangles and compares it to
# ADMIXTURE. The function is based on matrices, which contain the soft clustering
# level-wise (regarding the tangles tree). We also have to input the population
# memberships (from simulation) to plot accordingly. The agreement is given to
# distinguish the saved plots in the end. The comparison to ADMIXTURE can be turned off.
def admixture_like_plot(matrices, pop_membership, agreement, seed,
                        data_generation_mode, sorting_level="",
                        plot_ADMIXTURE = False, ADMIXTURE_file_name="", cost_fct = ""):
    n = np.array(matrices[1]).shape[0]      # number of samples
    nb_plots = len(matrices)                # number of plots to generate
    mtx_keys = list(matrices.keys())        # get keys of the matrices dictionary
    indv = list(range(0, n))                # list of individuals for x-axis of plot
    y = []                                  # list of soft pred in each level
    y_plot = []                             # list of soft pred to be plotted on y-axis
    color_order = []                        # list to save consistent order of colors
    colors_per_plot = []                    # list of colors for each bar plot
    #nb_plots = 12

    # create list of colors to plot from (cyclic color palette). If number of plots
    # is less 24, choose colors in specific order to increase color contrast. If more
    # colors needed, take usual order of colors.
    if nb_plots+1 < 24:
        c = sns.color_palette("husl", 24)   # choose color palette
        # change order of colors to increase color contrast:
        cmap = [c[0], c[12], c[6], c[18], c[3], c[9], c[15], c[21], c[1], c[4], c[7],
                c[10], c[13], c[16], c[19], c[22], c[2], c[5], c[8], c[11], c[14],
                c[17], c[20], c[23]]
    else:
        cmap = sns.color_palette("husl", nb_plots + 1)  # choose color palette
        # swap first color with middle color to have better color contrast:
        cmap[1], cmap[np.ceil((nb_plots+1)/2).astype(int)] = cmap[np.ceil(
            (nb_plots+1)/2).astype(int)], cmap[1]

    # fill y with soft predictions for each plot/ level of soft clustering
    # in same loop save color order for consistency throughout the bar plots
    for i in range(0, nb_plots):    # loop through matrices, always choose the split
        # with the next lowest cost
        if i == 0:
            # append soft predictions for left child of root
            y.append([row[0] for row in matrices[mtx_keys[i]]])
            # append soft predictions for right child of root
            y.append([row[1] for row in matrices[mtx_keys[i]]])
            # get colors:
            color_order.append(cmap[0])
            color_order.append(cmap[1])
            # save the soft predictions for first bar plot:
            y_plot.append(copy.deepcopy(y))  # Make a copy of y and append the copy to y_plot
            # save color order for first bar plot:
            colors_per_plot.append(copy.deepcopy(color_order))

        else:
            # if we are not at the root, we need to find out what the parent of the
            # split under consideration is in order to correctly classify the split,
            # i.e. the branching order.
            # Therefore, sum the soft predictions of left and right child of the
            # split compare with all predictions one level up. If they match,
            # we have found the parent split:
            split = [sum(s) for s in zip([row[0] for row in matrices[mtx_keys[i]]],
                                         [row[1] for row in matrices[mtx_keys[i]]])]
            check = 0
            # find branching order:
            for k in range(0,len(y)):
                if np.allclose(np.array(split), np.array(y[k]), 1e-15):
                    check = check + 1
                    if check == 1:  # parent split is found.
                        # place the soft predictions in the correct branching order:
                        y[k:k+1] = [row[0] for row in matrices[mtx_keys[i]]], [row[1] for
                                                                          row in matrices[mtx_keys[i]]]
                        # place the color in the correct branching order:
                        color_order[k:k+1] = color_order[k], cmap[i+1]
                    else:   # if two parents where found, something went wrong!
                        print("went wrong in plot ", i, " iteration k =", k)

            # check if determined branching oder is well defined:
            if check != 1:
                warnings.warn(
                    'The data is not processed correctly because the current script '
                    'cannot determine the correct branching order.',
                    stacklevel=1)
                # return
                nb_plots = i
                print("new number of plots:", nb_plots)
                break

            # append soft predictions for the considered level:
            y_plot.append(copy.deepcopy(y))  # Make a copy of y and append the copy to y_plot
            # append colors for the considered level:
            colors_per_plot.append(copy.deepcopy(color_order))


    # Sorting individuals within predefined populations according to their membership
    # in the main cluster of the population in the lowest level:
    if data_generation_mode == 'readVCF':
        unique_pop_membership_old = np.unique(pop_membership)
        unique_pop_membership_sorted_old = np.sort(unique_pop_membership_old)
        unique_pop_membership_sorted = np.array(['YRI', 'LWK', 'GWD', 'MSL', 'ESN',
                                                 'ASW', 'ACB', 'FIN', 'CEU', 'GBR',
                                                 'TSI', 'IBS', 'GIH', 'PJL', 'BEB',
                                                 'STU', 'ITU', 'CHB', 'JPT', 'CHS',
                                                 'CDX', 'KHV', 'MXL', 'PUR', 'CLM',
                                                 'PEL'])
        pop_sizes = np.array([np.sum(pop_membership == pop) for
                                     pop in
                           unique_pop_membership_sorted])
        print("pop sizes 1000G:", pop_sizes)

    else:
        pop_membership = pop_membership.astype(np.int64)
        print("len(pop_membership):", len(pop_membership))
        if len(pop_membership) != n:
            warnings.warn(
                'Population membership for individuals does not add up.',
                stacklevel=1)
            return
        pop_sizes = np.bincount(pop_membership) # population sizes

    nb_pop = len(pop_sizes)  # number of populations
    pop_member_idx = []  # list with boundaries of population affiliation
    # fill pop_member_idx with boundaries of population affiliation:
    pop_member_idx.append(0)
    for i in range(0, nb_pop):
        pop_member_idx.append(pop_member_idx[i] + pop_sizes[i])

    if sorting_level == "lowest":
        print("sort according to lowest level.")
        indv_sorted = []  # list to save individual idx sorted
        y_plot_sorted = []  # sorted soft predictions per level
        cluster_coeff = np.zeros((nb_pop, nb_plots + 1))
        # fill cluster_coeff with contribution of each ancestral population to each
        # geographical population:
        for i in range(0, nb_pop):
            for j in range(0, nb_plots + 1):
                cluster_coeff[i, j] = np.sum(y_plot[-1][j][pop_member_idx[i]:
                                                           pop_member_idx[i + 1]])
            # get ancestral population with most impact for each geographical
            # population to sort their individuals accordingly:
            major_cluster = np.argmax(cluster_coeff[i, :])
            indv_sorted.extend((np.array(y_plot[-1][major_cluster][pop_member_idx[i]:
                                                                   pop_member_idx[i +
                                                                                  1]]).argsort(

            )[::-1] + pop_member_idx[i]).tolist())

        # sort soft predictions of all levels according to the obtained individuals order
        # from the last level
        for j in range(0, nb_plots):
            y_sorted = []
            for m in range(len(y_plot[j])):
                y_sorted.append([y_plot[j][m][i] for i in indv_sorted])
            y_plot_sorted.append(copy.deepcopy(y_sorted))
        print("data sorting done.")

    if sorting_level == "all":
        print("sort according to each level separately. Sorting of individuals within populations might not be comparable between different levels.")
        indv_sorted = []    # list to save individual idx sorted
        y_plot_sorted = []  # sorted soft predictions per level
        # fill cluster_coeff with contribution of each ancestral population to each
        # geographical population for each level separately:
        for level in range(0, nb_plots):
            indv_sorted_level = []  # list to save individual idx sorted for each level
            cluster_coeff_level = np.zeros((nb_pop, level+2))
            for i in range(0, nb_pop):
                for j in range(0, level + 2):
                    cluster_coeff_level[i, j] = np.sum(y_plot[level][j][
                                                       pop_member_idx[i]:
                                                           pop_member_idx[i + 1]])

                # major_cluster = np.argmax(cluster_coeff_level[i, :])
                # indv_sorted.extend(
                #     (np.array(y_plot[level][major_cluster][pop_member_idx[i]:
                #                                         pop_member_idx[i +
                #                                                        1]]).argsort(
                #
                #     )[::-1] + pop_member_idx[i]).tolist())

                # sort ancestral population according to their impact for each
                # geographical population to sort their individuals accordingly:
                major_clusters = np.argsort(cluster_coeff_level[i, :])[::-1]
                # create list to of indv. to be sorted
                indv_sorted_level_pop = list(range(pop_member_idx[i+1] - pop_member_idx[
                    i]))
                # sort indv_sorted_level according to major_clusters[0] and secondary
                # to major_clusters[1:]:
                indv_sorted_level_pop.sort(key=lambda idx: (y_plot[level][
                                                            major_clusters[0]][
                                                            pop_member_idx[i]:
                                                        pop_member_idx[i +
                                                                       1]][
                    idx], secondary_sort([column[pop_member_idx[i]:pop_member_idx[
                    i+1]] for column in y_plot[level]], major_clusters, idx)))
                indv_sorted_level.extend([x + pop_member_idx[i] for x in
                                     indv_sorted_level_pop][
                                    ::-1])
                # # indv_sorted enthält nun die sortierten Indizes basierend auf der Priorität der Bedeutung und der sekundären Sortierung
                # print(indv_sorted[::-1])

            indv_sorted.append(indv_sorted_level)
        # sort soft predictions of all levels according to the obtained individuals order
        # from the last level
        for j in range(0, nb_plots):
            y_sorted = []
            for m in range(len(y_plot[j])):
                y_sorted.append([y_plot[j][m][i] for i in indv_sorted[j]])
            y_plot_sorted.append(copy.deepcopy(y_sorted))
        print("data sorting done.")

    #y_plot_sorted = y_plot

    # stacked bar plots:
    fig, axs = plt.subplots(nb_plots, figsize=(10, 6)) #20 40
    #fig.suptitle('Tangles', fontsize=20)
    fig.tight_layout()
    # Create the grid
    for subplot in axs:
        subplot.set_facecolor('white')
        subplot.set_xlim([-0.6, n - 0.4])
        subplot.set_yticks([])
        subplot.set_xticks([])
        if data_generation_mode == 'out_of_africa':
            subplot.set_xticks(
                [((x + 0.5) * n / nb_pop - 0.5) for x in range(0, nb_pop)])
            subplot.set_xticklabels(['YRI', 'CEU', 'CHB'])
        elif data_generation_mode == 'readVCF':
            #subplot.set_xticks([330, 834, 1260, 1763, 2259])
            #subplot.set_xticklabels(['AFR', 'AMR', 'EAS', 'EUR', 'SAS'])
            subplot.set_xticks([54, 158, 264, 363, 455, 535, 613, 711, 810, 905,
                                1004, 1111, 1216, 1315, 1406, 1500, 1602, 1705, 1808,
                                1913, 2012, 2108, 2189, 2273, 2372, 2462])
            subplot.set_xticklabels(['YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB',
                                     'FIN', 'CEU', 'GBR', 'TSI', 'IBS', 'GIH', 'PJL',
                                     'BEB', 'STU', 'ITU', 'CHB', 'JPT', 'CHS', 'CDX',
                                     'KHV', 'MXL', 'PUR', 'CLM', 'PEL'])
        else:
            subplot.set_xticks(
                np.cumsum(pop_sizes) - pop_sizes / 2 -0.5)
            subplot.set_xticklabels(list(string.ascii_uppercase[:nb_pop]))


    pos_pop_sep = np.cumsum(pop_sizes) - 0.5

    # Stacked bar chart with loop
    for j in range(0, nb_plots):
        for m in range(len(y_plot_sorted[j])):
            axs[j].bar(indv, y_plot_sorted[j][m], bottom=np.sum(y_plot_sorted[j][:m],axis=0),
                      color=colors_per_plot[j][m], width=1)
            #print("y u:", y_plot[j][m])
            #print("y s:", y_plot_sorted[j][m])
            #print("indv u:", indv)
            #print("indv s:", indv_sorted)
            #axs[j].set_xticklabels([str(x) for x in indv])
            #axs[j].xticks(indv, names, fontweight='bold')

        for pos in pos_pop_sep:
            axs[j].axvline(x=pos, color='black', linestyle='-', linewidth=0.5)
    #axs[j].set_xticks([((x + 0.5) * n / nb_pop - 0.5) for x in range(0,
    #                                                                         nb_pop)])
    #axs[j].set_xticklabels(list(string.ascii_uppercase[:nb_pop]))
    # for p, color in zip(axs.patches, cmap):
    #     p.set_facecolor(color)

    #plt.set_xticks([((x + 0.5) * n / nb_pop - 0.5) for x in range(0, nb_pop)])
    #plt.set_xticklabels(list(string.ascii_uppercase[:nb_pop]))
    # Fügen Sie vertikale schwarze Linien am Anfang und am Ende jeder Population hinzu


    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig('plots/tangles_plot_'+ data_generation_mode +'_n_' +str(n)
                +'_a_' + str(agreement)
                + '_seed_' + str(seed) + "_" + cost_fct + '.jpeg', format =
    'jpeg')
    plt.show()

    if plot_ADMIXTURE == True:
        if nb_plots > 12:
            print("restricted number of ADMIXTURE plots to 12.")
            nb_plots = 12
        if ADMIXTURE_file_name == "":
            warnings.warn(
                'Specify file name for ADMIXTURE!',
                stacklevel=1)
            return

        fig, axs = plt.subplots(nb_plots, figsize=(50, 30))
        #fig.suptitle('ADMIXTURE', fontsize=20)
        fig.tight_layout()
        # Create the grid
        for subplot in axs:
            subplot.set_facecolor('white')
            subplot.set_xlim([-0.6, n - 0.4])
            # subplot.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            subplot.set_yticks([])
            subplot.set_xticks([])
            subplot.set_xticks(
                [((x + 0.5) * n / nb_pop - 0.5) for x in range(0, nb_pop)])
            if data_generation_mode == 'out_of_africa':
                subplot.set_xticklabels(['YRI', 'CEU', 'CHB'])
            else:
                subplot.set_xticklabels(list(string.ascii_uppercase[:nb_pop]))
        # ADMIXTURE stacked bar plot:
        for j in range(0, nb_plots):
            K = j+2
            subprocess.run(
                ["bash", "admixture/P_Q/admixture_loop.sh", ADMIXTURE_file_name, str(K)])

            with open("admixture/P_Q/" + ADMIXTURE_file_name + '.' + str(K) + '.Q') \
                    as f:
                Q = np.loadtxt(f, dtype=str, delimiter='\n')
            Q = [list(map(float, q.split())) for q in Q]
            Q = np.array(Q).T
            Q = Q.tolist()
            Q_sorted = []
            for m in range(len(Q)):
                Q_sorted.append([Q[m][i] for i in indv_sorted])
            #print("Q:", Q)
            #print(len(Q))
            for m in range(len(Q)):
                axs[j].bar(indv, Q_sorted[m], bottom=np.sum(Q_sorted[:m], axis=0),
                           color=colors_per_plot[j][m], width=1)
            for pos in pos_pop_sep:
                axs[j].axvline(x=pos, color='black', linestyle='-', linewidth=0.5)
        axs[j].set_xticks(np.cumsum(pop_sizes) - pop_sizes / 2 -0.5)
        axs[j].set_xticklabels(list(string.ascii_uppercase[:nb_pop]))
        # for p, color in zip(axs.patches, cmap):
        #     p.set_facecolor(color)

        # plt.set_xticks([((x + 0.5) * n / nb_pop - 0.5) for x in range(0, nb_pop)])
        # plt.set_xticklabels(list(string.ascii_uppercase[:nb_pop]))
        plt.subplots_adjust(wspace=0, hspace=0.1)
        plt.savefig('plots/ADMIXTURE_plot_' + data_generation_mode + '_n_' + str(n)
                    + '_a_' + str(agreement)
                    + '_seed_' + str(seed) + "_" + cost_fct + '.jpeg', format =
    'jpeg')
        plt.show()
        print("admixture like plots done.")




# Function for secondary sorting of populations
def secondary_sort(y_plot, major_clusters, idx):
    sort_keys = []
    for imp in major_clusters[1:]:
        diff = np.array(y_plot)[imp] - np.array(y_plot)[major_clusters[0]]
    sort_keys.append(diff[idx])
    return tuple(sort_keys)

# this function works, but is not cleaned up yet. Basically the same as the function
# above, just that the tangles plot and ADMIXTURE plot are now plotted together s.t.
# for each level the two methods can be compared directly.
def admixture_comparison_plot(matrices, pop_membership, agreement,
                              plot_ADMIXTURE=False, ADMIXTURE_file_name=""):
    n = np.array(matrices[1]).shape[0]  # number of samples
    nb_plots = len(matrices)  # number of plots to generate
    mtx_keys = list(matrices.keys())
    indv = list(range(0, n))  # list of individuals for x-axis of plot
    y = []  # list of soft pred in each level
    y_plot = []  # list of soft pred to be plotted on y-axis
    color_order = []  # list to save consistent order of colors
    colors_per_plot = []  # list of colors for each bar plot

    # create list of colors to plot from (cyclic color palette):
    cmap = sns.color_palette("husl", nb_plots + 1)
    # swap first color with middle color to have better color contrast:
    cmap[1], cmap[np.ceil((nb_plots + 1) / 2).astype(int)] = cmap[np.ceil(
        (nb_plots + 1) / 2).astype(int)], cmap[1]

    # fill y with soft predictions for each plot/ level of soft clustering
    # in same loop save color order for consistency throughout the bar plots
    for i in range(0, nb_plots):
        # print("i = ", i)
        if i == 0:
            y.append([row[0] for row in matrices[mtx_keys[i]]])
            y.append([row[1] for row in matrices[mtx_keys[i]]])
            color_order.append(cmap[0])
            color_order.append(cmap[1])
            y_plot.append(
                copy.deepcopy(y))  # Make a copy of y and append the copy to y_plot
            colors_per_plot.append(copy.deepcopy(color_order))

        else:
            split = [sum(s) for s in zip([row[0] for row in matrices[mtx_keys[i]]],
                                        [row[1] for row in matrices[mtx_keys[i]]])]
            check = 0
            for k in range(0, len(y)):
                # print("I am here")
                # print(split)
                # print(y[k])
                if np.allclose(np.array(split), np.array(y[k]), 1e-15):
                    # if np.abs(np.array(split) - np.array(y[k])) < 1e-10:
                    check = check + 1
                    if check == 1:
                        # print("worked")
                        y[k:k + 1] = [row[0] for row in matrices[mtx_keys[i]]],\
                            [row[1] for row in matrices[mtx_keys[i]]]
                        color_order[k:k + 1] = color_order[k], cmap[i + 1]
                    else:
                        print("went wrong in plot ", i, " iteration k =", k)

            # print("check = ", check)
            # check if determined branching oder is well defined:
            if check != 1:
                warnings.warn(
                    'The data is not processed correctly because the current script '
                    'cannot determine the correct branching order.',
                    stacklevel=1)
                return

            y_plot.append(copy.deepcopy(y))  # Make a copy of y and append the copy to y_plot
            colors_per_plot.append(copy.deepcopy(color_order))

    # Sorting individuals within predefined populations according to their membership
    # in the main cluster of the population in the lowest level:
    if len(pop_membership) != n:
        warnings.warn(
            'Population membership for individuals does not add up.', stacklevel=1)
        return
    pop_sizes = np.bincount(pop_membership)
    print("pop sizes:", pop_sizes)
    nb_pop = len(pop_sizes)
    print("nb of pop:", nb_pop)
    pop_member_idx = []
    pop_member_idx.append(0)
    for i in range(0, nb_pop):
        pop_member_idx.append(pop_member_idx[i] + pop_sizes[i])
    print("pop idx:", pop_member_idx)
    print("number of plots:", nb_plots)
    # print(len(y_plot))
    # print(y_plot)

    indv_sorted = []
    y_plot_sorted = []
    cluster_coeff = np.zeros((nb_pop, nb_plots + 1))
    for i in range(0, nb_pop):
        for j in range(0, nb_plots + 1):
            cluster_coeff[i, j] = np.sum(y_plot[-1][j][pop_member_idx[i]:
                                                           pop_member_idx[i + 1]])
        major_cluster = np.argmax(cluster_coeff[i, :])
        # indv_sorted.extend((np.argsort((-1*y_plot[-1][major_cluster])[
        # pop_member_idx[i]:
        #                                                  pop_member_idx[i+1]]) +
        #                   pop_member_idx[i]).tolist())
        indv_sorted.extend((np.array(y_plot[-1][major_cluster][pop_member_idx[i]:
                                                                   pop_member_idx[i +
                                                                                  1]]).argsort(

            )[::-1] + pop_member_idx[i]).tolist())
        # print("major cluster:", major_cluster)
    # print("cluster coeff:", cluster_coeff)
    # print("ind sorted:", indv_sorted)

    for j in range(0, nb_plots):
        y_sorted = []
        for m in range(len(y_plot[j])):
            # print("before sorting:", y_plot_sorted[j][m])
            y_sorted.append([y_plot[j][m][i] for i in indv_sorted])
            # print("after sorting:", y_plot_sorted[j][m])
            # print("m:", m)
        y_plot_sorted.append(copy.deepcopy(y_sorted))

    print("data sorting done.")

    # print("y_plot[-1]:", y_plot[-1])

    if ADMIXTURE_file_name == "":
        warnings.warn('Specify file name for ADMIXTURE!', stacklevel=1)
        return

    # stacked bar plot:
    fig, axs = plt.subplots(2*nb_plots, figsize=(10, 30))
    #fig.tight_layout()
    # Create the grid
    v = 1
    for subplot in axs:
        subplot.set_facecolor('white')
        subplot.set_xlim([-0.6, n - 0.4])
        # subplot.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        subplot.set_yticks([])
        subplot.set_xticks([])
        if (v%2) ==0:
            subplot.set(ylabel="A" + str(np.ceil((v-1)/2).astype(int)))
        else:
            subplot.set(ylabel="T" +str(np.ceil(v/2).astype(int)))
        v = v +1


            # subplot.set_xticks(indv)
            # subplot.set_xticklabels(["3", "1", "0", "2", "4", "9", "8", "7", "5", "6",
            # "14", "11", "10", "13", "12"])
            # subplot.set_xticks([((x+0.5)*n/nb_pop-0.5) for x in range(0,nb_pop)])
            # subplot.set_xticklabels(list(string.ascii_uppercase[:nb_pop]))

    # Stacked bar chart with loop
    J=0
    for j in range(0, nb_plots):
        for m in range(len(y_plot_sorted[j])):
            axs[J].bar(indv, y_plot_sorted[j][m],
                        bottom=np.sum(y_plot_sorted[j][:m], axis=0),
                        color=colors_per_plot[j][m], width=1)
        J = J + 1
                # print("y u:", y_plot[j][m])
                # print("y s:", y_plot_sorted[j][m])
                # print("indv u:", indv)
                # print("indv s:", indv_sorted)
                # axs[j].set_xticklabels([str(x) for x in indv])
                # axs[j].xticks(indv, names, fontweight='bold')
        K = j + 2
        subprocess.run(["bash", "admixture/P_Q/admixture_loop.sh", ADMIXTURE_file_name,
                 str(K)])

        with open("admixture/P_Q/" + ADMIXTURE_file_name + '.' + str(K) + '.Q') as f:
            Q = np.loadtxt(f, dtype=str, delimiter='\n')
        Q = [list(map(float, q.split())) for q in Q]
        Q = np.array(Q).T
        Q = Q.tolist()
        Q_sorted = []
        for m in range(len(Q)):
            Q_sorted.append([Q[m][i] for i in indv_sorted])
        for m in range(len(Q)):
            axs[J].bar(indv, Q_sorted[m], bottom=np.sum(Q_sorted[:m], axis=0),
                           color=colors_per_plot[j][m], width=1)
        J = J + 1
    axs[J-1].set_xticks([((x + 0.5) * n / nb_pop - 0.5) for x in range(0, nb_pop)])
    axs[J-1].set_xticklabels(list(string.ascii_uppercase[:nb_pop]))
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig('admixture_comparison_plot_n_' + str(n) + '_a_' + str(agreement) +
                '_fst.pdf')
    plt.show()

    #print(matrices)
    #print("done.")


# with open('saved_soft_matrices.pkl', 'rb') as f:
#     loaded_soft_matrices = pickle.load(f)
# #print("loaded matrices:", loaded_soft_matrices)
#
# admixture_like_plot(loaded_soft_matrices)







