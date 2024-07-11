import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import warnings
import string
import subprocess

"""
Script to plot the the inferred ancestry, that is the population genetics specific 
soft clustering. As soft clustering is hierarchical by design, the resulting plot is 
also hierarchical. The function is based on matrices, which contain the soft 
clustering level-wise (regarding the tangles tree). Population membership per 
individual (where they have been sampled) is used to receive a meaningful plot, 
agreement parameter and seed is added to be able to distinguish different saved 
plots. The script is divided in the following steps

    1. Choose appropriate color palette
    2. Convert the soft clustering output such that the resulting bar plot is 
    consistent (pay attention to the order of the soft clustering so that the colors
    match the tangles tree).
    3. Sorting individuals within predefined populations according to their 
    membership such that individuals with similar soft clustering are grouped together
    4. Create a bar plot for each level in the soft clustering. This displays the 
    inferred ancestry.
"""


def plot_inferred_ancestry(matrices, pop_membership, agreement,
                        data_generation_mode, seed=[], char_cuts=[], num_char_cuts=[],
                        sorting_level="", cost_fct = ""):
    n = np.array(matrices[1]).shape[0]      # number of indv
    nb_plots = len(matrices)                # number of plots to generate
    mtx_keys = list(matrices.keys())        # get keys of the matrices dictionary
    indv = list(range(0, n))                # list of individuals for x-axis of plot
    y = []                                  # list of soft pred in each level
    y_plot = []                             # list of soft pred to be plotted on y-axis
    color_order = []                        # list to save consistent order of colors
    colors_per_plot = []                    # list of colors for each bar plot

    ## create list of colors to plot from (from named colors or cyclic color palette):
    if nb_plots+1 < 11:
        # color palette in publication:
        cmap = ['#0173b2', '#de8f05', '#029e73', '#920000', '#b66dff', '#924900',
                '#b66dff', '#fbafe4', '#d55e00',
                '#56b4e9',
                '#949494',
                '#ece133', '#ca9161', '#004949', '#490092', '#cc78bc'
                ]
    elif nb_plots+1 < 24:
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

    ## convert the soft clustering output such that the resulting bar plot is
    # consistent (pay attention to the order of the soft clustering so that the colors
    # match the tangles tree).Therefore, fill y with soft predictions for each plot/
    # level of soft clustering. In same loop save color order for consistency
    # throughout the bar plots.
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

    ## Sorting individuals within predefined populations according to their membership
    # in the main cluster of the population in the lowest level:
    # first compute population sizes, i.e. how many indidividuals have been sampled
    # in each region.
    if data_generation_mode == 'readVCF':   # pop sizes when using data from 1kG project
        unique_pop_membership_sorted = np.array(['YRI', 'LWK', 'GWD', 'MSL', 'ESN',
                                                 'ASW', 'ACB', 'FIN', 'CEU', 'GBR',
                                                 'TSI', 'IBS', 'GIH', 'PJL', 'BEB',
                                                 'STU', 'ITU', 'CHB', 'JPT', 'CHS',
                                                 'CDX', 'KHV', 'MXL', 'PUR', 'CLM',
                                                 'PEL'])
        pop_sizes = np.array([np.sum(pop_membership == pop) for pop in
                           unique_pop_membership_sorted])
        print("pop sizes in 1000G project:", pop_sizes)
    else:
        pop_membership = pop_membership.astype(np.int64)
        print("len(pop_membership):", len(pop_membership))
        if len(pop_membership) != n:
            warnings.warn(
                'Population membership for individuals does not add up.',
                stacklevel=1)
            return
        pop_sizes = np.bincount(pop_membership) # population sizes

    # initialise variables needed for sorting:
    nb_pop = len(pop_sizes)  # number of populations
    pop_member_idx = []  # list with boundaries of population affiliation
    # fill pop_member_idx with boundaries of population affiliation:
    pop_member_idx.append(0)
    for i in range(0, nb_pop):
        pop_member_idx.append(pop_member_idx[i] + pop_sizes[i])

    # start sorting. sorting_level == "lowest" indicates that individuals are sorted
    # by the lowest level, i.e. the most fine-grained inferred population structure
    # and this sorting is then used for all other levels.
    if sorting_level == "lowest":
        print("sort the indv. per population according to the finest-grained clustering and keep this sorting for all other clustering levels.")
        indv_sorted = []  # list to save individual idx sorted
        y_plot_sorted = []  # sorted soft predictions per level
        cluster_coeff = np.zeros((nb_pop, nb_plots + 1))
        # fill cluster_coeff with contribution of each ancestral population to each
        # geographical population for each level separately:
        for i in range(0, nb_pop):
            # get impact of each ancestral population to geographical population:
            for j in range(0, nb_plots + 1):
                cluster_coeff[i, j] = np.sum(y_plot[-1][j][pop_member_idx[i]:pop_member_idx[i+1]])
            # sort ancestral population according to their impact for each
            # geographical population to sort their individuals accordingly:
            major_clusters = np.argsort(cluster_coeff[i, :])[::-1]
            # create list to of indv. to be sorted
            indv_sorted_pop = list(range(pop_member_idx[i + 1] - pop_member_idx[i]))
            # sort indv_sorted_level according to major_clusters[0] and secondary
            # to major_clusters[1:]:
            indv_sorted_pop.sort(key=lambda idx: (y_plot[-1][major_clusters[0]]
                [pop_member_idx[i]:pop_member_idx[i+1]][idx], secondary_sort(
                [column[pop_member_idx[i]:pop_member_idx[i+1]]
                for column in y_plot[-1]],major_clusters,idx)))
            indv_sorted.extend([x + pop_member_idx[i] for x in indv_sorted_pop][::-1])

        # sort soft predictions of all levels according to the obtained individuals order
        # from the last level
        for j in range(0, nb_plots):
            y_sorted = []
            for m in range(len(y_plot[j])):
                y_sorted.append([y_plot[j][m][i] for i in indv_sorted])
            y_plot_sorted.append(copy.deepcopy(y_sorted))
        print("data sorting done.")

    ## stacked bar plots:
    fig, axs = plt.subplots(nb_plots, figsize=(50, 22)) #50 30 25 15
    fig.tight_layout()
    for subplot in axs:
        subplot.set_facecolor('white')
        subplot.set_xlim([-0.5, n - 0.5])
        axs[j].set_ylim([0, 1])
        subplot.set_yticks([])
        subplot.set_xticks([])
        if data_generation_mode == 'readVCF':
            subplot.set_xticks([54, 158, 264, 363, 455, 535, 613, 711, 810, 905,
                                1004, 1111, 1216, 1315, 1406, 1500, 1602, 1705, 1808,
                                1913, 2012, 2108, 2189, 2273, 2372, 2462])
            subplot.set_xticklabels([])
        else:
            subplot.set_xticks(np.cumsum(pop_sizes) - pop_sizes / 2 -0.5)
            subplot.set_xticklabels(list(string.ascii_uppercase[:nb_pop]),
                                    fontsize=35) # 50 for mini ex

    # get position of boarders of geographical population:
    pos_pop_sep = np.cumsum(pop_sizes) - 0.5

    # Stacked bar chart with loop
    for j in range(0, nb_plots):
        # for each level, loop through ancestral populations and stack the bar
        # charts. use specified colors.
        for m in range(len(y_plot_sorted[j])):
            axs[j].bar(indv, y_plot_sorted[j][m], bottom=np.sum(
                y_plot_sorted[j][:m],axis=0),
                      color=colors_per_plot[j][m], width=1)
        # vertical black lines to separate geographical populations:
        for pos in pos_pop_sep[:-1]:
            axs[j].axvline(x=pos, color='black', linestyle='-')
        # set limit for y-axis:
        axs[j].set_ylim([0, 1])
        axs[j].set_yticks([])
        # add label to y-axis:
        axs[j].set_ylabel(r"$\ell = $" + str(j + 2), rotation=0, fontsize=60,
                          verticalalignment='center', labelpad=15,
                          horizontalalignment='right')

    # white space between stacked bar plots:
    plt.subplots_adjust(left=0.05)
    plt.subplots_adjust(wspace=0, hspace=0.05, bottom=0.05)

    # label of x-axis on most fine-grained level:
    if data_generation_mode == 'readVCF':
        subplot.set_xticks([54, 158, 264, 363, 445, 535, 628, 711, 810, 905,
                            1004, 1111, 1216, 1315, 1406, 1500, 1602, 1705, 1808,
                            1913, 2012, 2108, 2189, 2273, 2372, 2462])
        subplot.set_xticklabels(['YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB',
                                 'FIN', 'CEU', 'GBR', 'TSI', 'IBS', 'GIH', 'PJL',
                                 'BEB', 'STU', 'ITU', 'CHB', 'JPT', 'CHS', 'CDX',
                                 'KHV', 'MXL', 'PUR', 'CLM', 'PEL'], fontsize=50)

    # save resulting figure as jpeg:
    plt.savefig('plots/tangleGen_plot_'+ data_generation_mode +'_n_' +str(n)
                +'_a_' + str(agreement)+ '_seed_' + str(seed) + "_" + cost_fct
                + '.jpeg', format = 'jpeg')
    plt.show()


# Function for secondary sorting of populations
def secondary_sort(y_plot, major_clusters, idx):
    sort_keys = []
    for imp in major_clusters[1:]:
        diff = np.array(y_plot)[imp] - np.array(y_plot)[major_clusters[0]]
        sort_keys.append(diff[idx])
    return tuple(sort_keys)