import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import warnings
import string
import subprocess
import time

"""
Script to plot the the inferred ancestry, that is the population genetics specific 
soft clustering. As soft clustering is hierarchical by design, the resulting plot is 
also hierarchical. The function is based on matrices, which contain the soft 
clustering level-wise (regarding the tangles tree). Population membership per 
individual (where they have been sampled) is used to receive a meaningful plot, 
agreement parameter and seed is added to be able to distinguish different saved plots. 
The script is divided in the following steps

    1. Choose appropriate color palette
    2. Convert the soft clustering output such that the resulting bar plot is 
    consistent (pay attention to the order of the soft clustering so that the colors
    match the tangles tree).
    3. Sorting individuals within predefined populations according to their 
    membership such that individuals with similar soft clustering are grouped together
    4. Create a bar plot for each level in the soft clustering. This displays the 
    inferred ancestry.
    5. If demanded, compute and plot ADMIXTURE for comparison.
"""


def plot_inferred_ancestry(matrices, pop_membership, agreement, data_generation_mode,
                           seed=[], char_cuts=[], num_char_cuts=[],
                        sorting_level="lowest",
                        plot_ADMIXTURE = False, ADMIXTURE_file_name="", cost_fct = ""):
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
        cmap = sns.color_palette("deep").as_hex()
        # cmap = ['#029e73', '#0173b2', '#de8f05', '#d55e00', '#56b4e9', '#cc78bc', '#949494',
        #          '#fbafe4', '#ece133', '#ca9161', '#004949', '#920000',
        #         '#924900', '#490092', '#b66dff']
        cmap = ['#029e73',
                '#0173b2',
                '#de8f05',
                '#d55e00',
                '#56b4e9',
                '#949494',
                '#cc78bc',
                '#fbafe4',
                '#ece133',
                '#ca9161',
                '#004949',
                '#920000',
                '#924900',
                '#490092',
                '#b66dff']

        # colors used for ADMIXTURE can be adapted to achieve a better compatibility
        # with tangles. This order of the colors is used for the different ADMIXTURE
        # plots in the publication. Note, these orders are only meaningful for a
        # specific seed.
        cmap2 = plt.get_cmap("tab20")
        # minimal example ADMIXTURE
        # ADMIXTURE_colors = [[cmap[1], cmap[0]], [cmap[2], cmap[1], cmap[0]]] # 2
        # ADMIXTURE_colors = [[cmap[1], cmap[0]], [cmap[2], cmap[1], cmap[0]]] # 8
        # ADMIXTURE_colors = [[cmap[0], cmap[1]], [cmap[2], cmap[0], cmap[1]]] # 3
        #ADMIXTURE_colors = [[cmap[1], cmap[0]], [cmap[0], cmap[2], cmap[1]]]  # 1
        # easy sim ADMIXTURE
        # ADMIXTURE_colors = [[cmap[1], cmap[0]],
        #                     [cmap[1], cmap[2], cmap[0]],
        #                     [cmap[3], cmap[1], cmap[0], cmap[2]],
        #                     [cmap[1], cmap[4], cmap[3], cmap[0], cmap[2]],
        #                     [cmap[7], cmap[2], cmap[9], cmap[3], cmap[0], cmap[1]],
        #                     [cmap[2], cmap[1], cmap[4], cmap[0], cmap[3], cmap[7],
        #                      cmap[6]],
        #                     [cmap[1], cmap[0], cmap[2], cmap[4], cmap[7], cmap[3],
        #                      cmap[6], cmap[5]]]
        # ADMIXTURE_colors = [[cmap[1], cmap[0]],
        #                     [cmap[1], cmap[2], cmap[0]],
        #                     [cmap[3], cmap[1], cmap[0], cmap[2]],
        #                     [cmap[1], cmap[4], cmap[3], cmap[0], cmap[2]],
        #                     [cmap[7], cmap[2], cmap[14], cmap[3], cmap[0], cmap[1]],
        #                     [cmap[2], cmap[1], cmap[4], cmap[0], cmap[3], cmap[7],
        #                      cmap[6]],
        #                     [cmap[1], cmap[0], cmap[2], cmap[4], cmap[7], cmap[3],
        #                      cmap[6], cmap[5]],
        #                     # K>8:
        #                     [cmap[7], cmap[0], cmap[1], cmap[6], cmap[3], cmap[2],
        #                      cmap[5], cmap[4], cmap[9]],
        #                     [cmap[14], cmap[5], cmap[12], cmap[4], cmap[6], cmap[1],
        #                      cmap[2], cmap[7], cmap[3], cmap[0]],
        #                     [cmap[0], cmap[10], cmap[5], cmap[13], cmap[14], cmap[2],
        #                      cmap[6], cmap[1], cmap[3], cmap[4], cmap[7]],
        #                     [cmap[6], cmap[3], cmap[0], cmap[14], cmap[8], cmap[5],
        #                      cmap[9], cmap[2], cmap[7], cmap[4], cmap[10], cmap[1]]]

        # complex sim ADMIXTURE K=12
        # ADMIXTURE_colors = [[cmap[4], cmap[0]],
        #                      [cmap[4], cmap[0], cmap[1]],
        #                      [cmap[1], cmap[4], cmap[2], cmap[0]],
        #                      [cmap[4], cmap[3], cmap[0], cmap[1], cmap[2]],
        #                      [cmap[4], cmap[0], cmap[1], cmap[5], cmap[3], cmap[2]],
        #                      [cmap[0], cmap[5], cmap[2], cmap[3], cmap[1], cmap[6],
        #                       cmap[4]],
        #                      [cmap[0], cmap[6], cmap[1], cmap[2], cmap[3], cmap[5],
        #                       cmap[7], cmap[4]],
        #                      [cmap[6], cmap[5], cmap[1], cmap[2], cmap[0], cmap[4],
        #                       cmap[9], cmap[3], cmap[7]],
        #                      [cmap[2], cmap[8], cmap[0], cmap[6], cmap[10], cmap[5],
        #                       cmap[4], cmap[1], cmap[3], cmap[7]],
        #                      [cmap[2], cmap[6], cmap[8], cmap[10], cmap[11], cmap[1],
        #                       cmap[3], cmap[0], cmap[13], cmap[7], cmap[4]],
        #                      [cmap[7], cmap[2], cmap[6], cmap[3], cmap[10], cmap[12],
        #                       cmap[13], cmap[0], cmap[4], cmap[1], cmap[8], cmap[5]]]

        # complex sim ADMIXTURE
        # ADMIXTURE_colors = [[cmap[4], cmap[0]],
        #                     [cmap[4], cmap[0], cmap[1]],
        #                     [cmap[1], cmap[4], cmap[2], cmap[0]],
        #                     [cmap[4], cmap[3], cmap[0], cmap[1], cmap[2]],
        #                     [cmap[4], cmap[0], cmap[1], cmap[5], cmap[3], cmap[2]],
        #                     [cmap[0], cmap[5], cmap[2], cmap[3], cmap[1], cmap[6],
        #                      cmap[4]],
        #                     [cmap[0], cmap[6], cmap[1], cmap[2], cmap[3], cmap[5],
        #                      cmap[7], cmap[4]]]

        # AIMs no AMR ADMIXTURE
        # ADMIXTURE_colors = [[cmap[1], cmap[0]], [cmap[1], cmap[2], cmap[0]],
        #                   [cmap[3], cmap[1], cmap[2], cmap[0]]]
        # complex sim FST
        # cmap = [cmap[4], cmap[0], cmap[1], cmap[3], cmap[7], cmap[2], cmap[5]]
        # ADMIXTURE aims K=6:
        # ADMIXTURE_colors = [[cmap[1], cmap[0]],
        #                     [cmap[1], cmap[2], cmap[0]],
        #                     [cmap[3], cmap[1], cmap[2], cmap[0]],
        #                     [cmap[2], cmap[1], cmap[4], cmap[3], cmap[0]],
        #                     [cmap[5], cmap[1], cmap[0], cmap[4], cmap[2], cmap[3]],
        #                     [cmap[2], cmap[5], cmap[0], cmap[3], cmap[4], cmap[1],
        #                      cmap[6]]]

        # ADMIXTURE migration seed=40
        # ADMIXTURE_colors = [[cmap[1], cmap[0]],
        #                     [cmap[3], cmap[0], cmap[1]],
        #                     [cmap[1], cmap[5], cmap[0], cmap[3]],
        #                     [cmap[2], cmap[4], cmap[0], cmap[1], cmap[3]],
        #                     [cmap[3], cmap[4], cmap[2], cmap[6], cmap[0], cmap[1]],
        #                     [cmap[6], cmap[0], cmap[4], cmap[3], cmap[5], cmap[1],
        #                      cmap[2]],
        #                     [cmap[1], cmap[0], cmap[4], cmap[3], cmap[5], cmap[7],
        #                      cmap[6], cmap[2]],
        #                     [cmap[4], cmap[0], cmap[6], cmap[5], cmap[1], cmap[3],
        #                      cmap[10], cmap[2], cmap[7]],
        #                     [cmap[4], cmap[14], cmap[2], cmap[0], cmap[7], cmap[6],
        #                      cmap[5], cmap[1], cmap[3], cmap[10]],
        #                     [cmap[14], cmap[10], cmap[7], cmap[5], cmap[4], cmap[8],
        #                      cmap[0], cmap[2], cmap[6], cmap[3], cmap[1]],
        #                     [cmap[6], cmap[1], cmap[10], cmap[8], cmap[4], cmap[2],
        #                      cmap[0], cmap[12], cmap[5], cmap[9], cmap[7], cmap[3]]]
        # ADMIXTURE migration seed=48
        # ADMIXTURE_colors = [[cmap[0], cmap[1]],
        #                     [cmap[0], cmap[1], cmap[2]],
        #                     [cmap[2], cmap[3], cmap[1], cmap[0]],
        #                     [cmap[1], cmap[6], cmap[2], cmap[3], cmap[0]],
        #                     [cmap[6], cmap[1], cmap[3], cmap[2], cmap[0], cmap[5]],
        #                     [cmap[4], cmap[2], cmap[5], cmap[1], cmap[0], cmap[3],
        #                      cmap[7]],
        #                     [cmap[4], cmap[12], cmap[0], cmap[1], cmap[5], cmap[6],
        #                      cmap[3], cmap[2]],
        #                     [cmap[0], cmap[5], cmap[6], cmap[1], cmap[8], cmap[4],
        #                      cmap[12], cmap[2], cmap[3]],
        #                     [cmap[14], cmap[2], cmap[6], cmap[7], cmap[4], cmap[0],
        #                      cmap[3], cmap[5], cmap[1], cmap[13]],
        #                     [cmap[12], cmap[14], cmap[2], cmap[10], cmap[5], cmap[4],
        #                      cmap[0], cmap[3], cmap[6], cmap[7], cmap[1]],
        #                     [cmap[12], cmap[14], cmap[7], cmap[4], cmap[10], cmap[1],
        #                      cmap[6], cmap[13], cmap[0], cmap[5], cmap[3], cmap[2]]]
        # ADMIXTURE migration seed=54
        # ADMIXTURE_colors = [[cmap[0], cmap[1]],
        #                     [cmap[3], cmap[1], cmap[0]],
        #                     [cmap[7], cmap[0], cmap[3], cmap[1]],
        #                     [cmap[2], cmap[0], cmap[1], cmap[3], cmap[5]],
        #                     [cmap[1], cmap[6], cmap[7], cmap[0], cmap[3], cmap[2]],
        #                     [cmap[0], cmap[7], cmap[6], cmap[2], cmap[3], cmap[5],
        #                      cmap[1]],
        #                     [cmap[5], cmap[1], cmap[4], cmap[0], cmap[2], cmap[6],
        #                      cmap[3], cmap[7]],
        #                     [cmap[3], cmap[5], cmap[6], cmap[2], cmap[12], cmap[4],
        #                      cmap[7], cmap[1], cmap[0]],
        #                     [cmap[5], cmap[2], cmap[13], cmap[1], cmap[6], cmap[4],
        #                      cmap[10], cmap[3], cmap[0], cmap[7]],
        #                     [cmap[0], cmap[3], cmap[5], cmap[1], cmap[4], cmap[6],
        #                      cmap[11], cmap[2], cmap[13], cmap[10], cmap[7]],
        #                     [cmap[12], cmap[5], cmap[0], cmap[7], cmap[10], cmap[14],
        #                      cmap[3], cmap[4], cmap[1], cmap[2], cmap[6], cmap[8]]]
        # ADMIXTURE migration seed=66
        # ADMIXTURE_colors = [[cmap[1], cmap[0]],
        #                     [cmap[0], cmap[3], cmap[0]],
        #                     [cmap[0], cmap[1], cmap[3], cmap[4]],
        #                     [cmap[1], cmap[0], cmap[6], cmap[5], cmap[4]],
        #                     [cmap[7], cmap[5], cmap[0], cmap[4], cmap[1], cmap[6]],
        #                     [cmap[7], cmap[6], cmap[4], cmap[0], cmap[3], cmap[5],
        #                      cmap[1]],
        #                     [cmap[6], cmap[4], cmap[2], cmap[3], cmap[0], cmap[5],
        #                      cmap[1], cmap[7]],
        #                     [cmap[3], cmap[1], cmap[0], cmap[13], cmap[4], cmap[5],
        #                      cmap[2], cmap[6], cmap[7]],
        #                     [cmap[0], cmap[13], cmap[3], cmap[1], cmap[4], cmap[5],
        #                      cmap[6], cmap[7], cmap[11], cmap[8]],
        #                     [cmap[12], cmap[7], cmap[3], cmap[4], cmap[6], cmap[5],
        #                      cmap[2], cmap[9], cmap[1], cmap[11], cmap[0]],
        #                     [cmap[2], cmap[10], cmap[6], cmap[1], cmap[5], cmap[3],
        #                      cmap[7], cmap[14], cmap[12], cmap[0], cmap[11], cmap[4]]]





    elif nb_plots+1 < 24:
        c = sns.color_palette("husl", 24)   # choose color palette
        # change order of colors to increase color contrast:
        cmap = [c[0], c[12], c[6], c[18], c[3], c[9], c[15], c[21], c[1], c[4], c[7],
                c[10], c[13], c[16], c[19], c[22], c[2], c[5], c[8], c[11], c[14],
                c[17], c[20], c[23]]

        cmap = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999',
    '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3',
    '#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02'
]
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

    # colors per plot tangleGen migration r=0.5:
    # colors_per_plot =   [[cmap[1], cmap[0]],
    #                     [cmap[1], cmap[3], cmap[0]],
    #                     [cmap[4], cmap[1], cmap[3], cmap[0]],
    #                     [cmap[4], cmap[1], cmap[3], cmap[6], cmap[0]],
    #                     [cmap[4], cmap[1], cmap[3], cmap[6], cmap[0],
    #                      cmap[5]],
    #                     [cmap[4], cmap[2], cmap[1], cmap[3], cmap[6],
    #                      cmap[0], cmap[5]],
    #                     [cmap[4], cmap[7], cmap[2], cmap[1], cmap[3],
    #                      cmap[6], cmap[0], cmap[5]]]
    # colors per plot tangleGen migration r=2:
    # colors_per_plot = [[cmap[0], cmap[1]],
    #                    [cmap[7], cmap[0], cmap[1]],
    #                    [cmap[7], cmap[0], cmap[1], cmap[2]],
    #                    [cmap[7], cmap[0], cmap[3], cmap[1], cmap[2]],
    #                    [cmap[7], cmap[0], cmap[3], cmap[1], cmap[2],
    #                     cmap[4]],
    #                    [cmap[7], cmap[0], cmap[5], cmap[3], cmap[1],
    #                     cmap[2], cmap[4]],
    #                    [cmap[7], cmap[0], cmap[5], cmap[3], cmap[6],
    #                     cmap[1], cmap[2], cmap[4]]]
    # colors per plot tangleGen migration r=4:
    # colors_per_plot = [[cmap[1], cmap[0]],
    #                    [cmap[1], cmap[0], cmap[4]],
    #                    [cmap[1], cmap[0], cmap[6], cmap[4]],
    #                    [cmap[1], cmap[0], cmap[5], cmap[6], cmap[4]],
    #                    [cmap[1], cmap[0], cmap[5], cmap[7], cmap[6],
    #                     cmap[4]],
    #                    [cmap[1], cmap[5], cmap[0], cmap[3], cmap[7],
    #                     cmap[6], cmap[4]]]
    # colors per plot tangleGen no c2 in cost function:
    # colors_per_plot = [[cmap[0], cmap[2]],
    #                    [cmap[0], cmap[3], cmap[2]],
    #                    [cmap[0], cmap[1], cmap[3], cmap[2]],
    #                    [cmap[0], cmap[1], cmap[3], cmap[2], cmap[4]],
    #                    [cmap[0], cmap[1], cmap[3], cmap[6], cmap[2],
    #                     cmap[4]],
    #                    [cmap[0], cmap[5], cmap[1], cmap[3], cmap[6],
    #                     cmap[2], cmap[4]],
    #                    [cmap[0], cmap[5], cmap[1], cmap[3], cmap[6],
    #                     cmap[2], cmap[4], cmap[7]]]
    # colors per plot tangleGen c2 in cost function b=0.5:
    # colors_per_plot = [[cmap[0], cmap[1]],
    #                    [cmap[0], cmap[2], cmap[1]],
    #                    [cmap[0], cmap[3], cmap[2], cmap[1]],
    #                    [cmap[0], cmap[3], cmap[4], cmap[2], cmap[1]],
    #                    [cmap[0], cmap[5], cmap[3], cmap[4], cmap[2],
    #                     cmap[1]]]

    cmap = ['#0173b2', '#de8f05', '#029e73', '#920000']
    colors_per_plot = [[cmap[0], cmap[1]],
                       [cmap[0], cmap[1], cmap[2]],
                       [cmap[0], cmap[1], cmap[2], cmap[3]]]

    ## Sorting individuals within predefined populations according to their membership
    # in the main cluster of the population in the lowest level:
    # first compute population sizes, i.e. how many indidividuals have been sampled
    # in each region.
    if data_generation_mode == 'readVCF':   # pop sizes when using data from 1kG project
        unique_pop_membership_sorted = np.array(['YRI', 'LWK', 'GWD', 'MSL', 'ESN',
                                                 'ASW', 'ACB', 'FIN', 'CEU', 'GBR',
                                                 'TSI', 'IBS', 'GIH', 'PJL', 'BEB',
                                                 'STU', 'ITU', 'CHB', 'JPT', 'CHS',
                                                 'CDX', 'KHV'])
        pop_sizes = np.array([np.sum(pop_membership == pop) for pop in
                              unique_pop_membership_sorted])
        # print("pop sizes in 1000G project:", pop_sizes)
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
    fig, axs = plt.subplots(nb_plots, figsize=(50, 20)) #4015 mini ex 5030 sim 5020 aims
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
                               1913, 2012, 2108])
            subplot.set_xticklabels([])
        else:
            subplot.set_xticks(np.cumsum(pop_sizes) - pop_sizes / 2 - 0.5)
            subplot.set_xticklabels([])  # 50 for mini ex

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
            axs[j].axvline(x=pos, color='black', linestyle='-', linewidth=2)

        # highlight indv 3 within mini ex:
        # for pos in [2.5, 3.5]:
        #    axs[j].axvline(x=pos, color='black', linestyle='--', linewidth=4)



        # set limit for y-axis:
        axs[j].set_ylim([0, 1])
        axs[j].set_yticks([])
        # add label to y-axis:
        axs[j].set_ylabel(r"$\ell = $" + str(j+2), rotation=0, fontsize=60,
                          verticalalignment='center', labelpad=15,
                          horizontalalignment='right')
        #axs[j].spines['left'].set_position(('outward', 70))

    # white space between stacked bar plots:
    plt.subplots_adjust(left=0.06)  # 0.06 for mini ex
    plt.subplots_adjust(wspace=0, hspace=0.05, bottom=0.05) #, top=0.98) # bottom 0.07,
    # top 0.98 for mini ex
    # axs[j].set_xticks([])

    # label of x-axis on most fine-grained level:
    if data_generation_mode == 'readVCF':
        axs[j].set_xticks([54, 158, 264, 363, 445, 535, 628, 711, 810, 905,
                            1004, 1111, 1216, 1315, 1406, 1500, 1602, 1705, 1808,
                            1913, 2012, 2108])
        axs[j].set_xticklabels(['YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB',
                                 'FIN', 'CEU', 'GBR', 'TSI', 'IBS', 'GIH', 'PJL',
                                 'BEB', 'STU', 'ITU', 'CHB', 'JPT', 'CHS', 'CDX',
                                 'KHV'], fontsize=60)
    else:
        axs[j].set_xticks(np.cumsum(pop_sizes) - pop_sizes / 2 - 0.5)
        axs[j].set_xticklabels(list(string.ascii_uppercase[:nb_pop]),
                               fontsize=60)
        # for mini ex:
        #axs[j].set_xticks([1, 2, 4.5, 7])
        #axs[j].set_xticklabels(["A", 'individual 3', "B", "C"],
        #                       fontsize=50)  # 50 for mini ex

        # plt.text(0.5, 0.5, 'individual 3', size=50,
        #          horizontalalignment='center',
        #          verticalalignment='center',
        #          transform=axs[0].transAxes)

    # red box around indv 3 for mini ex
    # xmin, xmax = 1.5, 2.5
    # trans = matplotlib.transforms.blended_transform_factory(axs[0].transData,
    #                                                         fig.transFigure)
    # r = matplotlib.patches.Rectangle(xy=(xmin, 0.07), width=xmax - xmin, height=0.91,
    #                                  transform=trans,
    #                                  fc='none', ec='k', lw=10, ls='--')
    # fig.add_artist(r)

    # save resulting figure as jpeg:
    plt.savefig('plots/tangles_plot_'+ data_generation_mode +'_n_' +str(n)
                +'_a_' + str(agreement)+ '_seed_' + str(seed) + "_" + cost_fct
                + '.jpeg', format = 'jpeg')
    plt.show()


    ## if demanded, compute and plot ADMIXTURE for comparison:
    if plot_ADMIXTURE:
        # K cannot exceed 13 to reduce run time. If needed, adapt this:
        if nb_plots > 12:
            print("restricted number of ADMIXTURE plots to 12.")
            nb_plots = 12
        if ADMIXTURE_file_name == "":
            warnings.warn(
                'Specify file name for ADMIXTURE!',
                stacklevel=1)
            return

        nb_plots = 11

        # create nb_plots many ADMIXTURE plots, ADMIXTURE is computed on the fly:
        fig, axs = plt.subplots(nb_plots, figsize=(50, 30)) # 4015 mini ex 5030 sim
        # 5020 aims
        fig.tight_layout()
        for subplot in axs:
            subplot.set_facecolor('white')
            # subplot.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            subplot.set_yticks([])
            subplot.set_xticks([])
            if data_generation_mode == 'readVCF':
                subplot.set_xticks([54, 158, 264, 363, 455, 535, 613, 711, 810, 905,
                                    1004, 1111, 1216, 1315, 1406, 1500, 1602, 1705,
                                    1808, 1913, 2012, 2108])
                subplot.set_xticklabels([])
            else:
                subplot.set_xticks(np.cumsum(pop_sizes) - pop_sizes / 2 - 0.5)
                subplot.set_xticklabels([]) # 50 for mini ex

        # track time for ADMIXTURE:
        times = np.zeros(nb_plots)

        # ADMIXTURE stacked bar plot:
        for j in range(0, nb_plots):
            K = j+2     # set number of ancestral populations to be indetified
            start_time = time.time()    # track time
            # run ADMIXTURE for specified K and seed:
            subprocess.run(
                ["bash", "admixture/P_Q/admixture_loop.sh", ADMIXTURE_file_name,
                 str(K), str(seed)])
            # load resulting Q matrix:
            with open("admixture/P_Q/" + ADMIXTURE_file_name + '.' + str(K) + '.Q') \
                    as f:
                Q = np.loadtxt(f, dtype=str, delimiter='\n')
            # transform Q into list
            Q = [list(map(float, q.split())) for q in Q]
            Q = np.array(Q).T
            Q = Q.tolist()
            end_time = time.time()              # stop time needed
            times[j] = end_time - start_time    # save time needed

            # For comparability, rearrange indv in Q in the same way as for the
            # stacked bar chart of tangles:
            Q_sorted = []
            for m in range(len(Q)):
                Q_sorted.append([Q[m][i] for i in indv_sorted])

            # create stacked bar plot:
            for m in range(len(Q)):
                axs[j].bar(indv, Q_sorted[m], bottom=np.sum(Q_sorted[:m], axis=0),
                           color=ADMIXTURE_colors[j][m], width=1)

            # vertical black lines to separate geographical populations:
            for pos in pos_pop_sep[:-1]:
                axs[j].axvline(x=pos, color='black', linestyle='-', linewidth=2)# 6
                # mini ex
                # set limits for x and y-axis:
            axs[j].set_xlim([-0.5, n - 0.5])
            axs[j].set_ylim([0, 1])
            axs[j].set_yticks([])
            # add label to y-axis:
            axs[j].set_ylabel(r"$K = $" + str(j + 2), rotation=0, fontsize=60,
                              verticalalignment='center', labelpad=15,
                              horizontalalignment='right')

        # white space between stacked bar plots:
        plt.subplots_adjust(left=0.06) # if K > 9 left=0.06 and mini ex
        plt.subplots_adjust(wspace=0, hspace=0.05, bottom=0.05)#, top=0.98) # mini ex
        #plt.subplots_adjust(wspace=0, hspace=0.05, bottom=0.05) # bottom 0.1 mini ex.
        # 0.05 else
        axs[j].set_xticks([])

        # label of x-axis on most fine-grained level:
        if data_generation_mode == 'readVCF':
            axs[j].set_xticks([54, 158, 264, 363, 445, 535, 628, 711, 810, 905,
                               1004, 1111, 1216, 1315, 1406, 1500, 1602, 1705, 1808,
                               1913, 2012, 2108])
            axs[j].set_xticklabels(['YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB',
                                    'FIN', 'CEU', 'GBR', 'TSI', 'IBS', 'GIH', 'PJL',
                                    'BEB', 'STU', 'ITU', 'CHB', 'JPT', 'CHS', 'CDX',
                                    'KHV'], fontsize=60)
        else:
            axs[j].set_xticks(np.cumsum(pop_sizes) - pop_sizes / 2 - 0.5)
            axs[j].set_xticklabels(list(string.ascii_uppercase[:nb_pop]),
                                   fontsize=60)  # 50 for mini ex

        # save resulting figure as jpeg:
        plt.savefig('plots/ADMIXTURE_plot_' + data_generation_mode + '_n_' + str(n)
                    + '_a_' + str(agreement) + '_seed_' + str(seed) + "_" + cost_fct
                    + '.jpeg', format = 'jpeg')
        plt.show()
        #print("run time for ADMIXTURE without plot:", np.sum(times))


# Function for secondary sorting of populations
def secondary_sort(y_plot, major_clusters, idx):
    sort_keys = []
    for imp in major_clusters[1:]:
        diff = np.array(y_plot)[imp] - np.array(y_plot)[major_clusters[0]]
        sort_keys.append(diff[idx])
    return tuple(sort_keys)







