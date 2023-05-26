import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import matplotlib.ticker as mticker


def admixture_like_plot(matrices):
    n = np.array(matrices[1]).shape[0]      # number of samples
    nb_plots = len(matrices)                # number of plots to generate
    mtx_keys = list(matrices.keys())
    indv = list(range(0, n))                # list of individuals for x-axis of plot
    y = []                                  # list of soft pred in each level
    y_plot = []                             # list of soft pred to be plotted on y-axis
    #color_list = ['r', 'b', 'y', 'g', 'k']  # list of colors to plot from

    # fill y with soft predictions for each plot/ level of soft clustering
    # challenge: if not root split, then only part of the bar plot is to be updated.
    for i in range(0, nb_plots):
        if i == 0:
            y.append([row[0] for row in matrices[mtx_keys[i]]])
            y.append([row[1] for row in matrices[mtx_keys[i]]])
            y_plot.append(copy.deepcopy(y))  # Make a copy of y and append the copy to y_plot

        else:
            split = [sum(s) for s in zip([row[0] for row in matrices[mtx_keys[i]]],
                                         [row[1] for row in matrices[mtx_keys[i]]])]
            for k in range(0,len(y)):
                if split == y[k]:
                    y[k:k+1] = [row[0] for row in matrices[mtx_keys[i]]], [row[1] for
                                                                          row in matrices[mtx_keys[i]]]
            y_plot.append(copy.deepcopy(y))  # Make a copy of y and append the copy to y_plot

    # stacked bar plot:
    fig, axs = plt.subplots(nb_plots, figsize=(8, 12))
    fig.suptitle('Admixture like plot with tangles')
    #print(indv)
    names = []
    # names = ["","","A","","","","", "B","","","","", "C", "","",]
    # Set the ticks and ticklabels for all axes
    plt.setp(axs, xticks=indv, xticklabels=names,
             yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Stacked bar chart with loop
    for j in range(0, nb_plots):
        for m in range(len(y_plot[j])):
            axs[j].bar(indv, y_plot[j][m], bottom=np.sum(y_plot[j][:m], axis=0))
            #axs[j].set_xticklabels([str(x) for x in indv])
            #axs[j].xticks(indv, names, fontweight='bold')

    plt.savefig('admixture_like_plot.pdf')
    plt.show()

    #print(matrices)
    #print("done.")


# with open('saved_soft_matrices.pkl', 'rb') as f:
#     loaded_soft_matrices = pickle.load(f)
# #print("loaded matrices:", loaded_soft_matrices)
#
# admixture_like_plot(loaded_soft_matrices)







