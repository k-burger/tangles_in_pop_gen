import numpy as np
import matplotlib.pyplot as plt

"""
Script to creat a runtime plot for both ADMIXTURE and tangles. runtimes need to be 
computed separately. 
"""

cmap = ['#029e73', '#0173b2', '#de8f05', '#d55e00', '#56b4e9', '#949494', '#cc78bc',
        '#fbafe4', '#ece133', '#ca9161', '#004949', '#920000', '#924900', '#490092',
        '#b66dff']

# SNP counts in runtime analysis:
snp_counts = [263, 508, 1031, 5041, 9987, 25132, 50042]

# labels for x-axis
x_labels = [250, 500, 1000, 5000, 10000, 25000, 50000]

# runtimes of each method:
runtimes_tangles = np.array([2.6, 4.6, 8.7, 53.4, 168, 1394.2, 5317.3])
runtimes_tangles_cost_precomputed = np.array([0.5, 0.9, 1.6, 7.9, 15.1, 38, 74.5])
runtimes_ADMIXTURE = np.array([4.5, 10.4, 17.6, 89.3, 220.6, 586, 1264.5])
runtimes_fastStructure = np.array([29.1, 39.3, 67.3, 351.4, 490.2, 763.1, 883.1])
runtimes_SCOPE = np.array([18.8, 19.6, 19.4, 24.2, 31.02, 36.1, 39.8])

# creat the plot:
fig, ax = plt.subplots(figsize=(10, 6))
# logarithmic scale for x and y-axis:
plt.yscale('log')
plt.xscale('log')
# plot:
plt.plot(snp_counts, runtimes_tangles, marker='o', label='tangleGen', color=cmap[0])
plt.plot(snp_counts, runtimes_tangles_cost_precomputed, marker='s', label='tangleGen '
                                                                          'with '
                                                                          'precomputed costs', color=cmap[1])
plt.plot(snp_counts, runtimes_ADMIXTURE, marker='^', label='ADMIXTURE', color=cmap[6])
plt.plot(snp_counts, runtimes_fastStructure, marker='*', label='fastStructure',
         color=cmap[3])
plt.plot(snp_counts, runtimes_SCOPE, marker='p', label='SCOPE', color=cmap[4])

plt.minorticks_off()
ax.set_xticks([263, 508, 1031, 5041, 9987, 25132, 50042], fontsize=16)
ax.set_xticklabels([250, 500, 1000, 5000, 10000, 25000, 50000], fontsize=16)
plt.xlabel('number of SNPs', fontsize=16)
plt.ylabel('runtime (in sec)', fontsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.tick_params(axis='x', labelsize=16)

# legend:
plt.legend(fontsize=16)
# save plot as jpeg:
plt.tight_layout()
plt.savefig('plots/runtime_plot_extended.jpeg', format = 'jpeg', dpi=300)
plt.show()




# sample counts in runtime analysis:
sample_counts = [400, 800, 1600, 3200, 6400]

# runtimes of each method:
runtimes_tangles = np.array([26.29, 52.63, 165.8, 543.78, 1894.48])
runtimes_tangles_cost_precomputed = np.array([6.53, 7.96, 9.46, 11.99, 18.18])
runtimes_ADMIXTURE = np.array([42.17, 102.81, 264.12, 451.86, 1357.93])
runtimes_fastStructure = np.array([265.95, 284.58, 454.22, 878.02, 3052.4])
runtimes_SCOPE = np.array([20.8, 21.55, 23.53, 28.13, 53.18])

# creat the plot:
fig, ax = plt.subplots(figsize=(10, 6))
# logarithmic scale for x and y-axis:
plt.yscale('log')
#plt.xscale('log')
# plot:
plt.plot(sample_counts, runtimes_tangles, marker='o', label='tangleGen', color=cmap[0])
plt.plot(sample_counts, runtimes_tangles_cost_precomputed, marker='s', label='tangleGen '
                                                                          'with '
                                                                          'precomputed costs', color=cmap[1])
plt.plot(sample_counts, runtimes_ADMIXTURE, marker='^', label='ADMIXTURE', color=cmap[6])
plt.plot(sample_counts, runtimes_fastStructure, marker='*', label='fastStructure',
         color=cmap[3])
plt.plot(sample_counts, runtimes_SCOPE, marker='p', label='SCOPE', color=cmap[4])
# plt.minorticks_off()
# ax.set_xticks([263, 508, 1031, 5041, 9987, 25132, 50042], fontsize=14)
# ax.set_xticklabels([250, 500, 1000, 5000, 10000, 25000, 50000], fontsize=14)
plt.xlabel('number of samples', fontsize=16)
plt.ylabel('runtime (in sec)', fontsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.tick_params(axis='x', labelsize=16)

# legend:
#plt.legend(fontsize=14)
# save plot as jpeg:
plt.tight_layout()
plt.savefig('plots/runtime_plot_extended_samples.jpeg', format = 'jpeg', dpi=300)
plt.show()