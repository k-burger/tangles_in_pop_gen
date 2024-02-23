import numpy as np
import matplotlib.pyplot as plt

"""
Script to creat a runtime plot for both ADMIXTURE and tangles. runtimes need to be 
computed separately. 
"""

# SNP counts in runtime analysis:
snp_counts = [263, 508, 1031, 5041, 9987, 25132, 50042]

# labels for x-axis
x_labels = [250, 500, 1000, 5000, 10000, 25000, 50000]

# runtimes of each method:
runtimes_tangles = np.array([2.6, 4.6, 8.7, 53.4, 168, 1394.2, 5317.3])
runtimes_tangles_cost_precomputed = np.array([0.5, 0.9, 1.6, 7.9, 15.1, 38, 74.5])
runtimes_ADMIXTURE = np.array([4.5, 10.4, 17.6, 89.3, 220.6, 586, 1264.5])

# creat the plot:
fig, ax = plt.subplots(figsize=(10, 6))
# logarithmic scale for x and y-axis:
plt.yscale('log')
plt.xscale('log')
# plot:
plt.plot(snp_counts, runtimes_tangles, marker='o', label='Tadmixture')
plt.plot(snp_counts, runtimes_tangles_cost_precomputed, marker='s', label='Tadmixture with precomputed costs')
plt.plot(snp_counts, runtimes_ADMIXTURE, marker='^', label='ADMIXTURE')

plt.minorticks_off()
ax.set_xticks([263, 508, 1031, 5041, 9987, 25132, 50042], fontsize=14)
ax.set_xticklabels([250, 500, 1000, 5000, 10000, 25000, 50000], fontsize=14)
plt.xlabel('number of SNPs', fontsize=14)
plt.ylabel('runtime (in sec)', fontsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tick_params(axis='x', labelsize=14)

# legend:
plt.legend(fontsize=14)
# save plot as jpeg:
plt.tight_layout()
plt.savefig('plots/runtime_plot.jpeg', format = 'jpeg')
plt.show()
