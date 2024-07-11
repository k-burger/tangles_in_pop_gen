# Inferring Ancestry with the Hierarchical Soft Clustering Approach tangleGen

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)
![tests](https://github.com/tml-tuebingen/tangles/workflows/pytesting/badge.svg)

<p align="center">
  <img src="/plots/plot_hierarchy.png" 
width="500"/>
</p>

> [!NOTE]
> This repository contains the code for the publication 'Inferring Ancestry  with the Hierarchical Soft Clustering Approach tangleGen' (doi: https://doi.org/10.1101/2024.03.27.586940).

tangleGen is a hierarchical clustering method designed for inferring ancestral 
relationships in population genetics. This method exploits the flexibility and robust 
hierarchical functionality of the Tangles framework, available as a Python package 
[here](https://github.com/tml-tuebingen/tangles). With tangleGen, we introduce several 
novel components specifically tailored for ancestry inference: the definition of 
cuts, cost function and soft clustering.

## Method Overview

To infer ancestries, tangleGen proceeds in four steps:

1. **Constructing basic cuts on the set of individuals:** The foundation of 
  tangleGen are
many bipartitions on the set of individuals, dividing them into two groups. Thus, 
bipartitions are referred as ”cuts”. Based on single nucleotide polymorphisms (SNPs), 
individuals are divided into two groups: those homozygous for the ancestral allele and 
the others.
2. **Assigning costs to cuts to sort them for the hierarchical clustering:** A 
well-chosen cost function favors cuts with higher discriminative power by assigning them 
  lower costs, while penalizing cuts that separate closely related groups of 
  individuals. The cost function for inferring ancestries is based on mean FST values 
  and incorporates k-nearest neighbors. The correspondingly sorted cuts enable the next
  iterative hierarchical clustering step.
3. **Iteratively composing the tangles tree by orienting cuts:** Beginning with the 
  lowest-cost cut, the algorithm iteratively evaluates the cuts, orienting them to 
  delimit clusters of individuals. Such a meaningful orientation of a subset of 
   cuts that identifies a cluster is called a "tangle". These tangles form the basis for
  constructing a tangles tree, which represents the resulting hierarchical cluster 
  structure.
4. **Computing a soft clustering based on characteristic cuts to infer ancestry:** 
  Characteristic cuts are cuts that define the tangles tree and, consequently, the 
   identified cluster structure. Based on them, the soft clustering is computed, a value
   between 0 and 1 for each individual and cluster, indicating how likely an 
  individual belongs to a particular cluster. Finally, the method infers the ancestry of
  individuals based on the soft clustering results.

## Getting Started

> [!TIP]
> A detailed demonstration on how to use tangleGen is added in our demo notebook
[demo.ipynb](https://github.com/k-burger/tangles_in_pop_gen/blob/main/demo.ipynb). 
This notebook provides a step-by-step guide on how to use tangleGen for inferring 
ancestral relationships using the 1000 Genomes data set.


## Setup

Clone the GitHub repository:

```
https://github.com/k-burger/tangles_in_pop_gen.git
```

We recommend to set up a conda environment and install all needed packages as 
specified in `conda_env.yml` via

```
conda env create -f conda_env.yml
```

Activate the conda environment

```
conda activate tangleGen_env
```


## Repository Overview

The repository is organized as follows:
+ the directory `src` contains the tangles algorithmic framework together with the 
  file `cost_functions.py`. This file contains all cost functions and if intended to 
  adapt the cost functions, simply add new cost functions in this file. 
+ the directory `data` contains the data management. Pre-computed cost functions and 
  k-nearest neighbour matrices are saved in `saved_costs` and `saved_kNN` while 
  `with_demography` consists of all simulated data files. vcf files are to be stored 
  in the subdirectory `vcf`.
+ the directory `plots` contains all generated plots. 

Besides this, the following scripts are central:
- `demo.ipynb`: tangleGen demo for the 1000 Genomes project.
- `read_vcf`: read and pre-process vcf files.
- `simulate_with_demography`: simulates phylogenetic data with an underlying 
  demography. With this script, all simulations of the publication are conducted.
- `compute_kNN.py`: pre-computes k-nearest neighbours for the cost computation.
- `reliability_factor.py`: computes the reliabilty factors of the cuts for the soft 
  clustering.
- `plot_soft_clustering.py`: plots the soft clustering.
+ `conda_env.yml`: conda environment to run tangleGen and Tangles.

Other scripts for reproducing plots from the publication:
+ `tangleGen_on_sim.py`: script to run tangleGen on simulation (Fig. 4 in publication).
+ `tangleGen_on_sim_migration.py`: script to run tangleGen on simulation with 
  significant migration between populations A to H (Fig. 5 in publication).
+ `tangleGen_on_AIMs.py`: script to run tangleGen on 1000 Genomes data based on 
  Kidd's AIMs panel (Fig. 6 in publication).
+ `tangleGen_on_mini_ex.py`: script to run tangleGen on minimal example (Fig. 3 in 
  publication).
+ `simulate_with_demography_migration_A_H.py`: simulation script to run 
  tangleGen_on_sim_migration.py.

Scripts for supplemental plots:
+ `tangleGen_on_AIMs_AMR.py`: script to run tangleGen on 1000 Genomes data based on 
  Kidd's AIMs panel including AMR.
+ `tangleGen_on_sim_many_SNPs.py`: script to run tangleGen on simulated data with 
  many SNPs and reduced time intervals between population splits.
+ `tangleGen_on_1kG_chr22.py`: script to run tangleGen on full chromosome 22 from 1000 
  Genomes Project Data (Phase 3).
+ `plot_soft_clustering_with_AMR.py`: plots the soft clustering with sorting for 
  AIMs including AMR populations.

## How to Run tangleGen
To infer ancestries with tangleGen, a script needs to contain the following modules:

    1. Imports
    2. Set parameters
    3. Load data
    4. Preprocess data
    5. Constructing cuts
    6. Computing costs
    7. Constructing the tangles tree: For each cut compute the tangles by expanding on the
       previous ones if it is consistent. If it is not possible, algorithm determines
    8. Read out cluster-typical genome 
    9. Compute set of characterizing cuts
    10. Postprocess in soft clustering and plot

A detailed execution of these steps for the 1000 Genomes project is shown in the 
demonstration [demo.ipynb](https://github.com/k-burger/tangles_in_pop_gen/blob/main/demo.ipynb).

Here is a quick walkthrough of the essentials for each of these steps:
1. Imports
```python
import pickle
import sys
from functools import partial
import numpy as np
import pandas as pd
import compute_kNN
import plot_soft_clustering
import read_vcf
import reliability_factor
from src import cost_functions, data_types, utils, tree_tangles
sys.path.append('..')
```
2. Set parameters
```python
n = <int>                               # number of individuals
agreement = <int>                       # agreement parameter
cost_fct = {"FST_kNN", "HWE_kNN"}       # choice of cost function
k = <int>                               # k for k-nearest neighbours
pruning = <int>                         # pruning parameter, usually 0
```
3. Load vcf data
```python
your_data = read_vcf.ReadVCF(n, data_set={'AIMs','chr22'}, 'filename', 'filepath')
your_data.read_vcf()
xs = np.transpose(your_data.G[0])           # get diploid genotype matrix
data = data_types.Data(xs=xs)               # embed data in tangles framework
```
4. Preprocess data: Preprocess data: If your data comes with an extra panel file to 
specify the sampling location of each individual, make sure that your data and the panel file have the same sorting. Load the panel file
```python
panel_df = pd.read_csv('your_panel_file', delimiter='\t')
pop_membership = np.array(panel_df['pop'])
```
5. Constructing cuts
```python
bipartitions = data_types.Cuts(values=(data.xs > 0).T,names=np.array(list(range(0, data.xs.shape[1]))))
```
6. Computing costs: as the cost function uses the k-nearest neighbours matrix, kNN 
   has to be computed first and pickled for the cost function to load. 
```python
kNN = compute_kNN.KNearestNeighbours(xs, k, 'filename', 'filepath')
kNN.compute_kNN()   # if once computed, this step can be replaced by loading the kNN-Matrix via kNN.load_kNN()
with open("data/saved_kNN/kNN", 'wb') as outp:  # pickle kNN-Matrix for cost function to load:
    pickle.dump(kNN, outp, pickle.HIGHEST_PROTOCOL)
cost_function = getattr(cost_functions, cost_fct)
# compute costs and order cuts from low to high cost:
bipartitions = utils.compute_cost_and_order_cuts(bipartitions, partial(cost_function,data.xs, None))
bipartitions = utils.merge_doubles(bipartitions)  # merge duplicate cuts
```
7. Constructing the tangles tree: For each cut compute the tangles by expanding on 
    the previous ones if it is consistent. If it is not possible, algorithm 
    determines.
```python
tangles_tree = tree_tangles.tangle_computation(cuts=bipartitions, agreement=agreement, verbose=0)
contracted_tree = tree_tangles.ContractedTangleTree(tangles_tree) # contract tree to necessary information
```
8. Read out cluster-typical genome: 0 means that the typical genome is homozygous for the ancestral allele at the SNP corresponding to the SNP index, otherwise 1.
```python
# identified populations (assign automatically in the future): 
id_pops = <list> # list of identified populations (assign automatically in the future) 
for c in range(len(tangles_tree.maximals)):
    print("cluster-typical genome of ", id_pops[c], ":",tangles_tree.maximals
    [c].tangle.get_specification(), flush=True)
# print corresponding SNP index:
print("\nSNP index in cluster-typical genomes:\n", tangles_tree.maximals[0].tangle.get_cuts().names, flush=True)
```
9. Compute set of characterizing cuts
```python
contracted_tree.calculate_setP()
# prune short paths:
contracted_tree.prune(bipartitions, pruning)
contracted_tree.calculate_setP()
```
10. Postprocess in soft clustering and plot
```python
# compute the soft clustering:
tree_tangles.compute_soft_predictions_children_popgen(node=contracted_tree.root, 
                                                      cuts=bipartitions, weight=np.ones(len(bipartitions.names)), 
                                                      cuts_probs=reliability_factor.compute_reliability(xs),
                                                      verbose=3)
contracted_tree.processed_soft_prediction = True

# Convert the soft clustering to be able to plot in a stacked bar plot:
matrices, char_cuts, positions = contracted_tree.to_matrix()
num_char_cuts_per_split = []
for k in range(1, len(list(char_cuts.keys())) + 1):
    num_char_cuts_per_split.append(np.sum(np.array( [name.count(",") + 1 for name in 
                                                     bipartitions.names[list(char_cuts[k].keys())]])))
num_char_cuts = dict(zip(char_cuts.keys(), num_char_cuts_per_split))

# if wanted, compute hard predictions:
ys_predicted, _ = utils.compute_hard_predictions(contracted_tree, cuts=bipartitions)

# plot the soft clustering:
plot_soft_clustering.plot_inferred_ancestry(matrices, pop_membership, agreement, "readVCF", cost_fct=cost_fct)
```
