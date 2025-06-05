# Cross-Species Cortical Parcellation via Homology Consensus Graph Representation Learning from Diffusion MRI Tractography

### Introduction

This project is the source code of a paper called Cross-Species Cortical Parcellation via Homology Consensus Graph Representation Learning from Diffusion MRI Tractography.

### Dependence
```
Python == 3.9
sklearn == 1.5.2
scipy == 1.13.1
nibabel == 5.3.2
```

### File Profile

__utils/*__ : It contains various cortical processing tool scripts, as well as fiber reading code.

__cluster_metrics.py__ : There are multiple parameter codes in the file for calculating clustering effects.

__gen_cluster-vertice_mat*__ : These scripts contain code for calculating the feature matrix between vertices and clusters.

__gen_connect_atlas_mat*__ : These scripts contain code for calculating the feature matrix between atlas and clusters.

__low_rank_tensor_learning.py__ ：Main code for matrix clustering optimization.

__parcel_aligment.py__ : Code for cross-species cortex surface clustering results alignment.
