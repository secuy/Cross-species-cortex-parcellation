# Cross-Species Cortical Parcellation via Homology Consensus Graph Representation Learning from Diffusion MRI Tractography

This repository contains the source code of the paper titled:  
**Cross-Species Cortical Parcellation via Homology Consensus Graph Representation Learning from Diffusion MRI Tractography**, submitted to a top conference in computational neuroscience.

---

## Introduction

Understanding conserved cortical regions across species is a central goal in comparative neuroscience. However, species differences in brain geometry and fiber projections pose significant challenges.  
This project presents a **two-stage clustering pipeline** for performing joint cortical parcellation of human and macaque brains using multimodal structural connectivity features and low-rank tensor learning.

We first compute **connectivity-based vertex features** from diffusion MRI tractography and a white matter tract atlas. These features are used in an iterative **super-vertex clustering** algorithm that integrates both geometric continuity and connectivity similarity.  
Next, the grouped super-vertices are refined using a **low-rank tensor learning framework** that jointly encodes homologous connectivity patterns across species, enabling the extraction of **biologically consistent cortical parcels**.

---

## Key Contributions

- **Joint Feature Integration**: Combines tractography-based cluster connections and atlas participation into a unified vertex-wise feature representation.
- **Cross-Species Super-Vertex Clustering**: Produces spatially coherent preparcellations via hybrid distance metrics across aligned cortical surfaces.
- **Low-Rank Tensor Learning**: Learns a consensus graph across humans and macaques using view-specific similarity matrices and spectral embeddings.
- **Homologous Parcellation Identification**: Achieves biologically meaningful cortical mappings by maximizing cross-species overlap and consistency.

---

## Project Structure

```bash
├── cluster_metrics.py              # Clustering evaluation metrics
├── gen_cluster-vertice_mat*.py     # Feature extraction: vertex-cluster matrix
├── gen_connect_atlas_mat*.py       # Feature extraction: cluster-atlas (XTRACT) matrix
├── low_rank_tensor_learning.py     # Main optimization algorithm for homologous parcel discovery
├── parcel_alignment.py             # Final step for parcellation alignment across species
└── utils/
    ├── read_tck.py                 # Fiber reading utility
    ├── surface_utils.py            # Surface registration and resampling tools
    └── mat_utils.py                # General matrix and similarity utilities
```

---

## Dependencies

```
Python == 3.9
scikit-learn == 1.5.2
scipy == 1.13.1
nibabel == 5.3.2
numpy >= 1.24
```

---

## Data Preparation

To reproduce the experiments, the following data are required:

1. Standard cortical surface templates (e.g., HCP fs_LR10k for humans, Yerkes19 for macaques).
2. XTRACT white matter tract atlases for both species.  
   → [XTRACT Atlases GitHub](https://github.com/SPMIC-UoN/XTRACT_atlases)
3. Whole-brain tractography cluster results:
   - Human: ~33,000 clusters
   - Macaque: ~12,000 clusters  
   These clusters should be obtained using multi-method tractogram merging (iFOD1, iFOD2, DET) and hierarchical clustering.

Place cluster files and preprocessed tractograms in your working directory with appropriate naming.

---

## Usage

### Step 1: Feature Matrix Generation

Generate the vertex-cluster and cluster-atlas (tract) matrices:

```bash
python gen_cluster-vertice_mat_human_thread.py
python gen_connect_atlas_mat_human_thread.py
```

Repeat for macaque by running the corresponding `*_macaque_*.py` scripts.

### Step 2: Super-Vertex Preparcellation

(Optional) Apply spatial + feature-based super-vertex clustering (see inside `low_rank_tensor_learning.py` for internal integration).  
Ensure that species surfaces are registered to a shared coordinate system.

### Step 3: Low-Rank Tensor Learning and Clustering

Run the tensor optimization and parcellation code:

```bash
python low_rank_tensor_learning.py
```

This script will compute:
- Multi-view similarity matrices (from VC and VT features)
- Consensus graphs via spectral embedding
- Final homologous cortical parcels using spectral clustering

### Step 4: Alignment and Evaluation

Align species-specific parcellation results to ensure homologous mapping:

```bash
python parcel_alignment.py
```

This step identifies pairs of cortical parcels across species with maximal overlap and homology score.

---

## Output

The final output includes:
- `.npy` files containing parcel label assignments for each vertex
- Similarity graphs and low-rank embeddings
- Evaluation metrics for cross-species consistency
- Optional surface visualization using FreeSurfer or Connectome Workbench

