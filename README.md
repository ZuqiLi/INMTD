# INMTD
Integrative Non-negative Matrix and Tensor Decomposition

![model2](https://github.com/user-attachments/assets/0cdeb77f-8d3a-454c-9018-264becdd160e)

INMTD (**I**ntegrative **N**on-negative **M**atrix and **T**ensor **D**ecomposition) is a novel multi-view clustering method which integrates 2D and 3D datasets for joint clustering and removes confounding effects. It learns an embedding matrix for each data dimension and subgroups the individuals from their embedding after removing vectors in the embedding space that are linked with confounders. More specifically, INMTD combines nonnegative matrix tri-factorization (NMTF) [1] and nonnegative Tucker decomposition (NTD) [2] to cluster subjects with multi-view data regardless of their dimensionality. We assume $p_1$ subjects described by two data views, a 2D matrix $`X_{12} \in \mathbb{R}_{+}^{p_1 \times p_2}`$ of $p_2$ features and a 3D tensor $`\mathcal{X}_{134} \in \mathbb{R}_{+}^{p_1 \times p_3 \times p_4}`$ of $p_3$ features with $p_4$ channels, both nonnegative. The aim of our method is to jointly compute the embedding matrices for each dimension and cluster the $p_1$ subjects based on its specific embedding.
The objective function of INMTD is as follows:
```math
\min_{G_i \geq 0, S_{12} \geq 0, S_{134} \geq 0}⁡ J = \| X_{12} - G_1 S_{12} G_2^T \|_F^2 + \| \mathcal{X}_{134} - \mathcal{S}_{134} ×_1 G_1 ×_2 G_3 ×_3 G_4 \|_F^2, \quad \mathrm{s.t.} \quad G_1^T G_1=I
```

## Environment
For better reproducibility, it's recommended to refer to the following hardware and software settings:
- Operating system: Ubuntu 20.04.6 LTS
- Processor: Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz
- Memory: 767 GiB
- Graphics: llvmpipe (LLVM 12.0.0, 256 bits)
- Python version: 3.9.7

The required packages can be installed with the conda environment file in this repository:
```
conda env create -f environment.yml
```

## Tutorial
To use INMTD, please download the `INMTD.py` file in the `Code` folder from this Github repository to your local. 
In `INMTD.py`, there are 3 important functions for users to run INMTD with customization:
&nbsp;
```
init_G(R12, R13, r1, r2, r3, r4, method)
```
### Description:
Initialize embedding matrices $G_1$, $G_2$, $G_3$, and $G_4$.
### Parameters:
- `R12`: The first dataset, a 2D numpy array of size $p1 \times p2$.
- `R13`: The second dataset, a 3D numpy array of size $p1 \times p3 \times p4$.
- `r1`: The rank of $G_1$, a positive integer value.
- `r2`: The rank of $G_2$, a positive integer value.
- `r3`: The rank of $G_3$, a positive integer value.
- `r4`: The rank of $G_4$, a positive integer value.
- `method`: The method used for initialization, a string. Possible values are 'random', 'svd', and 'rsvd'.
  - 'random': Initialize all embedding matrices with random sampling from a uniform distribution of range [0,1].
  - 'svd': Use singluar value decomposition to initialize all embedding matrices. Note that `R13` is decomposed along the last dimension and it singular vectors are averaged over the iterations because it's a 3D array. All singular vectors are ordered descendingly by their corresponding singular values. $G_1$ is then initialized by the average of `r1` left singular vectors of `R12` and `R13`. $G_2$ is initialized by the `r2` right singular vectors of `R12`. $G_3$ is initialized by the `r3` right singular vectors of `R13`. For $G_4$, `R13` is re-decomposed along the first dimension and the average of the `r4` right singular vectors becomes $G_4$. All initialized values <= 0 are replaced by 1e-5.
  - 'rsvd': Use random SVD to decompose `R12` in case of high dimensionality. The rest is the same as 'svd' initialization.
### Returns:
- `G1`: A nonnegative numpy array of size $p1 \times r1$ with `float32` data type.
- `G2`: A nonnegative numpy array of size $p2 \times r2$ with `float32` data type.
- `G3`: A nonnegative numpy array of size $p3 \times r3$ with `float32` data type.
- `G4`: A nonnegative numpy array of size $p4 \times r4$ with `float32` data type.

&nbsp;
```
find_best_r1(R12, R13, r1_list, r2, r3, r4, n_init=10, stop=200)
```
### Description:
Find the best value for `r1`, the rank of $G_1$. This function runs INMTD with random initialization for multiple (`n_init`) times with different `r1` values. In each repitition, a clustering of samples is derived by assigning each sample to the cluster with highest value in the corresponding column of $G_1$. Note that the number of columns of $G_1$, namely `r1`, is the same as number of clusters. Subsequently, a consensus clustering is calculated from the ensemble of clusterings with the same `r1`, yielding a stability score. The `r1` value with highest stability score is chosen as the best.
### Parameters:
- `R12`: The first dataset, a 2D numpy array of size $p1 \times p2$.
- `R13`: The second dataset, a 3D numpy array of size $p1 \times p3 \times p4$.
- `r1_list`: A list of positive integer values for the rank of $G_1$ to be tested.
- `r2`: A positive integer value for the rank of $G_2$.
- `r3`: A positive integer value for the rank of $G_3$.
- `r4`: A positive integer value for the rank of $G_4$.
- `n_init`: A positive integer value for the number of repititions of random initialization. The default is 10.
- `stop`: A positive integer value for the maximal number of iterations that INMTD runs in each repitition. The default is 200.
### Returns:
- A positive integer value for the best `r1`.

&nbsp;
```
INMTD(R12, R13, r1, r2, r3, r4, init='svd', stop=500)
```
### Description:
Run the INMTD model to joint decompose 2D and 3D datasets.
### Parameters:
- `R12`: The first dataset, a 2D numpy array of size $p1 \times p2$.
- `R13`: The second dataset, a 3D numpy array of size $p1 \times p3 \times p4$.
- `r1`: A positive integer value for the rank of $G_1$.
- `r2`: A positive integer value for the rank of $G_2$.
- `r3`: A positive integer value for the rank of $G_3$.
- `r4`: A positive integer value for the rank of $G_4$.
- `init`: A string for the initialization method. Possible values are 'random', 'svd', and 'rsvd'. The default is 'svd'.
- `stop`: A positive integer value for the maximal number of iterations that INMTD runs. The default is 500.
### Returns:
- `embedding`: A list containing embedding matrices $G_1$, $G_2$, $G_3$, $G_4$, the core matrix $S_{12}$ for `R12`, and the core tensor $\mathcal{S}_{13}$ for `R13`.
- `logging`: A 2D numpy array with 6 columns corresponding to the joint reconstruction error of `R12` and `R13`, the reconstruction error of `R12`, the reconstruction error of `R13`, the joint relative error of `R12` and `R13`, the relative error of `R12`, and the relative error of `R13`. Rows are the recording of the 6 metrics in the first 10 iterations and every 10 iterations afterwards.
  
## Example
Here is an example of how to run INMTD with simulated data. Functions and example datasets of the simulation can be found in the `Simulation` folder.
### Simulation
```
from simulation import generate_data

p1, p2, p3, p4 = 1000, 250, 80, 20
r1, r2, r3, r4 = 5, 10, 4, 2

R12, R13, clust1, clust2, clust3, clust4 = generate_data([p1, p2, p3, p4], [r1, r2, r3, r4])
```
The `generate_data` function in `simulation.py` takes 2 parameters as input. The first parameter is a list containing the numbers of dimensions of the 2 datasets to be simulated, namely `R12` and `R13`. The second parameter is another list containing the ranks of the embedding matrices composing `R12` and `R13`. It returns the 2D matrix `R12`, the 3D tensor `R13`, the true clustering on the dimension of `p1`, the true clustering on the dimension of `p2`, the true clustering on the dimension of `p3`, and the true clustering on the dimension of `p4`.

### INMTD pipeline
```
from INMTD import INMTD
import numpy as np

embedding, logging = INMTD(R12, R13, r1, r2, r3, r4)
print(logging)
```
Run INMTD model on the simulated data to derive the learnt embeddings.

&nbsp;
```
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

G1 = embedding['G1']
G1_scaled = StandardScaler().fit_transform(G1)
G1_clust = KMeans(n_clusters=r1, n_init=10, max_iter=300, random_state=123).fit(G1_scaled).labels_
G1_score = adjusted_rand_score(clust1, G1_clust)
print("ARI for KMeans on G1:", G1_score)

G2 = embedding['G2']
G2_scaled = StandardScaler().fit_transform(G2)
G2_clust = KMeans(n_clusters=r2, n_init=10, max_iter=300, random_state=123).fit(G2_scaled).labels_
G2_score = adjusted_rand_score(clust2, G2_clust)
print("ARI for KMeans on G2:", G2_score)

G3 = embedding['G3']
G3_scaled = StandardScaler().fit_transform(G3)
G3_clust = KMeans(n_clusters=r3, n_init=10, max_iter=300, random_state=123).fit(G3_scaled).labels_
G3_score = adjusted_rand_score(clust3, G3_clust)
print("ARI for KMeans on G3:", G3_score)

G4 = embedding['G4']
G4_scaled = StandardScaler().fit_transform(G4)
G4_clust = KMeans(n_clusters=r4, n_init=10, max_iter=300, random_state=123).fit(G4_scaled).labels_
G4_score = adjusted_rand_score(clust4, G4_clust)
print("ARI for KMeans on G4:", G4_score)
```
Cluster each dimension based on the corresponding embedding and compute the adjusted Rand index to assess how good the clustering is with respect to the true clustering.

&nbsp;
```
from sklearn.metrics.pairwise import cosine_similarity

S12 = embedding['S12']
S13 = embedding['S13']

# Map G2 and G3 to the space of G1
S12G2 = S12.dot(G2.T)
S13G3 = np.einsum('ijk,pj->ipk', S13, G3)
S13G3 = np.mean(S13G3, axis=2) # convert from 3D to 2D by averaging along the last axis
print("S12*G2 shape:", S12G2.shape)
print("S13*G3 shape:", S13G3.shape)

# Concatenation
merged = np.concatenate((StandardScaler().fit_transform(S12G2.T), StandardScaler().fit_transform(S13G3.T)), axis=0)
print("Concatenation shape:", merged.shape)

# Compute centroid of G1 clusters
G1cent = np.array([np.mean(G1[np.where(G1_clust == i)[0], :], axis=0) for i in range(r1)])
merged = np.concatenate((merged, StandardScaler().fit_transform(G1cent)), axis=0)
print("Concatenation shape:", merged.shape)

# Cosine similarity between each G2 feature and each G1 cluster centroid
G2simG1cent = cosine_similarity(merged[:p2,:], merged[-r1:,:])
print("G2simG1cent shape:", G2simG1cent.shape)
#G2simG1cent = np.max(G2simG1cent, axis=1) # the highest similarity of each G2 feature across all centroids
#G2_select = np.argsort(-G2simG1cent, axis=None)[:10] # select the top 10 G2 features

# Cosine similarity between each G3 feature and each G1 cluster centroid
G3simG1cent = cosine_similarity(merged[p2:-r1,:], merged[-r1:,:])
print("G3simG1cent shape:", G3simG1cent.shape)
#G3simG1cent = np.max(G3simG1cent, axis=1) # the highest similarity of each G3 feature across all centroids
#G3_select = np.argsort(-G3simG1cent, axis=None)[:10] # select the top 10 G3 features
```
Link $G_2$ and $G_3$ by projecting them to the common space of $G_1$. Every cluster from $G_1$ can be characterized by selecting features in $G_2$ and features in $G_3$ with highest cosine similarity to the cluster centroids in the joint space.

&nbsp;
```
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

# Dimensionality reduction
#reducer = PCA(n_components=2)
reducer = KernelPCA(n_components=2, kernel='cosine')
embedding = reducer.fit_transform(merged)
print("PC shape:", embedding.shape)
#print("Explained variance ratio:", reducer.explained_variance_ratio_)

# Plot G2 features, G3 features and G1 cluster centroids in the common space
fig, ax = plt.subplots(figsize=(11, 9))
plt.scatter(embedding[:, 0], embedding[:, 1],
    #c=np.repeat(['orange','blue'], [n_lms, n_lms]))
    c=np.repeat(['orange','blue','red'], [p2, p3, r1]))
plt.gca().set_aspect('equal', 'datalim')
plt.xlabel('The 1st principal component', fontsize=18)
plt.ylabel('The 2nd principal component', fontsize=18)
plt.show()
```
Plot all the features in $G_2$, features in $G_3$ and the cluster centroids of $G_1$ in a 2D space reduced by PCA with cosine kernel.

## Acknowledgement
## References
> [1] Ding, C., Li, T., Peng, W. & Park, H. Orthogonal nonnegative matrix t-factorizations for clustering. in Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining 126–135 (ACM, Philadelphia PA USA, 2006).\
> [2] Kim, Y.-D. & Choi, S. Nonnegative Tucker Decomposition. in 2007 IEEE Conference on Computer Vision and Pattern Recognition 1–8 (IEEE, Minneapolis, MN, USA, 2007).\
