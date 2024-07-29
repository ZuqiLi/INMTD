# INMTD
Integrative Non-negative Matrix and Tensor Decomposition

![model2](https://github.com/user-attachments/assets/0cdeb77f-8d3a-454c-9018-264becdd160e)

INMTD (**I**ntegrative **N**on-negative **M**atrix and **T**ensor **D**ecomposition) is a novel multi-view clustering method which integrates 2D and 3D datasets for joint clustering and removes confounding effects. It learns an embedding matrix for each data dimension and subgroups the individuals from their embedding after removing vectors in the embedding space that are linked with confounders. More specifically, INMTD combines nonnegative matrix tri-factorization (NMTF) [^1] and nonnegative Tucker decomposition (NTD) [^2] to cluster subjects with multi-view data regardless of their dimensionality. We assume $p_1$ subjects described by two data views, a 2D matrix $`X_{12} \in \mathbb{R}_{+}^{p_1 \times p_2}`$ of $p_2$ features and a 3D tensor $`\mathcal{X}_{134} \in \mathbb{R}_{+}^{p_1 \times p_3 \times p_4}`$ of $p_3$ features with $p_4$ channels, both nonnegative. The aim of our method is to jointly compute the embedding matrices for each dimension and cluster the $p_1$ subjects based on its specific embedding.
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
To use INMTD, please download the code files from this Github repository to your local. 
In `INMTD.py` there are 3 important functions for users to properly run INMTD:
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
  
## Usage
There're 2 ways to use netMUG: call the all-in-one-go function or break it down to steps.
#### Strategy 1: all-in-one-go function
```
# The first data view of shape [n x p1]
X <- matrix(runif(5000), nrow=100)
# The second data view of shape [n x p2]
Y <- matrix(runif(4000), nrow=100)
# The extraneous variable of shape [n]
Z <- runif(100)
# l1, l2 are the sparsity parameters (more explanation can be found in SmCCNet)
l1 <- 0.2
l2 <- 0.2
# s1, s2 are the subsampling parameters (more explanation can be found in SmCCNet)
s1 <- 0.8
s2 <- 0.9

# netMUG returns a list: the selected features from X, the selected features from Y, ISNs, and the final clustering
res <- netMUG(X, Y, Z, l1, l2, s1, s2)
```
#### Strategy 2: step-by-step pipeline
```
# Step 1: select multi-view features informed by an extraneous variable
smccnet <- selectFeatures(X, Y, Z, l1, l2, s1, s2)
Xsub <- X[, smccnet$featureX]
Ysub <- Y[, smccnet$featureY]

# Step 2: build ISNs from the selected features
V <- cbind(Xsub, Ysub)
ISNs <- buildInfISNs(V, Z, nCores = 1)

# Step 3: compute distances between ISNs
dis <- computeDist(ISNs)

# Step 4: Ward's hierarchical clustering with Dynamic Tree Cut
dendro <- hclust(as.dist(dis), method = "ward.D2")
clust <- cutreeDynamic(dendro, minClusterSize = 1, distM = dis, 
                      deepSplit = 0)
clust <- as.factor(clust)
```
### Acknowledgement
### References
> [^1] Ding, C., Li, T., Peng, W. & Park, H. Orthogonal nonnegative matrix t-factorizations for clustering. in Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining 126–135 (ACM, Philadelphia PA USA, 2006).\
> [^2] Kim, Y.-D. & Choi, S. Nonnegative Tucker Decomposition. in 2007 IEEE Conference on Computer Vision and Pattern Recognition 1–8 (IEEE, Minneapolis, MN, USA, 2007).\
