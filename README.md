# INMTD
Integrative Non-negative Matrix and Tensor Decomposition

INMTD (**I**ntegrative **N**on-negative **M**atrix and **T**ensor **D**ecomposition) is a novel multi-view clustering method which integrates 2D and 3D datasets for joint clustering and removes confounding effects. It learns an embedding matrix for each data dimension and subgroups the individuals from their embedding after removing vectors in the embedding space that are linked with confounders. More specifically, INMTD combines nonnegative matrix tri-factorization (NMTF) [1] and nonnegative Tucker decomposition (NTD) [2] to cluster subjects with multi-view data regardless of their dimensionality. We assume $p_1$ subjects described by two data views, a 2D matrix $`X_{12} \in \mathbb{R}_{+}^{p_1 \times p_2}`$ of $p_2$ features and a 3D tensor $`\mathcal{X}_{134} \in \mathbb{R}_{+}^{p_1 \times p_3 \times p_4}`$ of $p_3$ features with $p_4$ channels, both nonnegative. The aim of our method is to jointly compute the embedding matrices for each dimension and cluster the $p_1$ subjects based on its specific embedding.
The objective function of INMTD is as follows:
```math
\min_{G_i \geq 0, S_{12} \geq 0, S_{134} \geq 0}⁡ J = \| X_{12} - G_1 S_{12} G_2^T \|_F^2 + \| \mathcal{X}_{134} - \mathcal{S}_{134} ×_1 G_1 ×_2 G_3 ×_3 G_4 \|_F^2, \quad \mathrm{s.t.} \quad G_1^T G_1=I
```

### Environment
For better reproducibility, it's recommended to refer to the following hardware and software settings:
```
Operating system: Ubuntu 20.04.6 LTS
Processor: Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz
Memory: 767 GiB
Graphics: llvmpipe (LLVM 12.0.0, 256 bits)
Python version: 3.9.7
RStudio version: 2022.07.1+554
```

### Installation
Before using netMUG in your R environment, please download the code files from this Github repository in R:
```
# Download netMUG functions
if (!require("devtools")) install.packages("devtools")
library(devtools)
source_url("https://raw.githubusercontent.com/ZuqiLi/netMUG/main/R/netMUG.R")
```
The current netMUG is developed under R version 4.2.1 with the following packages:
- parallel (4.2.1)
- devtools (2.4.3)
- dynamicTreeCut (1.63.1)
- SmCCNet (0.99.0)

### Usage
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
> [1] Ding, C., Li, T., Peng, W. & Park, H. Orthogonal nonnegative matrix t-factorizations for clustering. in Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining 126–135 (ACM, Philadelphia PA USA, 2006).\
> [2] Kim, Y.-D. & Choi, S. Nonnegative Tucker Decomposition. in 2007 IEEE Conference on Computer Vision and Pattern Recognition 1–8 (IEEE, Minneapolis, MN, USA, 2007).\
