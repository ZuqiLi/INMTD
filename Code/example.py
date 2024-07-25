from simulation import generate_data
from INMTD import INMTD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics.pairwise import cosine_similarity


#############################
######  Simulate data  ######
#############################
p1, p2, p3, p4 = 1000, 250, 80, 20
r1, r2, r3, r4 = 5, 10, 4, 2
R12, R13, clust1, clust2, clust3, clust4 = generate_data([p1, p2, p3, p4], [r1, r2, r3, r4])


#############################
#####  Run INMTD model  #####
#############################
embedding, logging = INMTD(R12, R13, r1, r2, r3, r4)
print(logging)


############################################
#####  Clustering from the embeddings  #####
############################################
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


#############################
#####  Link G2 with G3  #####
#############################
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



