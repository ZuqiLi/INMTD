import os
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"

import numpy as np
import numpy.ma as ma
from numpy.linalg import multi_dot, svd
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.special import comb
from sklearn.utils.extmath import randomized_svd as rsvd
from sklearn.preprocessing import Normalizer


def onehot(x):
    X = np.zeros((x.size, x.max()+1))
    X[np.arange(x.size), x] = 1
    return(X)


def init_G(R12, R13, c1, c2, c3, c4, method):
    # R: n x p
    # G: n x k
    if isinstance(R12, pd.DataFrame): R12 = R12.to_numpy()
    if isinstance(R13, pd.DataFrame): R13 = R13.to_numpy()

    if method == 'random': # Random initialization
        G1 = np.random.uniform(low = 0, high = 1, size = (R12.shape[0], c1))
        G2 = np.random.uniform(low = 0, high = 1, size = (R12.shape[1], c2))
        G3 = np.random.uniform(low = 0, high = 1, size = (R13.shape[1], c3))
        G4 = np.random.uniform(low = 0, high = 1, size = (R13.shape[2], c4))
    elif method == 'kmeans': # K-means initialization
        km1 = KMeans(n_clusters=c1, n_init=10, random_state=1).fit(R12)
        km2 = KMeans(n_clusters=c2, n_init=10, random_state=1).fit(R12.T)
        km3 = KMeans(n_clusters=c3, n_init=10, random_state=1).fit(R13.T)
        G1 = onehot(km1.labels_)
        G2 = onehot(km2.labels_)
        G3 = onehot(km3.labels_)
    elif method == 'acol': # ACOL initialization
        R12cp = np.random.permutation(R12.shape[1]) # shuffle over columns of R
        idxs = np.array_split(R12cp, c1) # split by column
        subs = [np.mean(R12[:,idx], axis=1) for idx in idxs] # average columns in each subset
        G1 = np.vstack(subs).T

        R12rp = np.random.permutation(R12.shape[0]) # shuffle over rows of R
        idxs = np.array_split(R12rp, c2) # split by row
        subs = [np.mean(R12[idx,:], axis=0) for idx in idxs] # average rows in each subset
        G2 = np.vstack(subs).T
        
        R13rp = np.random.permutation(R13.shape[0]) # shuffle over rows of R
        idxs = np.array_split(R13rp, c3) # split by row
        subs = [np.mean(R13[idx,:], axis=0) for idx in idxs] # average rows in each subset
        G3 = np.vstack(subs).T
    elif method == 'svd': # SVD initialization
        u, s, vh = svd(R12, full_matrices=False)
        G12 = u[:, :c1]
        G2 = vh.T[:, :c2]
        G12[G12 <= 0] = 1e-5
        G2[G2 <= 0] = 1e-5
        u, s, vh = svd(np.transpose(R13, (2,0,1)), full_matrices=False) # permute -> 3 x N x L
        G13 = np.mean(u[:,:,:c1], axis=0)
        G3 = np.mean(vh[:,:c3,:], axis=0).T
        G13[G13 <= 0] = 1e-5
        G3[G3 <= 0] = 1e-5
        u, s, vh = svd(R13, full_matrices=False)
        G4 = np.mean(vh[:,:c4,:], axis=0).T
        G4[G4 <= 0] = 1e-5
        G1 = (G12 + G13) / 2 # G1 is the average of the intialization from both datasets
    elif method == 'rsvd': # random SVD initialization for G2 and normal SVD for the others
        u, s, vh = rsvd(R12, n_components=c2, n_oversamples=10, random_state=0)
        G12 = u[:, :c1]
        G2 = vh.T[:, :c2]
        G12[G12 <= 0] = 1e-5
        G2[G2 <= 0] = 1e-5
        u, s, vh = svd(np.transpose(R13, (2,0,1)), full_matrices=False) # permute -> 3 x N x L
        G13 = np.mean(u[:,:,:c1], axis=0)
        G3 = np.mean(vh[:,:c3,:], axis=0).T
        G13[G13 <= 0] = 1e-5
        G3[G3 <= 0] = 1e-5
        u, s, vh = svd(R13, full_matrices=False)
        G4 = np.mean(vh[:,:c4,:], axis=0).T
        G4[G4 <= 0] = 1e-5
        G1 = (G12 + G13) / 2 # G1 is the average of the intialization from both datasets
    else:
        print("Unknown initializer: %s"%(method))
        exit(0)
    return(G1.astype(np.float32), G2.astype(np.float32), G3.astype(np.float32), G4.astype(np.float32))


def frob_norm(R12, R13, G1, G2, G3, G4, S12, S13):
    err12 = R12 - multi_dot([G1, S12, G2.T])
    norm12 = np.linalg.norm(err12, ord = 'fro')
    R13_ = np.einsum('ijk,ni->njk', S13, G1)
    R13_ = np.einsum('njk,pj->npk', R13_, G3)
    R13_ = np.einsum('npk,sk->nps', R13_, G4)
    err13 = R13 - R13_
    norm13 = np.linalg.norm(err13, ord = None)
    return(norm12, norm13)
 

def frob_norm_HD(R12, R13, G1, G2, G3, G4, S12, S13, chunk_size):
    # version for high-dimensional R12
    norm12 = 0
    for i in range(0, R12.shape[1], chunk_size):
        end = i + chunk_size
        err12 = R12[:, i:end] - G1.dot(S12).dot(G2[i:end, :].T)
        norm12 += np.sum(np.square(err12))
    norm12 = np.sqrt(norm12)
    R13_ = np.einsum('ijk,ni->njk', S13, G1)
    R13_ = np.einsum('njk,pj->npk', R13_, G3)
    R13_ = np.einsum('npk,sk->nps', R13_, G4)
    err13 = R13 - R13_
    norm13 = np.linalg.norm(err13, ord = None)
    return(norm12, norm13)


def relative_err(J12, J13, normR12, normR13):
    err12 = J12 / normR12
    err13 = J13 / normR13
    return(err12, err13)


def make_nonneg(X):
    if np.any(X < 0):
        print(np.min(X))
        X = np.clip(X, a_min = 0, a_max = None)
    assert np.all(X >= 0), "ERR: negative value exists when updating G"
    return(X)


def sign_decom(X):
    pos = np.zeros(X.shape, dtype=np.float32)
    neg = np.zeros(X.shape, dtype=np.float32)
    pos[X >= 0] = X[X >= 0]
    neg[X < 0] = abs(X[X < 0])
    return(pos, neg)


def update_S12(R12, G1, G2, S12):
    t1 = R12.dot(G2)
    t1 = G1.T.dot(t1)
    #t1_pos, t1_neg = sign_decom(t1)
    t2 = G2.T.dot(G2)
    t2 = G1.T.dot(G1).dot(S12).dot(t2)
    #t2_pos, t2_neg = sign_decom(t2)
    numer = t1
    denom = t2
    denom[denom <= 0] = 1e-5
    factor = np.sqrt(np.divide(numer, denom))
    return(np.multiply(S12, factor))


def update_S13(R13, G1, G3, G4, S13):
    t1 = np.einsum('nqs,ni->iqs', R13, G1)
    t1 = np.einsum('iqs,qj->ijs', t1, G3)
    t1 = np.einsum('ijs,sk->ijk', t1, G4)
    #t1_pos, t1_neg = sign_decom(t1)
    t2 = np.einsum('ijk,im->mjk', S13, G1.T @ G1)
    t2 = np.einsum('ijk,jm->imk', t2, G3.T @ G3)
    t2 = np.einsum('ijk,km->ijm', t2, G4.T @ G4)
    #t2_pos, t2_neg = sign_decom(t2)
    numer = t1
    denom = t2
    denom[denom <= 0] = 1e-5
    factor = np.sqrt(np.divide(numer, denom))
    return(np.multiply(S13, factor))


def update_G1(R12, R13, G1, G2, G3, G4, S12, S13):
    t1 = R12.dot(G2).dot(S12.T)
    #t1_pos, t1_neg = sign_decom(t1)

    R13_1 = R13.reshape((R13.shape[0], -1))
    SG_1 = np.einsum('ijk,qj->iqk', S13, G3)
    SG_1 = np.einsum('iqk,sk->iqs', SG_1, G4)
    SG_1 = SG_1.reshape((S13.shape[0], -1))
    t2 = R13_1 @ SG_1.T
    #t2_pos, t2_neg = sign_decom(t2)

    t3 = G1.dot(G1.T)

    numer = t1 + t2
    denom = t3 @ (t1 + t2)
    denom[denom <= 0] = 1e-5
    factor = np.sqrt(np.divide(numer, denom))
    return(np.multiply(G1, factor))


def update_G2(R12, G2, G1, S12):
    t1 = G1.dot(S12)
    t2 = G1.T.dot(G1)
    t2 = S12.T.dot(t2).dot(S12)
    numer = R12.T @ t1
    denom = G2 @ t2
    denom[denom <= 0] = 1e-5
    factor = np.sqrt(np.divide(numer, denom))
    G2 = np.multiply(G2, factor)
    return(G2)


def update_G2_HD(R12, G2, G1, S12, chunk_size):
    t1 = G1.dot(S12)
    t2 = G1.T.dot(G1)
    t2 = S12.T.dot(t2).dot(S12)
    for i in range(0, R12.shape[1], chunk_size):
        end = i + chunk_size
        numer = R12.T[i:end,:] @ t1
        denom = G2[i:end,:] @ t2
        denom[denom <= 0] = 1e-5
        factor = np.sqrt(np.divide(numer, denom))
        G2[i:end,:] = np.multiply(G2[i:end,:], factor)
    return(G2)


def update_G3(R13, G3, G1, G4, S13):
    R13_2 = np.transpose(R13, (1,2,0)).reshape((R13.shape[1], -1))
    SG_2 = np.einsum('ijk,ni->njk', S13, G1)
    SG_2 = np.einsum('njk,sk->njs', SG_2, G4)
    SG_2 = np.transpose(SG_2, (1,2,0)).reshape((S13.shape[1], -1))
    t1 = R13_2 @ SG_2.T
    #t1_pos, t1_neg = sign_decom(t1)
    
    t2 = SG_2 @ SG_2.T
    #t2_pos, t2_neg = sign_decom(t2)

    numer = t1
    denom = G3 @ t2
    denom[denom <= 0] = 1e-5
    factor = np.sqrt(np.divide(numer, denom))
    return(np.multiply(G3, factor))


def update_G4(R13, G4, G1, G3, S13):
    R13_3 = np.transpose(R13, (2,0,1)).reshape((R13.shape[2], -1))
    SG_3 = np.einsum('ijk,ni->njk', S13, G1)
    SG_3 = np.einsum('njk,qj->nqk', SG_3, G3)
    SG_3 = np.transpose(SG_3, (2,0,1)).reshape((S13.shape[2], -1))
    t1 = R13_3 @ SG_3.T
    #t1_pos, t1_neg = sign_decom(t1)
    
    t2 = SG_3 @ SG_3.T
    #t2_pos, t2_neg = sign_decom(t2)

    numer = t1
    denom = G4 @ t2
    denom[denom <= 0] = 1e-5
    factor = np.sqrt(np.divide(numer, denom))
    return(np.multiply(G4, factor))


def co_occur_mat(labels):
    n_labels = len(labels[0])
    mat = np.zeros((n_labels, n_labels), np.int32)
    for label in labels:
        for i, li in enumerate(label):
            for j, lj in enumerate(label):
                if li == lj: mat[i,j] += 1
    mat = mat / len(labels)
    return(mat)
   

def dispersion(M, c):
    e = 1/c
    # take the lower triangle of the co-occurrence matrix (without diagnal)
    m = M[np.tril_indices(M.shape[0], k = 1)]
    coef = sum(np.square(m - e)) / len(m) / (e - e*e)
    return(coef)


def dispersion_Kim(M):
    return np.sum(4 * np.square(M - 0.5)) / M.shape[0] / M.shape[1]


def rand_index(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def stability(labels):
    RI = []
    n_init = len(labels)
    for i in range(n_init-1):
        for j in range(i+1, n_init):
            RI.append(rand_index(labels[i], labels[j]))
    return(np.mean(RI))


def find_best_c1(R12, R13, c1_list, c2, c3, c4, n_init=10, stop=200):
    for c1 in c1_list:
        labels = []
        metrics = np.zeros((n_init,4)) # [stability, dispersion, Silhouette(R12), Silhouette(R13)]
        for i in range(n_init):
            print(c1, "clusters - Run:", i+1, "......")
            # Initialize embedding matrices
            G1, G2, G3, G4 = init_G(R12, R13, c1, c2, c3, c4, method='random')
            # Initialize core matrices
            S12 = R12.dot(G2)
            S12 = G1.T.dot(S12)
            S13 = np.einsum('nqs,ni->iqs', R13, G1)
            S13 = np.einsum('iqs,qj->ijs', S13, G3)
            S13 = np.einsum('ijs,sk->ijk', S13, G4)

            # Start optimization and stop when meet the criteria
            niter = 0
            while(niter < stop):
                niter += 1
                # Update all latent matrices
                G1 = update_G1(R12, R13, G1, G2, G3, G4, S12, S13)
                G1 = Normalizer(norm='l2').fit_transform(G1.T).T
                G2 = update_G2(R12, G2, G1, S12)
                G3 = update_G3(R13, G3, G1, G4, S13)
                G4 = update_G4(R13, G4, G1, G3, S13)
                S12 = update_S12(R12, G1, G2, S12)
                S13 = update_S13(R13, G1, G3, G4, S13)
            
            # Derive the sample clustering
            C1 = np.argmax(G1, axis=1)
            labels.append(C1)
    
        print('=========================================================')
        score = stability(labels)
        metrics[i, 0] = score
        print("Stability =", score)
        C = co_occur_mat(labels)
        score = dispersion(C, c1)
        metrics[i, 1] = score
        print("Dispersion coefficient =", score)
        sc = SpectralClustering(c1, n_init=10, affinity='precomputed', assign_labels='kmeans').fit(C)
        label = sc.labels_
        #label = labels[0]
        score = silhouette_score(R12, label)
        metrics[i, 2] = score
        print('Silhouette(R12) =', score)
        score = silhouette_score(R13, label)
        metrics[i, 3] = score
        print('Silhouette(R13) =', score)
    
    print('=========================================================')
    best = np.argmax(metrics[:,0])
    print("The highest stability =", metrics[best,0])
    print("The best c1 =", c1_list[best])

    return c1_list[best]


def INMTD(R12, R13, c1, c2, c3, c4, stop=500, eps=1e-6):
    # Compute the norm of R12 and R13
    normR12 = np.linalg.norm(R12, ord = 'fro')
    normR13 = np.linalg.norm(R13, ord=None)

    # Initialize embedding matrices
    G1, G2, G3, G4 = init_G(R12, R13, c1, c2, c3, c4, method='svd')
    print(G1.shape, G2.shape, G3.shape, G4.shape)
    # Initialize core matrices
    S12 = R12.dot(G2)
    S12 = G1.T.dot(S12)
    S13 = np.einsum('nqs,ni->iqs', R13, G1)
    S13 = np.einsum('iqs,qj->ijs', S13, G3)
    S13 = np.einsum('ijs,sk->ijk', S13, G4)

    # Start optimization and stop when meet the criteria
    niter = 0
    Js, J12s, J13s, REs, RE12s, RE13s, crits = [], [], [], [], [], [], [1]
    # Compute objective function
    J12, J13 = frob_norm(R12, R13, G1, G2, G3, G4, S12, S13)
    J12s.append(J12)
    J13s.append(J13)
    J = J12 + J13
    Js.append(J)
    # Compute relative error
    RE12, RE13 = relative_err(J12, J13, normR12, normR13)
    RE12s.append(RE12)
    RE13s.append(RE13)
    RE = RE12 + RE13
    REs.append(RE)
    print("  Iter:", niter, "\tJ =", J, "=", [J12, J13], "\tRel err:", RE)

    while(niter < stop):
        niter += 1
        # Update all latent matrices
        G1 = update_G1(R12, R13, G1, G2, G3, G4, S12, S13)
        G1 = Normalizer(norm='l2').fit_transform(G1.T).T
        G2 = update_G2(R12, G2, G1, S12)
        G3 = update_G3(R13, G3, G1, G4, S13)
        G4 = update_G4(R13, G4, G1, G3, S13)
        S12 = update_S12(R12, G1, G2, S12)
        S13 = update_S13(R13, G1, G3, G4, S13)
            
        # Record the objective function and relative error every 10 iterations
        if niter < 10 or niter % 10 == 0:
            J12, J13 = frob_norm(R12, R13, G1, G2, G3, G4, S12, S13)
            J12s.append(J12)
            J13s.append(J13)
            J = J12 + J13
            Js.append(J)
            RE12, RE13 = relative_err(J12, J13, normR12, normR13)
            RE12s.append(RE12)
            RE13s.append(RE13)
            RE = RE12 + RE13
            REs.append(RE)
            crit = abs(Js[-2] - Js[-1]) / Js[-2]
            crits.append(crit)
            print("  Iter:", niter, "\tJ =", J, "=", [J12, J13], 
                    "\tRel err:", RE, "\tStop crit:", crit)

        # stopping criterion
        #if crit < eps: break

    embedding = {'G1':G1, 'G2':G2, 'G3':G3, 'G4':G4, 'S12':S12, 'S13':S13}
    logging = np.array([Js, J12s, J13s, REs, RE12s, RE13s, crits]).T

    return embedding, logging








