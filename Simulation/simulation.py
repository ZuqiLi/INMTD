import numpy as np


def onehot(x):
    X = np.zeros((x.size, x.max()+1))
    X[np.arange(x.size), x] = 1
    return(X)

def generate_data(Ps, Cs):
    np.random.seed(1)
    p1, p2, p3, p4 = Ps
    c1, c2, c3, c4 = Cs

    clust_1 = np.random.choice(c1, size=p1, replace=True)
    clust_2 = np.random.choice(c2, size=p2, replace=True)
    clust_3 = np.random.choice(c3, size=p3, replace=True)
    clust_4 = np.random.choice(c4, size=p4, replace=True)

    G1 = onehot(clust_1)# + np.abs(np.random.normal(0, 0.1, size=(p1, c1)))
    G2 = onehot(clust_2)# + np.abs(np.random.normal(0, 0.1, size=(p2, c2)))
    G3 = onehot(clust_3)# + np.abs(np.random.normal(0, 0.1, size=(p3, c3)))
    G4 = onehot(clust_4)# + np.abs(np.random.normal(0, 0.1, size=(p4, c4)))

    S12 = np.random.uniform(0, 1, size=(c1, c2))
    S134 = np.random.uniform(0, 1, size=(c1, c3, c4))

    R12 = G1 @ S12 @ G2.T
    R134 = np.einsum('ijk,ni->njk', S134, G1)
    R134 = np.einsum('njk,qj->nqk', R134, G3)
    R134 = np.einsum('nqk,sk->nqs', R134, G4)
    
    R12 += np.random.uniform(0, 1, size=(p1, p2))
    R134 += np.random.uniform(0, 1, size=(p1, p3, p4))
    #R12 += np.random.normal(0, 0.1, size=(p1, p2))
    #R134 += np.random.normal(0, 0.1, size=(p1, p3, p4))
    #R12 = np.abs(R12)
    #R134 = np.abs(R134)

    return(R12, R134, clust_1, clust_2, clust_3, clust_4)











