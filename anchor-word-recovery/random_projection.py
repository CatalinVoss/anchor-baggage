import numpy as np
import math

# Project the columns of the matrix M into the
# lower dimension new_dim
def Random_Projection(M, new_dim, prng):
    old_dim = M[:, 0].size
    p = np.array([1./6, 2./3, 1./6])
    c = np.cumsum(p)
    randdoubles = prng.random_sample(new_dim*old_dim)
    R = np.searchsorted(c, randdoubles)
    R = math.sqrt(3)*(R - 1)
    R = np.reshape(R, (new_dim, old_dim))
    
    M_red = np.dot(R, M)
    return M_red