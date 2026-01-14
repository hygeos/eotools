from scipy.ndimage import convolve
import numpy as np


def stdev(S, S2, N, fillv=np.nan):
    '''
    Returns standard deviation from:
        * S sum of the values
        * S2 sum of the squared values
        * N number of values
    The values where N=0 are filled with fillv
    '''
    R = np.zeros_like(S) + fillv
    ok = N != 0
    R[ok] = S2[ok]/N[ok] - (S[ok]/N[ok])**2
    R[R<0] = 0.   # because a few values may be slightly negative
    R[ok] = np.sqrt(R[ok])
    return R    

def stdNxN(X, N, mask=None, fillv=np.nan):
    '''
    Standard deviation over NxN blocks over array X
    '''
    output = np.zeros_like(X, dtype=float)
    if mask is None: M = 1.
    else: M = mask

    ker = np.ones((N,N))
    S = convolve(X*M, ker, mode='constant', cval=0)
    S2 = convolve(X*X*M, ker, mode='constant', cval=0)
    C = convolve(np.ones(X.shape)*M, ker, mode='constant', cval=0)
    output = stdev(S, S2, C, fillv=fillv)

    return output