"""
util.py utilities for persistent surveillance tracking

"""
import numpy, scipy, scipy.linalg, random, math

from itertools import izip
argmax = lambda array: max(izip(array, xrange(len(array))))[1] # Daniel Lemire

def index_to_permute(i,  # Index in range [0,n!-1]
                     n   # Number of items
                     ):  # Returns array of n items describing the permutation
    outs = []
    for k in xrange(2,n):
        j = i%k
        i /= k
        outs.append(j)
    outs.append(i)
    v = []
    ins = range(n)
    for k in xrange(n-1,0,-1):
        v = v + [ins.pop(outs.pop())]
    return v + ins

def permute_to_index(v):
    order = []
    i = 0
    while len(v) > 1:
        F = v.pop(0)
        c = 0
        for t in v:
            if t < F:
                c += 1
        i *= len(v)+1
        i +=c
    return i

def column(a):
    """
    A utility that forces argument 'a' to be numpy column vector.
    """
    m = numpy.mat(a)
    r,c = m.shape
    if min(r,c) > 1:
        raise RuntimeError,"(r,c)=(%d,%d)"%(r,c)
    if c == 1:
        return m
    else:
        return m.T

def normalS(mu,sigma,C=None):
    """
    A utility that returns a sample from a vector normal distribtuion.
    """
    if C == None:
        C = numpy.mat(scipy.linalg.cholesky(sigma))
    r,c = C.shape  # rows and columns
    z = []
    for i in xrange(r):
        z.append(random.gauss(0,1))
    z =  numpy.matrix(z)
    return column(mu) + C*z.T
    
def normalP(mu,sigma, x):
    """
    A utility that returns the density of N(mu,sigma) at x.
    """
    from scipy.linalg import solve, det
    r,c = sigma.shape
    d = x-mu
    Q = d.T*solve(sigma,d)
    D = det(sigma)
    norm = 1/((2*math.pi)**(r/2.0)*D)
    P = norm*math.exp(-Q)
    return P

def Hungarian(w,  # dict of weights indexed by tuple (i,j)
              m,  # Cardinality of S (use i \in S)
              n   # Cardinality of T (use j \in T)
              ):
    """ Find mapping X from S to T that maximizes \sum_{i,j}
    X_{i,j}w_{i,j}.  Return X as a dict with X[i] = j
    """
    w_list = w.values()
    Min = min(w_list)
    for key in w.keys():   # Ensure all w[ij] >= 0
        w[key] = w[key] - Min
    S_labels = {}
    T_labels = {}
    S_All = {}
    T_All = {}
    for i in xrange(m):
        S_All[i] = None
    for j in xrange(n):
        T_All[j] = None
    # Begin Lawler's step 0
    X_S2T = {}
    X_T2S = {}
    u = scipy.ones(m)*max(w.values())
    v = scipy.zeros(n)
    pi = scipy.ones(n)*1e20
    S_label = {}
    T_label = {}
    # End Lawler's step 0
    def augment(X_S2T,X_T2S,i,j):
        """ If arc (i,j) is in X, take it out.  If it is not in X, put
        it in.
        """
        if X_S2T.has_key(i) and X_T2S.has_key(j):
            del X_S2T[i]
            del X_T2S[j]
            return
        if X_S2T.has_key(i) or X_T2S.has_key(j):
            raise RuntimeError,'Arc exists in one dict but not the other'
        X_S2T[i] = j
        X_T2S[j] = i
        return
    def backtrack_i(X_S2T,X_T2S,S_label,T_label,i):
        if not S_label.has_key(i):
            return
        j = S_label[i]
        augment(X_S2T,X_T2S,i,j)
        backtrack_j(X_S2T,X_T2S,S_label,T_label,j)
    def backtrack_j(X_S2T,X_T2S,S_label,T_label,j):
        if not T_label.has_key(j):
            return
        i = T_label[j]
        augment(X_S2T,X_T2S,i,j)
        backtrack_i(X_S2T,X_T2S,S_label,T_label,i)
    unscanned_S = S_All.copy()
    unscanned_T = T_All.copy()
    k = 0
    while True: # This is Lawler's step 1 (labeling).  I make it the main loop
        k += 1
        assert(k<n**3*m**3)
        for i in unscanned_S.keys():
            if not X_S2T.has_key(i):  # Step 1.0
                S_label[i] = None
                continue
            # Begin step step 1.3 on i
            for j in xrange(n):
                if not w.has_key((i,j)) or X_S2T[i] == j :
                    continue
                if u[i] + v[j] - w[(i,j)] > pi[j]:
                    continue
                T_label[j] = i
                pi[j] = u[i] + v[j] - w[(i,j)]
            del unscanned_S[i]
            # End step step 1.3 on i
        for j in unscanned_T.keys():
            if pi[j] > 0:
                continue
            # Begin step step 1.4 on j
            if not X_T2S.has_key(j):
                # Begin Lawler's step 2 (augmentation)
                backtrack_j(X_S2T,X_T2S,S_label,T_label,j)
                pi = scipy.ones(n)*1e20
                S_label = {}
                T_label = {}
                # End Lawler's step 2
            else:
                S_label[X_T2S[j]] = j
                del unscanned_T[j]
        # Begin Lawler's step 1.1 (check for step 3)
        if len(unscanned_S) > 0:
            continue # Start another iteration of the labeling loop
        skip3 = False
        for j in unscanned_T.keys():
            if pi[j] > 0:
                continue # Continue checking j's
            else:
                skip3 = True
                break
        if skip3:
            continue # Start another iteration of the labeling loop
        # End step 1.1
        # Begin Lawler's step 3 (change dual varibles)
        delta_1 = min(u)
        delta_2 = min(pi) # FixMe: is delta_2>0?
        assert(delta_2 > 0)
        delta = min(delta_1,delta_2)
        for i in S_label.keys():
            u[i] -= delta
        for j in T_label.keys():
            if pi[j] > 0:
                v[j] += delta
        if delta < delta_1:
            continue # Start another iteration of the labeling loop
        # Finished
        return X_S2T

if __name__ == '__main__':  # Test code
    M = scipy.array([
        [ 1, 2, 3, 0],
        [ 2, 4, 0, 8],
        [-1, 0,-3,-4]])
    w = {}
    m = 3
    n = 4
    for i in xrange(m):
        for j in xrange(n):
            w_ij = M[i,j]
            if w_ij*w_ij < .1:
                continue
            w[(i,j)] = w_ij * 1.1
    X = Hungarian(w,m,n)
    print X


#---------------
# Local Variables:
# eval: (python-mode)
# End:
