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
    debug = True
    w_list = w.values()
    Max = max(w_list)
    Min = min(w_list)
    tol = (Max - Min)*1e-6  # Tolerance for some comparisons
    if Min < tol:           # Ensure all w[ij] >= 0
        for key in w.keys():
            w[key] = w[key] - Min + 2*tol
        Max = Max - Min + 2*tol
        Min = 2*tol
    inf = Max*10            # "Infinity" for pi values
    unscanned_S = {}
    unscanned_T = {}
    def augment(X,X_S,X_T,i,j):
        """ If arc (i,j) is in X, take it out.  If it is not in X, put
        it in.
        """
        if X.has_key((i,j)):
            del X[(i,j)]
            if X_S[i] == j:
                del X_S[i]
            if X_T[j] == i:
                del X_T[j]
        else:
            X[(i,j)] = True
            X_S[i] = j
            X_T[j] = i
        return
    def backtrack_i(X,X_S,X_T,S_label,T_label,i):
        if not S_label.has_key(i) or S_label[i] is None:
            return
        j = S_label[i]
        if debug:
            print '(%d,%d)'%(i,j),
        augment(X,X_S,X_T,i,j)
        backtrack_j(X,X_S,X_T,S_label,T_label,j)
    def backtrack_j(X,X_S,X_T,S_label,T_label,j):
        if not T_label.has_key(j):
            return
        i = T_label[j]
        if debug:
            print '(%d,%d)'%(i,j),
        augment(X,X_S,X_T,i,j)
        backtrack_i(X,X_S,X_T,S_label,T_label,i)
    # Begin Lawler's step 0
    X = {}   # Dict of links X[(i,j)] = True
    X_S = {} # Dict of nodes. X_S[i]=j if i linked to j.
    X_T = {} # Dict of nodes. X_T[j]=i if j linked to i.
    u = scipy.ones(m)*Max
    v = scipy.zeros(n)
    pi = scipy.ones(n)*inf
    S_label = {}
    T_label = {}
    # Begin Lawler's step 1.0: Exposed S nodes get label None
    for i in xrange(m):
        if not X_S.has_key(i):
            S_label[i] = None
            unscanned_S[i] = True
    k = 0
    while True: # This is Lawler's step 1 (labeling).  I make it the main loop
        k += 1
        assert(k<2*n*m**2),'k=%d'%k
        for i in unscanned_S.keys():
            # Begin step 1.3 on i
            for j in xrange(n):
                if not w.has_key((i,j)):
                    continue
                if X.has_key((i,j)):
                    continue
                if u[i] + v[j] - w[(i,j)] > pi[j]:
                    continue
                T_label[j] = i
                unscanned_T[j] = True
                pi[j] = u[i] + v[j] - w[(i,j)]
            del unscanned_S[i]
            # End step step 1.3 on i
        for j in unscanned_T.keys():
            if pi[j] > 0:
                continue
            # Begin step step 1.4 on j
            if not X_T.has_key(j):
                # Begin Lawler's step 2 (augmentation)
                if debug:
                    print '\nBefore step 2 augmentation X=',
                    for key in X.keys():
                        print key,
                    print '\nj=%d, S_label='%j,S_label, 'T_label=',T_label
                    print 'Augmenting path=',
                backtrack_j(X,X_S,X_T,S_label,T_label,j)
                if debug:
                    print '\nAfter step 2 augmentation:  X=',
                    for key in X.keys():
                        print key,
                    print ''
                pi = scipy.ones(n)*inf
                S_label = {}
                T_label = {}
                unscanned_S = {}
                unscanned_T = {}
                for i in xrange(m): # Step 1.0
                    if not X_S.has_key(i):
                        S_label[i] = None
                        unscanned_S[i] = True
                if debug:
                    print '              and unscanned_S=',unscanned_S
                # End Lawler's step 2
            else:
                i = X_T[j]
                S_label[i] = j
                unscanned_S[i] = True
                del unscanned_T[j]
        # Begin Lawler's step 1.1 (check for step 3)
        if len(unscanned_S) > 0:
            continue # Start another iteration of the labeling loop
        skip3 = False
        for j in unscanned_T.keys():
            if pi[j] > tol:
                continue # Continue checking j's
            else:
                skip3 = True
                break
        if skip3:
            continue # Start another iteration of the labeling loop
        # End step 1.1
        # Begin Lawler's step 3 (change dual varibles)
        delta_1 = min(u)
        assert(float(pi.min()) > -tol),"float(pi.min())=%f, tol=%f"%(
            pi.min(),tol)
        Lim = pi.max()
        Mask = (pi < tol)
        pi_ = pi + Lim*Mask
        delta = float(pi_.min())
        if delta > delta_1:
            # Finished
            return X
        if debug:
            print '\nBefore step 3 adjustment by delta=%5.3f:'%delta
            print 'u=',u,
            print 'v=',v,
            print 'pi=',pi,
        for i in S_label.keys():
            u[i] -= delta
        for j in xrange(n):
            if pi[j] > tol:
                pi[j] -= delta
            else:
                v[j] += delta
        if debug:
            print '  After step 3 adjustment:'
            print 'u=',u,
            print 'v=',v,
            print 'pi=',pi
            print 'New scannable T node:',
            for j in unscanned_T.keys():
                if pi[j] < tol:
                    print j,
            print '\nS_label=',S_label, 'T_label=',T_label
        # Start another iteration of the labeling loop

class NODE:
    """ Class for Murty's algorithm that find the n best assignments
    for any n.  Note that Murty's paper considers assignments with
    costs, I use utilities here.

    Properties:
    IN           List of used vertices in parent util matrix
    u_in         Total utility of used vertices
    OUT          List of excluded vertices
    util         Utility "matrix" stored as dict/list?
    ij_max       Best assignment in util
    u_max        Utility of ij_max added to u_in

    Methods:
    __init__     New node from utility matrix and other arguments
    partition    Return both nodes and (u_max, node) pairs of
                 partition on self.ij_max
    """
    def __init__(self, # NODE
                 IN,   #
                 OUT,  #
                 u_in, #
                 util, #
                 m,    # Number of different input vertices in util
                 n     # Number of different output vertices in util
                 ):
        self.IN = IN
        self.u_in = u_in
        self.OUT = OUT
        self.util = util
        X = Hungarian(util,m,n)
        self.ij_max = X
        u_max = u_in
        for ij in X:
            u_max += util[ij]
        self.u_max = u_max
    def reduce_in(self,    # NODE
               new_in      # Additional vertex for IN
               ):
        """ Augment self.IN by new_in and modify self.util
        """
        ij_O = new_in.pop()
        self.IN.append(ij_O)
        # tranlate ij from orignal coordinates to current coordinates
        i_strike = self.Oi_2_i[ij_O[0]]
        self.Oi_2_i[i_strike] = -0.5
        for i in xrange(i_strike+1,self.m):
            self.Oi_2_i[i] -= 1
        j_strike = self.Oj_2_j[ij_O[1]]
        self.Oj_2_j[j_strike] = -0.5
        for j in xrange(j_strike+1,self.n):
            self.Oj_2_j[j] -= 1
        # Remove a row and column
        for ij in self.util.keys():
            if ij[0] == i_strike:
                del self.util[ij]
                continue
            if ij[1] == j_strike:
                del self.util[ij]
        # Adjust the indices of the reduced utility matrix
        new = {}
        for ij in self.util.keys():
            i,j = ij
            if i < i_strike and j < j_strike:
                continue
            if i > i_strike:
                i -= 1
            if j > j_strike:
                j -= 1
            new[(i,j)] = self.util[ij]
            del self.util[ij]
        self.util.update(new)  # Fold the dict new into self.util
    def reduce(self,     # NODE
               new_in,   # List of additional vertices for IN
               new_out   # Additional vertices for OUT
               ):
        """ Augment self.OUT by new_out and modify self.util, then
        call reduce_in to handle new_in
        """
        # tranlate ij from orignal coordinates to current coordinates
        i = self.Oi_2_i[new_out[0]]
        j = self.Oj_2_j[new_out[1]]
        del self.util[(i,j)]            
        self.OUT.append(new_out)
        for ij in new_in:
            self.reduce_in(ij)
    def partition(self):
        """ Return both nodes and (u_max, node) pairs of partition on
        self.ij_max
        """
        children = []
        pairs = []
        new_in = [:-1]
        for ij in self.ij_max.keys():
            new_node_parts = self.reduce(new_in,ij)
            new_in.append(ij)
            child = NODE(new_node_parts)
            children.append(child)
            pairs.append((child.u_max,child))
        return (children,pairs)
if __name__ == '__main__':  # Test code
    def print_wx(w,X):
        for i in xrange(m):
            for j in xrange(n):
                if X.has_key((i,j)):
                    print '%5.2f'%w[(i,j)],
                else:
                    print 5*' ',
            print '\n',
        print ''
        return
    # This is tests noninteger, negative and missing entries
    M = scipy.array([
        [ 1, 2, 3, 0, 6],
        [ 2, 4, 0, 2, 7],
        [ 4, 3, 4, 8, 9],
        [-1, 0,-3,-4,-2]
        ])*1.1
    m = 4
    n = 5
    sol = {(0, 2): True, (1, 4): True, (2, 3): True, (3,0):True }
    # This is eaiser to follow
    M = scipy.array([
        [ 7, 3, 2],
        [ 8, 5, 4]
        ])
    m = 2
    n = 3
    sol = {(0, 0): True, (1, 1): True}
    w = {}
    for i in xrange(m):
        for j in xrange(n):
            w_ij = M[i,j]
            if w_ij*w_ij < .1:
                continue
            w[(i,j)] = w_ij
    print 'w='
    print_wx(w,w)
    print 'Call Hungarian.  Expect (within offset) result:'
    print_wx(w,sol)
    X = Hungarian(w,m,n)
    print 'Returned from Hungarian with result:'
    print_wx(w,X)


#---------------
# Local Variables:
# eval: (python-mode)
# End:
