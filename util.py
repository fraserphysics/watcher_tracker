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
def print_x(X,w):
    keys = X.keys()
    keys.sort()
    u = 0.0
    print 'U[',
    for key in keys:
        u += w[key]
        print '(%d,%d)'%(key[0]+1,key[1]+1),
    print '] = %5.2f'%u
    return u
    
def Hungarian(w,  # dict of weights indexed by tuple (i,j)
              m,  # Cardinality of S (use i \in S)
              n   # Cardinality of T (use j \in T)
              ):
    """ Find mapping X from S to T that maximizes \sum_{i,j}
    X_{i,j}w_{i,j}.  Return X as a dict with X[i] = j
    """
    debug = True
    debug = False
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

class M_NODE:
    """ Class for Murty's algorithm that finds the n best assignments
    for any n.  Note that Murty's paper considers assignments with
    costs, I use utilities here.

    Properties:
    IN           List of used vertices in parent util matrix
    u_in         Total utility of used vertices
    OUT          List of excluded vertices
    util         Utility "matrix" stored as dict
    m,n          Dimensions of util
    ij_max       Best assignment in util
    u_max        Utility of ij_max added to u_in

    Methods:
    __init__     New m_node from utility matrix and other arguments
    spawn        Modify m_node for a single new_out and list of new_in
    partition    Return both m_nodes and (u_max, m_node) pairs of
                 partition on self.ij_max

Citation: An Algorithm for Ranking all the Assignments in Order of
Increasing Cost Katta G. Murty Operations Research, Vol. 16, No. 3
(May - Jun., 1968), pp. 682-687
    """
    def __init__(self, # M_NODE
                 IN,   # List of vertices included in all a \in M_NODE
                 OUT,  # List of vertices excluded from all a \in M_NODE
                 u_in, # Utility of vertices in IN
                 util, # Utility "matrix" stored as dict
                 m,    # Number of different input vertices in util
                 n,    # Number of different output vertices in util
                 Ri_2_Oi=None, # List for mapping reduced i to original
                 Rj_2_Oj=None  # List for mapping reduced j to original
                 ):
        if Ri_2_Oi is None:
            self.Ri_2_Oi = range(m)
        else:
            self.Ri_2_Oi = Ri_2_Oi[:] # shallow copy list
        if Rj_2_Oj is None:
            self.Rj_2_Oj = range(n)
        else:
            self.Rj_2_Oj = Rj_2_Oj[:] # shallow copy list
        self.IN = IN[:]             # shallow copy list
        self.OUT = OUT[:]           # shallow copy list
        self.u_in = u_in            # Float
        self.util = util.copy()     # shallow copy dict
        X = Hungarian(util,m,n)
        self.ij_max = X
        u_max = u_in
        for ij in X:
            u_max += util[ij]
        self.u_max = u_max
        self.m = m
        self.n = n
    def dump(self      #M_NODE
             ):
        print "\nDumping an M_NODE:"
        print "IN=",self.IN
        print "OUT=",self.OUT
        print "Ri_2_Oi=",self.Ri_2_Oi
        print "Rj_2_Oj=",self.Rj_2_Oj
        print "ij_max=",self.ij_max.keys()
        print "u_in=%5.2f,  u_max=%5.2f, m=%d, n=%d, util="%(self.u_in,
            self.u_max, self.m, self.n)
        for i in xrange(self.m):
            for j in xrange(self.n):
                if self.util.has_key((i,j)):
                    print '%5.2f'%self.util[(i,j)],
                else:
                    print 5*' ',
            print '\n',
        print ''
        return
    def spawn(self,      # M_NODE
               new_in,   # List of additional vertices in reduced form for IN
               new_out   # Additional vertex for in reduced form OUT
               ):
        """ Augment self.OUT by new_out and modify self.util, then
        call reduce_in to handle new_in
        """
        new = M_NODE(self.IN, self.OUT, self.u_in, self.util, self.m,
                   self.n, self.Ri_2_Oi, self.Rj_2_Oj) # FixMe: build parts before this
        del new.util[new_out] 
        # tranlate ij from orignal coordinates to current coordinates
        Ri,Rj = new_out
        new.OUT.append((self.Ri_2_Oi[Ri],self.Rj_2_Oj[Rj]))
        # Now remove the row and coulumn for each ij in new_in from util
        self.m -= len(new_in)
        self.n -= len(new_in)
        i_map = dict(map(lambda x: (x,True),range(self.m)))
        j_map = dict(map(lambda x: (x,True),range(self.n)))
        for Ri,Rj in new_in:
            self.IN.append((self.Ri_2_Oi[Ri],self.Rj_2_Oj[Rj]))
            del i_map[Ri]
            del j_map[Rj]
        temp = i_map.keys()
        temp.sort()
        i_map = {}
        new.Ri_2_Oi = []
        for k in xrange(len(temp)):
            i = temp[k]
            new.Ri_2_Oi.append(self.Ri_2_Oi[i])
            i_map[i] = k
        temp = j_map.keys()
        temp.sort()
        j_map = {}
        new.Rj_2_Oj = []
        for k in xrange(len(temp)):
            j = temp[k]
            new.Rj_2_Oj.append(self.Rj_2_Oj[j])
            j_map[j] = k
        new.util = {}
        for key in self.util.keys():
            i,j = key
            if i_map.has_key(i) and j_map.has_key(j):
                new.util[(i_map[i],j_map[j])] = self.util[key]
        return new
    def partition(self # M_NODE
                  ):
        """ Return both m_nodes and (u_max, m_node) pairs of partition on
        self.ij_max
        """
        if self.m == 0 or len(self.util) == 0:
            return ([],[])
        children = []
        pairs = []
        new_in = []
        for ij in self.ij_max.keys():
            try:
                child = self.spawn(new_in,ij)
            except:
                print 'Calling spawn with len(new_in)=%d, ij='%len(new_in),ij
                print '   new_in=',new_in,'self.util='
                print_wx(self.util,self.util)
                child = self.spawn(new_in,ij)
            new_in.append(ij)
            children.append(child)
            pairs.append((child.u_max,child))
        return (children,pairs)
class M_LIST:
    """ A class that implements a list of M_NODEs for Murty's algorithm.

    Properties:

      node_list

      association_list

    Methods:

      __init__

      next   Find the next best association

      till(N,U) Find more associations until either there are N in
        association_list or you reach an association with utility less
        than U
    """
    def __init__(self, # M_LIST
                 w,    # A dict of utilities indexed by tuples (i,j)
                 m,    # Range of i values
                 n     # Range of j values
                 ):
        """
        """
        node_0 = M_NODE([],[],0.0,w,m,n)
        self.node_list = [(node_0.u_max,node_0)]
        self.association_list = []
        self.next()
        return
    def next(self,    # M_LIST
                 ):
        u,node = self.node_list.pop()
        self.association_list.append((u,node.IN+node.ij_max.keys()))
        try:
            children,pairs = node.partition()
        except:
            node.dump()
            children,pairs = node.partition()
        self.node_list += pairs
        self.node_list.sort()
        return
    """ Find the next best association, put it in
    self.association_list, and partition the M_NODE from which it
    came.
    """
    def till(self,    # M_LIST
             N,
             U
                 ):
        """ Call self.next until either len(self.association_list) >=
        N or utility(association_list[-1]) <= U.
        """
        while len(self.association_list) < N \
              and self.association_list[0][0] > U:
            print 'Starting another iteration of while loop in till'
            print '  association_list=',self.association_list
            self.next()
        return
if __name__ == '__main__':  # Test code
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
    test0 = (M,m,n,sol)
    # This is eaiser to follow
    M = scipy.array([
        [ 7, 3, 2],
        [ 8, 5, 4]
        ])
    m = 2
    n = 3
    sol = {(0, 0): True, (1, 1): True}
    test1 = (M,m,n,sol)
    # This is the matrix in Murty's paper (scaled)
    M = (100-scipy.array([
        [ 7, 51, 52, 87, 38, 60, 74, 66, 0, 20 ],
        [ 50, 21, 0, 64, 8, 53, 0, 46, 76, 42 ],
        [ 27, 77, 0, 18, 22, 48, 44, 13, 0, 57 ],
        [ 62, 0, 3, 8, 5, 6, 14, 0, 26, 39 ],
        [ 0, 97, 0, 5, 13, 0, 41, 31, 62, 48 ],
        [ 79, 68, 0, 0, 15, 12, 17, 47, 35, 43 ],
        [ 76, 99, 48, 27, 34, 0, 0, 0, 28, 0 ],
        [ 0, 20, 9, 27, 46, 15, 84, 19, 3, 24 ],
        [ 56, 10, 45, 39, 0, 93, 67, 79, 19, 38 ],
        [ 27, 0, 39, 53, 46, 24, 69, 46, 23, 1 ]
        ]))/10.0
    m = 10
    n = 10
    sol = {(6, 9): True, (0, 8): True, (7, 0): True, (9, 1): True, (4, 5): True, (1, 6): True, (2, 2): True, (3, 7): True, (5, 3): True, (8, 4): True}
    test2 = (M,m,n,sol)
    M,m,n,sol = test2 # Funny way to enable different tests with small edit
    w = {}
    for i in xrange(m):
        for j in xrange(n):
            w_ij = M[i,j]
            if w_ij*w_ij < .01:
                continue
            w[(i,j)] = w_ij
    #print 'w='
    #print_wx(w,w)
    #print 'Call Hungarian.  Expect (within offset) result:'
    #print_wx(w,sol)
    #X = Hungarian(w,m,n)
    #print 'Returned from Hungarian with result:'
    #print_wx(w,X)
    X = Hungarian(w,m,n)
    print_x(X,w)
    print 'before M_LIST'
    ML = M_LIST(w,m,n)
    print 'before till'
    ML.till(10,90)
    for U,X in ML.association_list:
        print_x(X,w)


#---------------
# Local Variables:
# eval: (python-mode)
# End:
