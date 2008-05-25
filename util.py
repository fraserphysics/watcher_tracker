"""
util.py utilities for persistent surveillance tracking

"""
import numpy, scipy, scipy.linalg, random, math, time

debug = False

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

def print_wx(w,X,m,n):
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
    try: # Works if X is dict or list
        keys = X.keys()
    except:
        keys = X
    keys.sort()
    u = 0.0
    print 'U[',
    for key in keys:
        u += w[key]
        #print '(%d,%d)'%(key[0]+1,key[1]+1), # +1 to match Murty's paper
        print '(%d,%d)'%(key[0],key[1]),
    print '] = %5.2f'%u
    return u

def M_2_w(M):
    w = {}
    for i in xrange(m):
        for j in xrange(n):
            w_ij = M[i,j]
            if w_ij*w_ij < .01:
                continue
            w[(i,j)] = w_ij
    return w
def print_cvx_mat(M,name):
    print '%s='%name
    m,n = M.size
    for i in xrange(m):
        for j in xrange(n):
            print '%2d'%int(M[i,j]),
        print '\n',
    print ''
def H_cvx(
    wO,       # dict of weights indexed by tuple (i,j)
    m,        # Cardinality of S (use i \in S)
    n,        # Cardinality of T (use j \in T)
    i_gnd={}, # Nodes in S with unlimited capacity
    j_gnd={}  # Nodes in T with unlimited capacity
    ):
    """ Find the maximum weight assignment with weights given by wO.
    The indices of wO are 2-tuples.  Each tuple has elements i and j.
    The only restriction on the elements i and j is that they are used
    individually to index a dict.
    """
    import cvxopt, cvxopt.base, cvxopt.modeling, cvxopt.solvers
    import cvxopt.glpk

    cvxopt.solvers.options['show_progress'] = False
    cvxopt.glpk.options['LPX_K_MSGLEV'] = 0 # Keep glpk quiet
    wO_keys = wO.keys()
    L = len(wO)
    W = cvxopt.base.matrix(0,(1,L),'d')
    size = (m + n - len(i_gnd) - len(j_gnd),L)
    ST_mat = cvxopt.base.matrix(0,size,'d')
    ij_dict = {}
    # This loop extracts a dense vector W from wO and a matrix ST_mat
    for col in xrange(L):
        ij = wO_keys[col]
        W[col] = wO[ij] + 1e-6 # 1e-6 to keep cvxopt from dropping zeros
        i,j = ij
        i_key = ('i',i)
        j_key = ('j',j)
        if not i_gnd.has_key(i):
            if not ij_dict.has_key(i_key):
                row = len(ij_dict)
                ij_dict[i_key] = row
        if not j_gnd.has_key(j):
            if not ij_dict.has_key(j_key):
                row = len(ij_dict)
                ij_dict[j_key] = row
        for key in [i_key,j_key]:
            if ij_dict.has_key(key):
                row = ij_dict[key]
                ST_mat[row,col] = 1.0
    X = cvxopt.modeling.variable(len(W),'X')
    f = -W*X
    ST_constraint = (ST_mat*X == 1)
    positive = (0 <= X)
    # (X[0]<=2.0) circumvents a bug in cvxopt
    LP = cvxopt.modeling.op(f,[ST_constraint,(X[0]<=2.0),positive])
    LP.solve('dense','glpk')
    #LP.solve()
    assert (LP.status == 'optimal'),'LP.status=%s'%LP.status
    if False:
        print 'LP.solve() failed m=%d, n=%d'%(m,n)
        print ' i_gnd=',i_gnd.keys()
        print ' j_gnd=',j_gnd.keys()
        print ' wO=',wO
        print 'printing varibles',LP.variables()
        print 'printing objective',LP.objective
        print 'printing inequalities',LP.inequalities()
        print 'printing equalities',LP.equalities()
        print '\n*** Calling solve ***'
        LP.solve('dense','glpk')
        #LP.solve()
        assert (LP.status == 'optimal'),'LP.status=%s'%LP.status
    RD = {}
    for k in xrange(L):
        ij = wO_keys[k]
        if X.value[k] > 0.5:
            RD[ij] = True
    #print 'Returning from H_cvx with:'
    #print_wx(wO,RD,m,n)
    #print_wx(wO,wO,m,n)
    return RD
    
def Hungarian(wO,      # dict of weights indexed by tuple (i,j)
              m,       # Cardinality of S (use i \in S)
              n,       # Cardinality of T (use j \in T)
              j_gnd={} # Nodes in T with unlimited capacity
              ):
    """ Find mapping X from S to T that maximizes \sum_{i,j}
    X_{i,j}w_{i,j}.  Return X as a dict with X[(i,j)] = True
    """
    if len(wO) == 0:
        return {}
    Y = {}
    w_list = wO.values()
    Max = max(w_list)
    Min = min(w_list)
    tol = max(1e-25,(Max - Min)*1e-6) # Tolerance for some comparisons
    if Min < tol:                     # Ensure all w[ij] >= 0
        w = {}                        # Don't modify wO
        for key in wO.keys():
            w[key] = wO[key] - Min + 2*tol
        Max = Max - Min + 2*tol
        Min = 2*tol
    else:
        w = wO
    inf = Max*10            # "Infinity" for pi values
    scannable_S = {}
    scannable_T = {}
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
        """Follow path of labels and "augment" the links along the way
        """
        if not S_label.has_key(i):
            return
        j = S_label[i]
        if debug:
            print '(%d,%d)'%(i,j),
        augment(X,X_S,X_T,i,j)
        backtrack_j(X,X_S,X_T,S_label,T_label,j)
    def backtrack_j(X,X_S,X_T,S_label,T_label,j,j_gnd={}):
        if not T_label.has_key(j):
            return
        i = T_label[j]
        if debug:
            print '(%d,%d)'%(i,j),
        if j_gnd.has_key(j):
            if X.has_key((i,j)):
                raise RuntimeError,'X has key (%d,%d) but %d is in j_gnd'%(
                    i,j,j)
            Y[(i,j)] = True
            X_S[i] = j       # Treat S node number i as "covered"
        else:
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
    # Begin Lawler's step 1.0
    for i in xrange(m):
            scannable_S[i] = True
    k = 0
    while True: # This is Lawler's step 1 (labeling).  I make it the main loop
        k += 1
        if k >= 2*n*m**2:
            print "Trouble in Hungarian. m=%d, n=%d, w="%(m,n)
            print_wx(w,w,m,n)
            print 'u=',u
            print 'v=',v
            print 'pi=',pi
            print 'scannable_S.keys()=',scannable_S.keys()
            print 'scannable_T.keys()=',scannable_T.keys()
            print 'X.keys()=',X.keys()
        assert(k<2.5*n*m**2),'k=%d, m=%d, n=%d'%(k,m,n)
        if debug:
            print """
Begin step 1.3: Find pi by searching scannable_s and links not in X."""
            print '  scannable_S=',scannable_S.keys()
            print '  u=',u
            print '  v=',v
            print ' pi=',pi
        for i in scannable_S.keys():
            # Begin step 1.3 on i
            for j in xrange(n):
                if not w.has_key((i,j)):
                    continue
                if X.has_key((i,j)):
                    continue
                if u[i] + v[j] - w[(i,j)] > pi[j] - tol: #FixMe > or >=
                    continue
                T_label[j] = i
                scannable_T[j] = True
                pi[j] = u[i] + v[j] - w[(i,j)]
            del scannable_S[i]
            # End step step 1.3 on i
        if debug:
            print 'Found pi=',pi
        skip3 = False
        for j in scannable_T.keys():
            if pi[j] > tol:
                continue
            # Begin step 1.4 on j
            pi[j] = 0.0
            if not X_T.has_key(j):
                # Begin Lawler's step 2 (augmentation)
                if debug:
                    print """
Since pi[%d]==0 and %d is not in X_T, start step 2 augmentation from j=%d
  with X="""%(j,j,j),
                    for key in X.keys():
                        print key,
                    print '\n S_label=',S_label, 'T_label=',T_label
                    print ' Augmenting path=',
                backtrack_j(X,X_S,X_T,S_label,T_label,j,j_gnd=j_gnd)
                if debug:
                    print '\n After step 2 augmentation:  X=',
                    for key in X.keys():
                        print key,
                    print 'Y=',
                    for key in Y.keys():
                        print key,
                    print ''
                pi = scipy.ones(n)*inf
                S_label = {}
                T_label = {}
                scannable_S = {}
                scannable_T = {}
                for i in xrange(m): # Step 1.0
                    if not X_S.has_key(i):
                        scannable_S[i] = True
                if debug:
                    print '              and scannable_S=',scannable_S.keys()
                # End Lawler's step 2
            else: # X_T.has_key(j) and pi[j] == 0
                skip3 = True
                i = X_T[j]
                S_label[i] = j
                scannable_S[i] = True
                del scannable_T[j]
                if debug:
                    print """
Since pi[%d]==0 and X_T[%d]=%d, set S_label[%d] = %d, mark S[%d] unscanned
and mark T[%d] scanned"""%(j,j,i,i,j,i,j)
        # Begin Lawler's step 1.1 (check for step 3)
        if len(scannable_S) > 0:
            continue # Start another iteration of the labeling loop
        # If there is a scannable j with pi[j] == 0 start labeling loop again
        if skip3:
            continue # Start another iteration of the labeling loop
        # End step 1.1

        # If no scannable j with pi[j] == 0, begin Lawler's step 3
        # (change dual varibles)
        min_u = min(u)
        assert(float(pi.min()) > -tol),\
         "float(pi.min())=%f, tol=%f len(wO)=%d"%(pi.min(),tol,len(wO))
        pi_max = pi.max()
        Mask = (pi < tol)
        pi_ = pi + pi_max*Mask
        delta = float(pi_.min()) # delta = min_{i:pi[i]>0} pi[i]
        if delta > min_u:
            if debug:
                print """
Finished! delta=%5.3f > min_u=%5.3f"""%(delta,min_u)
                print 'u=',u,'v=',v,'pi=',pi
            # Finished
            X.update(Y) # Fold the Y dict into the X dict
            return X
        if debug:
            print """
Before step 3, delta=%5.3f:"""%delta
            print '  u=',u,'v=',v,'pi=',pi
            print '  subtract from u[i] for i in',S_label.keys()
            print '  subtract from pi[j]>0 otherwise add to v[j]'
        for i in S_label.keys():
            u[i] -= delta
        for j in xrange(n):
            if pi[j] > tol:
                pi[j] -= delta
            else:
                v[j] += delta
        for j in j_gnd.keys():
            v[j] = 0
            pi[j] = 0
        if debug:
            print ' After step 3 adjustment:'
            print '  u=',u,'v=',v,'pi=',pi
            print '  New scannable T nodes:',
            for j in scannable_T.keys():
                if pi[j] < tol:
                    print j,
            print '\n  S_label=',S_label, 'T_label=',T_label
        # Start another iteration of the labeling loop
    # End of Hungarian
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
    def __init__(
        self,         # M_NODE
        IN,           # List of vertices included in all a \in M_NODE
        OUT,          # List of vertices excluded from all a \in M_NODE
        u_in,         # Utility of vertices in IN
        util,         # Utility "matrix" stored as dict
        m,            # Number of different input vertices in util
        n,            # Number of different output vertices in util
        i_gnd={},     # Dict of S nodes with unlimited capacity
        j_gnd={}      # Dict of T nodes with unlimited capacity
        ):
        self.IN = IN[:]             # shallow copy list
        self.OUT = OUT[:]           # shallow copy list
        self.u_in = u_in            # Float
        self.util = util.copy()     # shallow copy dict
        self.i_gnd=i_gnd
        self.j_gnd=j_gnd
        try:
            X = H_cvx(util,m,n,i_gnd=i_gnd,j_gnd=j_gnd)
        except: # Indicate that there is no feasible solution
            self.ij_max = None
            self.u_max = None
            return
        self.m = m
        self.n = n
        self.ij_max = X
        u_max = u_in
        for ij in X:
            u_max += util[ij]
        self.u_max = u_max
    def dump(self      #M_NODE
             ):
        print "\nDumping an M_NODE:"
        print "IN=",self.IN
        print "OUT=",self.OUT
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
               new_in,   # List of additional vertices for IN
               new_out   # Vertex for OUT
               ):
        """ Create and retrun an M_NODE called 'new' that is like self
        except:
        
           Vertex new_out removed from new.util and appended to
              new.OUT
        
           for each vertex v in new_in:
              Remove row and column of v unless in *_gnd
              Append v to new.IN
              Add utility of v to new.u_in
        """
        # Create util dict for new node
        i_kill = {}
        j_kill = {}
        new_m = self.m
        new_n = self.n
        for i,j in new_in:
            if not self.i_gnd.has_key(i): # Don't drop i's in self.i_gnd
                i_kill[i] = True
                new_m -= 1
            if not self.j_gnd.has_key(j): # Don't drop j's in self.j_gnd
                j_kill[j] = True
                new_n -= 1
        util = self.util.copy()
        del util[new_out]
        for key in util.keys():
            i,j = key
            if i_kill.has_key(i) or j_kill.has_key(j):
                del util[key]
        u_in = self.u_in
        for ij in new_in:
            u_in += self.util[ij]
        IN = self.IN + new_in
        OUT = self.OUT + [new_out]
        new = M_NODE(IN, self.OUT, u_in, util, new_m, new_n,
                     i_gnd=self.i_gnd,j_gnd=self.j_gnd)
        if new.u_max is None:
            return None # No feasible solution
        return new
    # End of spawn()
    def partition(self # M_NODE
                  ):
        """ Create partition self on maximal assignment, self.ij_max.
        Return list of (u_max, m_node) pairs for nodes of the partition.
        """
        pairs = []
        new_in = []
        for ij in self.ij_max.keys():
            child = self.spawn(new_in,ij)
            if child is None: # No feasible association
                continue
            new_in.append(ij)
            pairs.append((child.u_max,child))
        return pairs
class M_LIST:
    """ A class that implements a list of M_NODEs for Murty's algorithm.

    Properties:

      node_list

      association_list

      H_count           Count of number of calls to Hungarian

    Methods:

      __init__

      next   Find the next best association

      till(N,U) Find more associations until either there are N in
        association_list or you reach an association with utility less
        than U
    """
    def __init__(
        self,     # M_LIST
        w,        # A dict of utilities indexed by tuples (i,j)
        m,        # Number of i values
        n,        # Number of j values
        i_gnd={}, # j values, ie, T nodes with unlimited capacity
        j_gnd={}  # j values, ie, T nodes with unlimited capacity
                 ):
        """
        """
        node_0 = M_NODE([],[],0.0,w,m,n,i_gnd=i_gnd,j_gnd=j_gnd)
        self.association_list = []
        self.i_gnd=i_gnd
        self.j_gnd=j_gnd
        self.H_count = 1
        self.start_time = time.time()
        if node_0.u_max == None:
            self.node_list = []
            return
        self.node_list = [(node_0.u_max,node_0)]
        self.next()
        return
    def next(self,    # M_LIST
                 ):
        """ Find the next best association, put it in
        self.association_list, and partition the M_NODE from which it
        came.
        """
        u,node = self.node_list.pop()
        A =  node.IN + node.ij_max.keys()
        self.association_list.append((u,A))
        pairs = node.partition()
        self.H_count += len(pairs)
        self.node_list += pairs
        self.node_list.sort()
        return
    def till(self,    # M_LIST
             N,
             U
                 ):
        """ Call self.next until either len(self.association_list) >= N or
        utility(association_list[-1]) <= U or no more associations are
        possible.
        """
        while len(self.association_list) < N and len(self.node_list) > 0:
            self.next()
            if self.association_list[-1][0] < U:
                self.stop_time = time.time()
                return
        self.stop_time = time.time()
        return
if __name__ == '__main__':  # Test code
    # This is tests noninteger, negative and missing entries
    M = scipy.array([
        [ 1, 2, 4, 0, 6],
        [ 2, 4, 0, 2, 7],
        [ 4, 3, 4, 7, 9],
        [-3, 0, 1,-4,-2],
        [ 4, 3, 4, 8, 9]
        ],'float')
    m = 5
    n = 5
    sol = {(0, 0): True, (1, 1): True, (2,4):True, (3, 2): True, (4,3):True }
    test0 = (M,m,n,sol)
    # This is eaiser to follow
    M = scipy.array([
        [ 4, 5, 8],
        [ 2, 3, 7]
        ])
    m = 2
    n = 3
    sol = {(1, 2): True, (0, 1): True}
    test1 = (M,m,n,sol)
    # This is the matrix in Murty's paper (scaled)
    M = (100-scipy.array([
        [ 7,  51, 52, 87, 38, 60, 74, 66, 0,  20 ],
        [ 50, 21, 0,  64, 8,  53, 0,  46, 76, 42 ],
        [ 27, 77, 0,  18, 22, 48, 44, 13, 0,  57 ],
        [ 62, 0,  3,  8,  5,  6,  14, 0,  26, 39 ],
        [ 0,  97, 0,  5,  13, 0,  41, 31, 62, 48 ],
        [ 79, 68, 0,  0,  15, 12, 17, 47, 35, 43 ],
        [ 76, 99, 48, 27, 34, 0,  0,  0,  28, 0 ],
        [ 0,  20, 9,  27, 46, 15, 84, 19, 3,  24 ],
        [ 56, 10, 45, 39, 0,  93, 67, 79, 19, 38 ],
        [ 27, 0,  39, 53, 46, 24, 69, 46, 23, 1 ]
        ]))/10.0
    m = 10
    n = 10
    sol = {(6, 9): True, (0, 8): True, (7, 0): True, (9, 1): True,
           (4, 5): True, (1, 6): True, (2, 2): True, (3, 7): True,
           (5, 3): True, (8, 4): True}
    test2 = (M,m,n,sol)
    M,m,n,sol = test0 # Funny way to enable different tests with small edit
    debug = False
    w = M_2_w(M)
    print 'w='
    print_wx(w,w,m,n)
    print 'Call Hungarian.  Expect result:'
    print_wx(w,sol,m,n)
    X = Hungarian(w,m,n)
    print 'Returned from Hungarian with result:'
    print_wx(w,X,m,n)
    X = H_cvx(w,m,n)
    print 'Returned from H_cvx with result:'
    print_wx(w,X,m,n)
    print 'Calling H_cvx with j_gnd={j:True,1:True} yields:'
    X = H_cvx(w,m,n,j_gnd={2:True,1:True})
    print_wx(w,X,m,n)
    print 'Calling H_cvx with i_gnd={2:True,1:True} yields:'
    X = H_cvx(w,m,n,i_gnd={2:True,1:True})
    print_wx(w,X,m,n)
    print 'Calling H_cvx with i_gnd={1:True} & j_gnd={2:True} yields:'
    X = H_cvx(w,m,n,i_gnd={1:True},j_gnd={2:True})
    print_wx(w,X,m,n)
    M,m,n,sol = test2
    w = M_2_w(M)
    ML = M_LIST(w,m,n)
    ML.till(10,95)
    print "Result of Murty's algorithm:"
    for U,X in ML.association_list:
        X.sort()
        print_x(X,w)
    # Test with no capacity limit for row and column
    M,m,n,sol = test0
    w = M_2_w(M)
    ML = M_LIST(w,m,n,i_gnd={1:True},j_gnd={2:True})
    ML.till(200,10) # FixMe should quit gracefully if till asks for too much
    print "Result of Murty's algorithm with i_gnd={1:True},j_gnd={2:True}}:"
    for U,X in ML.association_list:
        X.sort()
        print_x(X,w)


#---------------
# Local Variables:
# eval: (python-mode)
# End:
