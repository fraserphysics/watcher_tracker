"""
mv1.py unrealistic code for small simple models

"""
import numpy, scipy, scipy.linalg, random, math

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

class MV1:
    """A simple model of observed motion with the following groups of
    methods:

    Tools for applications:
      forward()
      decode()
      simulate()

    Service methods: check()

    Debugging method: dump()

    """
    def __init__(self,
                 N_obj = 3,                      # Number of objects
                 A = [[0.81,1],[0,.81]],         # Linear state dynamics
                 Sigma_D = [[0.01,0],[0,0.4]],   # Dynamical noise
                 O = [[1,0]],                    # Observation porjection
                 Sigma_O = [[0.25]],             # Observational noise
                 Sigma_init = [[25.0,0],[0,1.0]],# Initial state distribution
                 mu_init = None                  # Initial state distribution
                 ):
        self.N_obj = N_obj
        self.A = scipy.matrix(A)
        self.Sigma_D = scipy.matrix(Sigma_D)
        self.O = scipy.matrix(O)
        self.Sigma_O = scipy.matrix(Sigma_O)
        self.N_perm = int(scipy.factorial(N_obj))
        self.Sigma_init = scipy.matrix(Sigma_init)
        if mu_init == None:
            dim = self.Sigma_init.shape[0]
            self.mu_init = scipy.matrix(scipy.zeros(dim))
        else:
            self.mu_init = scipy.matrix(mu_init)

    def simulate(self, T):
        """ Return a sequence of T observations and a sequence of T
        states."""
        s_j = []
        for j in xrange(self.N_obj):
            s_j.append(normalS(self.mu_init,self.Sigma_init))

        s_dim = self.mu_init.shape[1]
        zero_s = scipy.matrix(scipy.zeros(s_dim))
        y_dim = self.Sigma_O.shape[0]
        zero_y = scipy.matrix(scipy.zeros(y_dim))
        
        obs = []
        states = []
        for t in xrange(T):
            permute = index_to_permute(random.randint(0,self.N_perm-1),self.N_obj)
            obs_t = range(self.N_obj) # Make list with length N_obj
            state_t = []
            for j in xrange(self.N_obj):
                epsilon = normalS(zero_s,self.Sigma_D) # Dynamical noise
                eta = normalS(zero_y,self.Sigma_O) # Observational noise
                s_t = self.A*s_j[j] + epsilon
                y_t = self.O * s_t + eta
                s_j[j] = s_t
                
                obs_t[permute[j]] = y_t
                state_t.append(s_t)
                
            obs.append(obs_t)
            states.append(state_t)
        return obs,states
    
#---------------
# Local Variables:
# eval: (python-mode)
# End:
