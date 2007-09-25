"""
mv1.py unrealistic code for small simple models

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

class MV1:
    """A simple model of observed motion with the following groups of
    methods:

    Tools for applications:
      forward()
      decode()
      simulate()

    Service methods:
      check()
      __init__()

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
            self.mu_init = scipy.matrix(scipy.zeros(dim)).T
        else:
            self.mu_init = scipy.matrix(mu_init).T

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
            if t>0:
                i = random.randint(0,self.N_perm-1)
            else:
                i = 0 # No shuffle for t=0
            permute = index_to_permute(i,self.N_obj)
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
    
    def new_nu_k(self,
                 old_nu_k, # The nu_s dict for the object last time
                 y         # The observation of the object at the current time
                 ):
        """ Return the dict of nu_s for the object at the current time"""

        # Unpack necessary matrices from self and old_nu_k
        mu_t = old_nu_k['mu']
        Sig_t = old_nu_k['Sigma']
        R_t = old_nu_k['R']
        
        O = self.O
        Sig_O_I = scipy.linalg.inv(self.Sigma_O)
        Sig_D = self.Sigma_D
        A = self.A

        # Calculate new values
        X =  scipy.linalg.inv(Sig_D + A*Sig_t*A.T) # An intermediate result
        Sig_tnew_I = O.T*Sig_O_I*O + X
        Sig_tnew = scipy.linalg.inv(Sig_tnew_I)
        mu_tnew = A*mu_t + Sig_tnew*O.T*Sig_O_I*(y-O*A*mu_t)
        R_tnew = R_t - float(mu_t.T*A.T*X*A*mu_t - mu_tnew.T*Sig_tnew_I*mu_tnew)/2
        new_nu_k = {}
        new_nu_k['mu'] = mu_tnew
        new_nu_k['Sigma'] = Sig_tnew
        new_nu_k['R'] = R_tnew
        return new_nu_k
    
    def new_nu(self,
               nu_oldt, # List of nu_s dicts for each object
               perm_new, # Perm dict for current permutation
               y_t       # List of current observations of each object
              ):
        """ Return a list consisting of a nu_s dict for each object"""
        p_vec = perm_new['p_vector']
        return_nu = []
        for k in xrange(len(nu_oldt)):
            return_nu.append(self.new_nu_k(nu_oldt[k],y_t[p_vec[k]]))
        return return_nu
    
    def decode(self,
               Ys # List of lists of observations
               ):
        """Return MAP state sequence """
        T = len(Ys)

        # Set up storage for nu's and B's
        perm_list = []
        for p in xrange(self.N_perm):
            perm = {}
            perm['p_vector'] = index_to_permute(p,self.N_obj)
            perm['B_p'] = []  # List of best predecessor permutations
            perm['nu_s'] = [] # List of utlity of best path to s
            for t in xrange(T):
                perm['B_p'].append(None)
                nu_s_t = []
                for k in xrange(self.N_obj):
                    nu_s_t_k = {}
                    nu_s_t_k['mu'] = None
                    nu_s_t_k['Sigma'] = None
                    nu_s_t_k['R'] = None
                    nu_s_t.append(nu_s_t_k)
                perm['nu_s'].append(nu_s_t)
            perm_list.append(perm)

        # For first time step (t=0), only consider permutation zero
        # and fudge the remainders so that permutation zero will be
        # decoded
        old_nu = []
        for k in xrange(self.N_obj):
            nu_s_t_k = {}
            nu_s_t_k['mu'] = self.mu_init
            nu_s_t_k['Sigma'] = self.Sigma_init
            nu_s_t_k['R'] = 0.0
            old_nu.append(nu_s_t_k)
        for perm_new in perm_list:
            perm_new['nu_s'][0] = self.new_nu(old_nu,perm_new,Ys[0])
        for perm_new in perm_list[1:]: # For every perm except first
            for nu_s_t_k in perm_new['nu_s'][0]:
                nu_s_t_k['R'] -= 1     # Make first permutation at
                                       # first time the best
        
        # Forward pass through time
        for t in xrange(1,T):
            for perm_new in perm_list:
                trial_nu_R = scipy.zeros(self.N_perm)
                trial_nu = []
                for p in xrange(self.N_perm):
                    old_nu = perm_list[p]['nu_s'][t-1]
                    trial_nu.append(self.new_nu(old_nu,perm_new,Ys[t]))
                    R = 0.0
                    for nu_s_t_k in trial_nu[p]:
                        R = R + nu_s_t_k['R']
                    trial_nu_R[p] = R
                b = trial_nu_R.argmax()
                perm_new['nu_s'][t] = trial_nu[b]
                perm_new['B_p'][t] = b

        # Find the best last permutation and state
        R = scipy.zeros(self.N_perm)
        for p in xrange(self.N_perm):
            nu_last = perm_list[p]['nu_s'][T-1]
            R_p = 0.0
            for nu_last_k in nu_last:
                R_p = R_p + nu_last_k['R']
        b = R.argmax()
        
        s_all = range(T)
        s_old = []
        for k in xrange(self.N_obj):
            best_last_k = perm_list[b]['nu_s'][t][k]['mu']
            s_old.append(best_last_k)
        s_all[T-1] = s_old
        
        # Backtrack to get trajectories
        A = self.A
        X = A.T * scipy.linalg.inv(self.Sigma_D) # An intermediate
        for t in xrange(T-2,-1,-1):
            b = perm_list[b]['B_p'][t+1]
            perm_t = perm_list[b]
            s_t = []
            for k in xrange(self.N_obj):
                nu_t_k = perm_t['nu_s'][t][k]
                Sig_t_I = scipy.linalg.inv(nu_t_k['Sigma'])
                mu_t = nu_t_k['mu']
                
                s_t.append(scipy.linalg.inv(Sig_t_I + X*A)*
                (Sig_t_I*mu_t + X*s_old[k]))
            
            s_all[t] = s_t
            s_old = s_t
        return s_all
            
#---------------
# Local Variables:
# eval: (python-mode)
# End: