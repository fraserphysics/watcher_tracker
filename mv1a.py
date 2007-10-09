"""
mv1a.py: First variation on mv1.  The model is the same, but the
classes and algorithms are different.
"""
import numpy, scipy, scipy.linalg, random, math

import util

class TARGET:
    """A TARGET is a possible moving target, eg, a car.  Its values
    are determined by initialzation values and a sequence of
    observations that are "Kalman filtered".
    """
    def __init__(self,
                 mod,      # Parent model
                 m_t,      # History of hit indices used
                 mu_t,     # History of means
                 Sigma_t,  # History of variances
                 R_t       # History of residuals
                 ):
        self.mod = mod
        self.m_t = m_t
        self.mu_t = mu_t
        self.Sigma_t = Sigma_t
        self.R_t = R_t
        self.children = None # List of targets at t+1 updated with
                             # plausible hits and this target

    def make_children(self,        # self is a TARGET
                      y_t,         # list of hits at time t
                      All_children # Dict of children of all permutations
                      ):
        """ For each of the hits that could plausibly be an
        observation of self, make a child target.  Collect the
        children in a dict, attach it to self and return it.
        """
        self.forecast()
        self.children = {}
        if self.mod.MaxD > 0.01: # Make a child for any hit closer than MaxD
            MD = self.mod.MaxD
            for k in xrange(len(y_t)):
                distance = self.distance(y_t[k])
                if distance < MD:
                    key = tuple(self.m_t+[k])
                    if All_children.has_key(key):
                        self.children[k] = All_children[key]
                        continue
                    self.children[k] = self.update(y_t[k],k)
                    All_children[key] = self.children[k]
        else:   # MaxD is near zero, ie, pruning is off
            for k in xrange(len(y_t)):
                key = tuple(self.m_t+[k])
                if All_children.has_key(key):
                    self.children[k] = All_children[key]
                    continue
                self.children[k] = self.update(y_t[k],k)
                All_children[key] = self.children[k]
    def forecast(self):
        """ Calculate forecast mean and covariance for both state and
        observation.  Also calculate K and Sigma_next.  Save all six
        for use by Update or Distance.
        """
        A = self.mod.A
        O = self.mod.O
        self.mu_a = A*self.mu_t[-1]
        self.Sigma_a = A*self.Sigma_t[-1]*A.T + self.mod.Sigma_D
        self.y_forecast = O*self.mu_a
        Sig_y = O*self.Sigma_a*O.T + self.mod.Sigma_O
        self.Sigma_y_forecast_I = scipy.linalg.inv(Sig_y)
        self.K = self.Sigma_a*self.mod.O.T*self.Sigma_y_forecast_I
        self.Sigma_next = (self.mod.Id-self.K*self.mod.O)*self.Sigma_a
    def distance(self,y):
        Delta_y = y - self.y_forecast    # Error of forecast observation
        d_sq = Delta_y.T*self.Sigma_y_forecast_I*Delta_y
        return float(d_sq)**.5
    def update(self,
           y,        # The observation of the target at the current time
           m         # Index of the observation
           ):
        """ Create a new target with updated m_t, mu_t, Sigma_t and
        R_t for the observation, index pair (y,m).  This is the second
        half of Kalman filtering step."""
        Delta_y = y - self.y_forecast    # Error of forecast observation
        m_L = self.m_t+[m]
        Sigma_L = self.Sigma_t + [self.Sigma_next]
        mu_L = self.mu_t + [self.mu_a + self.K*Delta_y]
        R_L = self.R_t + [self.R_t[-1]
              - float(Delta_y.T*self.Sigma_y_forecast_I*Delta_y)/2]
        return TARGET(self.mod,m_L,mu_L,Sigma_L,R_L)
        
    def KF(self,
           y,        # The observation of the target at the current time
           m         # Index of the observation
           ):
        """ Create a new target with updated m_t, mu_t, Sigma_t and
        R_t for the observation, index pair (y,m).  This is
        essentially Kalman filtering."""
        self.forecast()
        return self.update(y,m)
 
    def backtrack(self):
        T = len(self.mu_t)
        A = self.mod.A
        X = A.T * scipy.linalg.inv(self.mod.Sigma_D) # An intermediate
        s_t = range(T)
        s_t[T-1] = self.mu_t[T-1]
        for t in xrange(T-2,-1,-1):
            Sig_t_I = scipy.linalg.inv(self.Sigma_t[t])
            mu_t = self.mu_t[t]
            s_t[t]=scipy.linalg.inv(Sig_t_I + X*A)*(Sig_t_I*mu_t + X*s_t[t+1])
        return s_t
        
class PERMUTATION:
    """A representation of a particular association of hits to targets
    at a particular time.

    Methods:

     forward: Create plausible sucessor permutations

     argmax:  Select best predecessor, evaluate self.nu, collect list
              of appropriate child targets from that predecessor,
              attach the list to self, and return the list

     make_children:  Call target.make_children() for each target

       
    """

    def __init__(self,              # Permutation
                 N_tar,
                 key,
                 targets=None
                 ):
        self.N_tar = N_tar
        self.key = key              # Permutation tuple: hits -> targets
        self.targets = targets      # List of targets
        self.predecessor_perm = []  # List of predecessor permutations
        self.predecessor_u_prime=[] # List of u' values for the predecessors
        self.nu = None              # Utility of best path ending here
        
    def forward(self,
                new_perms   # A dict of permutations for the next time step
                ):
        """ For each plausible successor S of the PERMUTATION self
        append the following pair of values to S.predecessor: 1. A
        pointer back to self and 2. The value of u'(self,S,t+1).
        """
        # Create a list of successor permutations to consider
        old_list = []
        for child in self.targets[0].children.values():
            m_tail = child.m_t[-1]
            old_list.append({
                'dup_check':{m_tail:None}, # Hash table to ensure unique
                                           # hit associations
                'perm':[m_tail],           # Map from targets to hits
                'R':child.R_t[-1]          # u'(self,suc,t+1)
                })
        for k in xrange(1,len(self.targets)):
            new_list = []
            for child in self.targets[k].children.values():
                m_tail = child.m_t[-1]
                for partial in old_list:
                    if partial['dup_check'].has_key(m_tail):
                        continue
                    new_dict = partial['dup_check'].copy()
                    new_dict[m_tail] = None
                    new_perm = partial['perm']+[m_tail]
                    new_R = partial['R']+child.R_t[-1]
                    new_list.append({'dup_check':new_dict,'perm':new_perm,
                                     'R':new_R})
            old_list = new_list
        # y[t+1][old_list[i]['perm'][j]] is associated with target[j]

        # Initialize successors if necessary and set their predecessors
        for entry in old_list:
            key = tuple(entry['perm'])  # Dict keys can be tuples but not lists
            if not new_perms.has_key(key):
                new_perms[key] = PERMUTATION(self.N_tar,key)
            successor = new_perms[key]
            successor.predecessor_perm.append(self)
            successor.predecessor_u_prime.append(entry['R'])
    def argmax(self):
        """Select best predecessor, evaluate self.nu, collect list of
        appropriate child targets from that predecessor, and attach
        that list to self
        """
        k_max = util.argmax(self.predecessor_u_prime)
        self.nu = self.predecessor_u_prime[k_max]
        best = self.predecessor_perm[k_max]
        self.targets = []
        for k in xrange(self.N_tar):
            self.targets.append(best.targets[k].children[self.key[k]])
    def make_children(self, # self is a PERMUTATION
                      y_t,  # All observations at time t
                      cousins
                      ):
        for target in self.targets:
            target.make_children(y_t,cousins)
class MV1a:
    """A simple model of observed motion with the following groups of
    methods:

    Tools for applications:
      decode()
      simulate()

    Service methods:
      __init__()
    """
    def __init__(self,                           # MVa1
                 N_tar = 3,                      # Number of targets
                 A = [[0.81,1],[0,.81]],         # Linear state dynamics
                 Sigma_D = [[0.01,0],[0,0.4]],   # Dynamical noise
                 O = [[1,0]],                    # Observation porjection
                 Sigma_O = [[0.25]],             # Observational noise
                 Sigma_init = None,              # Initial state distribution
                 mu_init = None,                 # Initial state distribution
                 MaxD = 0,                       # Threshold for hits
                 MaxP = 120
                 ):
        self.N_tar = N_tar
        self.A = scipy.matrix(A)
        dim,check = self.A.shape
        if dim != check:
            raise RuntimeError,"A should be square but it's dimensions are (%d,%d)"%(dim,check)
        self.Id = scipy.matrix(scipy.identity(dim))
        self.Sigma_D = scipy.matrix(Sigma_D)
        self.O = scipy.matrix(O)
        self.Sigma_O = scipy.matrix(Sigma_O)
        self.N_perm = int(scipy.factorial(N_tar))
        if mu_init == None:
            self.mu_init = scipy.matrix(scipy.zeros(dim)).T
        else:
            self.mu_init = scipy.matrix(mu_init).T
        if Sigma_init == None:# A clever man would simply solve
            Sigma_init = scipy.matrix(scipy.identity(dim))
            for t in xrange(100):
                Sigma_init = self.A*Sigma_init*self.A.T + self.Sigma_D
            self.Sigma_init = Sigma_init
        else:
            self.Sigma_init = scipy.matrix(Sigma_init)
        self.MaxD = MaxD
        self.MaxD_limit = MaxD
        self.MaxP = MaxP

    def simulate(self, T):
        """ Return a sequence of T observations and a sequence of T
        states."""
        s_j = []
        for j in xrange(self.N_tar):
            s_j.append(util.normalS(self.mu_init,self.Sigma_init))

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
            permute = util.index_to_permute(i,self.N_tar)
            obs_t = range(self.N_tar) # Make list with length N_tar
            state_t = []
            for j in xrange(self.N_tar):
                epsilon = util.normalS(zero_s,self.Sigma_D) # Dynamical noise
                eta = util.normalS(zero_y,self.Sigma_O) # Observational noise
                s_t = self.A*s_j[j] + epsilon
                y_t = self.O * s_t + eta
                s_j[j] = s_t
                obs_t[permute[j]] = y_t
                state_t.append(s_t)
            obs.append(obs_t)
            states.append(state_t)
        return obs,states

    def decode(self,
               Ys # Observations. Ys[t][k] is the kth hit at time t
               ):
        """Return MAP state sequence """
        T = len(Ys)

        # Initialize by making a target for each of the hits at t=0
        # and collecting those targets in a single permutation.
        target_0 = TARGET(self,[],[self.mu_init],[self.Sigma_init],[0.0])
        targets = []
        for k in xrange(self.N_tar):
            target_k = target_0.KF(Ys[0][k],k)
            for list in (target_k.mu_t,target_k.Sigma_t,target_k.R_t):
                del(list[0])
            targets.append(target_k)
        key = tuple(range(self.N_tar))
        old_perms = {key:PERMUTATION(self.N_tar,key,targets=targets)}
        
        # Forward pass through time
        for t in xrange(1,T):
            len_new_perms = 0
            child_targets = {} # Dict of all targets for time t
            while len_new_perms is 0:
                # For each old target, collect all plausibly
                # associated hits and create a corresponding child
                # target by Kalman filtering
                for perm in old_perms.values():
                    perm.make_children(Ys[t],child_targets)
                # Build u' lists for all possible successor
                # permutations at time t.  Use dict for new_perms so
                # many predecessors can find same successor 
                new_perms = {}       
                for perm in old_perms.values():
                    perm.forward(new_perms)

                len_new_perms = len(new_perms.keys())
                self.MaxD *= 2
            self.MaxD = max(self.MaxD/4,self.MaxD_limit)

            # For each permutation at time t, find the best predecessor
            # and the associated targets
            for perm in new_perms.values():
                perm.argmax()

            # Pass up to MaxP new_perms to old_perms
            if len(new_perms) > self.MaxP:
                Rs = []
                for perm in new_perms.values():
                    Rs.append(perm.nu)
                Rs.sort()
                limit = Rs[-self.MaxP]
                old_perms = {}
                for key in new_perms.keys():
                    if new_perms[key].nu >= limit:
                        old_perms[key] = new_perms[key]
            else:
                old_perms = new_perms

        # Find the best last permutation
        keys = old_perms.keys()
        R = scipy.zeros(len(keys))
        for i in xrange(len(keys)):
            R[i] = old_perms[keys[i]].nu
        perm_best = old_perms[keys[R.argmax()]]
        
        # Backtrack to get trajectories

        tracks = []
        for target in perm_best.targets:
            tracks.append(target.backtrack())
        # "Transpose" for backward compatibility
        s_all = []
        for t in xrange(len(tracks[0])):
            s_t = []
            for k in xrange(len(tracks)):
                s_t.append(tracks[k][t])
            s_all.append(s_t)
        return s_all

# Test code
if __name__ == '__main__':
    import time
    random.seed(3)
    ts = time.time()
    M = MV1a(N_tar=4)
    y,s = M.simulate(5)
    d = M.decode(y)
    print 'len(y)=',len(y), 'len(s)=',len(s),'len(d)=',len(d)
    for t in xrange(len(y)):
        print 't=%d    y         s           d'%t
        for k in xrange(len(y[t])):
            print ' k=%d  %4.2f  '%(k,y[t][k][0,0]),
            for f in (s[t][k],d[t][k]):
                print '(%4.2f, %4.2f)  '%(f[0,0],f[1,0]),
            print ' '
    print 'Elapsed time = %4.2f seconds.  '%(time.time()-ts)+\
          'Takes 0.58 seconds on my AMD Sempron 3000'

#---------------
# Local Variables:
# eval: (python-mode)
# End:
