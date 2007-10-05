"""
mv1a.py Variant a on mv1.  The model is the same, but the classes and
algorithms are different.  First change is to use class called TARGET.

"""
import numpy, scipy, scipy.linalg, random, math

import util

class TARGET:
    """A TARGET is a possible moving target, eg, a car.  Its values
    are determined by initialzation values and a sequence of
    observations that are "Kalman filtered".  It contains the following:
    
    mu_t       Sequence of updated means
    Sigma_t    Sequence of updated covariances
    R_t        Sequence of residuals
    m_t        Sequence of observation indices
    children
    """
    def __init__(self,
                 A,        # Linear state dynamics
                 Sigma_D,  # Dynamical noise
                 O,        # Observation porjection
                 Sigma_O,  # Observational noise
                 m_t,      # History of hit indices used
                 mu_t,     # History of means
                 Sigma_t,  # History of variances
                 R_t       # History of residuals
                 ):
        self.A = A
        self.Sigma_D = Sigma_D
        self.O = O
        self.Sigma_O = Sigma_O
        self.m_t = m_t
        self.mu_t = mu_t
        self.Sigma_t = Sigma_t
        self.R_t = R_t
        self.children = None
        self.child_threshold = None

    def make_children(self,
                      y_t,   #list of hits at the next time
                      ):
        """ For each of the hits that could plausibly be an
        observation of self, make a child target.  Collect the
        children in a dict, attach it to self and return it.
        """
        if self.child_threshold is None: # make a child for each hit
            self.children = {}
            for k in xrange(len(y_t)):
                self.children[k] = self.KF(y_t[k],k)
        else:
            raise RuntimeError,'Call Planned Parenthood'
        
    def KF(self,
           y,        # The observation of the target at the current time
           m         # Index of the observation
           ):
        """ Create a new target with updated m_t, mu_t, Sigma_t and
        R_t for the observation, index pair (y,m).  This is
        essentially Kalman filtering."""

        # Unpack necessary matrices
        mu_t = self.mu_t[-1]
        Sig_t = self.Sigma_t[-1]
        R_t = self.R_t[-1]
        
        O = self.O
        Sig_O = self.Sigma_O
        Sig_D = self.Sigma_D
        A = self.A
        Id = scipy.matrix(scipy.identity(Sig_D.shape[0]))

        # Calculate intermediates
        Sig_a = A*Sig_t*A.T + Sig_D      # Covariance of state forecast
        mu_a = A*mu_t                    # Mean of state forecast
        Delta_y = y - O*mu_a             # Error of forecast observation
        Sig_y = O*Sig_a*O.T+self.Sigma_O # Covariance of forecast observation
        Sig_y_I = scipy.linalg.inv(Sig_y)
        K = Sig_a*O.T*Sig_y_I            # Kalman gain
        
        # Calculate new values and put them in lists
        m_L = self.m_t+[m]
        Sigma_L = self.Sigma_t + [(Id-K*O)*Sig_a]
        mu_L = self.mu_t + [mu_a + K*Delta_y]
        R_L = self.R_t + [R_t - float(Delta_y.T*Sig_y_I*Delta_y)/2]
        return TARGET(A,Sig_D,O,Sig_O,m_L,mu_L,Sigma_L,R_L)
    def backtrack(self):
        T = len(self.mu_t)
        A = self.A
        X = A.T * scipy.linalg.inv(self.Sigma_D) # An intermediate
        s_t = range(T)
        s_t[T-1] = self.mu_t[T-1]
        for t in xrange(T-2,-1,-1):
            Sig_t_I = scipy.linalg.inv(self.Sigma_t[t])
            mu_t = self.mu_t[t]
            s_t[t]=scipy.linalg.inv(Sig_t_I + X*A)*(Sig_t_I*mu_t + X*s_t[t+1])
        return s_t
        
class PERMUTATION:
    """A representation of a particular association of hits to targets
    at a particular time.  Attributes:

       targets:             A list of targets
       predecessor_perm:    A list of predecessor permutations
       predecessor_u_prime: A list of u' values for the predecessors
       nu:                  Utility of best path ending here
       key:                 A permutation tuple that maps hits t targets

    Methods:

     forward: Create plausible sucessor permutations

     argmax:  Select best predecessor, evaluate self.nu, collect list
              of appropriate child targets from that predecessor,
              attach the list to self, and return the list

     make_children:

       
    """

    def __init__(self,
                 N_tar,
                 key,
                 targets=None    # A list of targets
                 ):
        self.N_tar = N_tar
        self.key = key
        self.targets = targets
        self.predecessor_perm = []
        self.predecessor_u_prime = []
        
    def forward(self,
                new_perms   # A dict of permutations for the next time step
                ):
        """
        """
        # Create a list of successor permutations to consider
        old_list = []
        for child in self.targets[0].children.values():
            old_list.append({'perm':[child.m_t[-1]],'R':child.R_t[-1]})
        for k in xrange(1,len(self.targets)):
            new_list = []
            for child in self.targets[k].children.values():
                last_m_t = child.m_t[-1]
                for partial in old_list:
                    try: # Kludge to make sure each entry in a perm is unique
                        i = partial['perm'].index(last_m_t)
                        continue
                    except ValueError:
                        pass
                    new_perm = partial['perm']+[last_m_t]
                    new_R = partial['R']+child.R_t[-1]
                    new_list.append({'perm':new_perm,'R':new_R})
            old_list = new_list
        # old_list[i]['perm'] is a permutation where
        # y[t][old_list[i]['perm'][j]] is associated with target[j]

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
        appropriate child targets from that predecessor and attach
        that list to self
        """
        k_max = util.argmax(self.predecessor_u_prime)
        self.nu = self.predecessor_u_prime[k_max]
        best = self.predecessor_perm[k_max]
        self.targets = []
        for k in xrange(self.N_tar):
            self.targets.append(best.targets[k].children[self.key[k]])
    def make_children(self,y_t):
        for target in self.targets:
            target.make_children(y_t)
class MV1a:
    """A simple model of observed motion with the following groups of
    methods:

    Tools for applications:
      decode()
      simulate()

    Service methods:
      __init__()
    """
    def __init__(self,
                 N_tar = 3,                      # Number of targets
                 A = [[0.81,1],[0,.81]],         # Linear state dynamics
                 Sigma_D = [[0.01,0],[0,0.4]],   # Dynamical noise
                 O = [[1,0]],                    # Observation porjection
                 Sigma_O = [[0.25]],             # Observational noise
                 Sigma_init = [[25.0,0],[0,1.0]],# Initial state distribution
                 mu_init = None                  # Initial state distribution
                 ):
        self.N_tar = N_tar
        self.A = scipy.matrix(A)
        self.Sigma_D = scipy.matrix(Sigma_D)
        self.O = scipy.matrix(O)
        self.Sigma_O = scipy.matrix(Sigma_O)
        self.N_perm = int(scipy.factorial(N_tar))
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
        target_0 = TARGET(self.A,self.Sigma_D,self.O,self.Sigma_O,[],
                          [self.mu_init],[self.Sigma_init],[0.0])
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
            
            # For each old target, collect all plausibly associated
            # hits and create a corresponding child target by Kalman filtering
            for perm in old_perms.values():
                perm.make_children(Ys[t])
                
            # Build u' lists for all possible successor permutations at time t
            new_perms = {} # Use dict so many predecessors can find
                           # same successor
            for perm in old_perms.values():
                perm.forward(new_perms)

            # For each permutation at time t, find th best predecessor
            # and the associated targets
            for perm in new_perms.values():
                perm.argmax()

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
    print 'elapsed time=',time.time()-ts

#---------------
# Local Variables:
# eval: (python-mode)
# End:
