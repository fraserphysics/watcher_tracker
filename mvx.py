"""
FILE: mvx.py  All model variations in a single file.  I started this
after writing mv3.py which imported mv2.py and mv1a.py.  I wanted more
coherence before starting on mv4 which allows a variable number of
targets.

mv1a.py used a map from targets to observations.  I want to use a map
from observations to causes for mv4 and I modified mv3 to do that.
This file will use maps from observations to causes for all model
versions.
"""
import numpy, scipy, scipy.linalg, random, math, util

Target_Counter = 0
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
                 R_t,      # Most recent residual
                 index=None
                 ):
        if isinstance(index,type(0)): # Check for int
            self.index = index
        else:
            global Target_Counter
            self.index = Target_Counter
            Target_Counter += 1
        self.mod = mod
        self.m_t = m_t
        self.mu_t = mu_t
        self.Sigma_t = Sigma_t
        self.R_t = R_t
        self.children = None # List of targets at t+1 updated with
                             # plausible hits and this target
    def New(self, *args,**kwargs):
        return TARGET(*args,**kwargs)
    def dump(self):
        print 'Dump target: m_t=',self.m_t
        print '             index=%d, len(mu_t)=%d'%(self.index,len(self.mu_t))
        if self.children is not None:
            print'  len(self.children)=%d\n'%len(self.children), self.children
    def make_children(self,        # self is a TARGET
                      y_t,         # list of hits at time t
                      All_children # Dict of children of all permutations
                      ):
        """ For each of the hits that could plausibly be an
        observation of self, make a child target.  Collect the
        children in a dict and attach it to self.
        """
        self.forecast()
        self.children = {}
        MD = self.mod.MaxD
        for k in xrange(len(y_t)):
            if MD < 0.01 or self.distance(y_t[k]) < MD:
                # If MaxD is near zero, ie, pruning is off or hit is
                # closer than MaxD
                key = tuple(self.m_t+[k])
                if not All_children.has_key(key):
                    All_children[key] = self.update(y_t[k],k)
                self.children[k] = All_children[key]
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
        """ Rather than the Malhalanobis distance of y from forecast
        y.  Here I generalize to sqrt(-2 Delta_R)
        """
        return float(-2*self.utility(y)[0])**.5
    def update(self,
           y,        # The observation of the target at the current time
           m         # Index of the observation
           ):
        """ Create a new target with updated m_t, mu_t, Sigma_t and R_t
        for the observation, index pair (y,m)."""
        m_L = self.m_t+[m]
        Delta_R,mu_new,Sigma_new = self.utility(y)
        Sigma_L = self.Sigma_t + [Sigma_new]
        mu_L = self.mu_t + [mu_new]
        return self.New(self.mod,m_L,mu_L,Sigma_L,Delta_R,
                             index=self.index)
    def utility(self,y,R0=0.0):
        """ Calculates Delta_R, mu_new, and Sigma_new for both
        update() and distance().  This is the second half of Kalman
        filtering step.
        """
        Delta_y = y - self.y_forecast    # Error of forecast observation
        Sigma_new = self.Sigma_next
        mu_new = self.mu_a + self.K*Delta_y
        Delta_R = R0-float(Delta_y.T*self.Sigma_y_forecast_I*Delta_y)/2
        return (Delta_R,mu_new,Sigma_new)
    def KF(self,
           y,        # The observation of the target at the current time
           m         # Index of the observation
           ):
        """ Create a new target with unique index, fresh m_t, mu_t,
        Sigma_t and R_t for the observation.  This is essentially
        Kalman filtering."""
        self.forecast()
        r = self.update(y,m)
        return self.New(self.mod,[r.m_t[-1]],[r.mu_t[-1]],
                             [r.Sigma_t[-1]],r.R_t)
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

class TARGET2(TARGET):
    def New(self, *args,**kwargs):
        return TARGET2(*args,**kwargs)
    def make_children(self,y_t,All_children):
        """  Include invisibility.
        """
        TARGET.make_children(self,y_t,All_children)
        key = tuple(self.m_t + [-1])
        if not All_children.has_key(key):
            All_children[key] = self.update(None,-1)
        self.children[-1] = All_children[key]  # Child for invisible y

    def utility(self,y):
        """ Calculates Delta_R, mu_new, and Sigma_new for both
        update() and distance().  This is the second half of Kalman
        filtering step.
        """
        if self.m_t[-1] is -1:
            v_old = 1
        else:
            v_old = 0 # Last time target was visible
        if y is None:
            v_new = 1 # This time the target is invisible
        else:
            v_new = 0
        Delta_R = math.log(self.mod.PV_V[v_old,v_new])
        if y is None:
            Sigma_new = self.Sigma_a
            mu_new = self.mu_a
            Delta_R -= self.mod.log_det_Sig_D/2
            print 'TARGET2.utility(): v_old=%d, v_new=%d, Delta_R=%f'%(v_old,v_new,Delta_R)
            return (Delta_R,mu_new,Sigma_new)
        return TARGET.utility(self,y,R0=Delta_R)

class CAUSE:
    def __init__(self,
                 target=None  # Child target that caused y
                 ):
        self.index = target.index # Time invariant index of target
        self.target= target
        self.R = target.R_t
class SUCCESSOR_DB:
    """
    """
    def __init__(self):
        self.successors = {}
    def enter(self,       
              association  # A candidate history
              ):
        key = tuple(association.h2c) # An explanation of each component of y_t
        u_prime = association.nu     # The utility of the candidate
        if not self.successors.has_key(key):
            self.successors[key] = {'associations':[],'u_primes':[]}
        self.successors[key]['associations'].append(association)
        self.successors[key]['u_primes'].append(u_prime)
    def length(self):
        return len(self.successors)
    def maxes(self):
        """ Return the associations with the largest corresponding u_primes
        """
        rv = []
        for suc in self.successors.values():
            rv.append(suc['associations'][util.argmax(suc['u_primes'])])
        return rv
class ASSOCIATION:
    """This is the discrete part of the state.  It gives an
    explanation for how the collection of observations at time t were
    made.

    Methods:

     __init__: Called with None or with an existing association and a
               cause to create a new association with an explanation
               for an additional observation.
     
     forward:  Create plausible sucessor associations

     make_children:  Call target.make_children() for each target
    """
    def __init__(self,
                 parent=None, # A partial association
                 cause=None,  # CAUSE of next hit in y_t
                 nu=None      # nu from parent association/permutation
                 ):
        if parent is None and cause is None:
            self.par_tars = {}  # Dict of parent targets
            self.targets = []   # List of child targets
            self.h2c = []       # Map from hits to causes.  Will become key
            self.nu = nu        # Utility of best path ending here
            return
        if not (parent and cause):
            raise RuntimeError,\
                  "parent and cause must both be None or both be defined"
        self.par_tars = parent.par_tars.copy()
        self.par_tars[cause.index] = None # Prevent reuse of target
        self.targets = parent.targets + [cause.target]
        self.h2c = parent.h2c + [cause.index]
        self.nu = parent.nu + cause.R
    
    def dump(self):
        print 'dumping a association. N_tar=%d, nu=%d, len(targets)=%d'%\
              (self.N_tar,self.nu,len(self.targets))
        print 'hits -> targets map:', self.key
        for target in self.targets:
            target.dump()
    
    def make_children(self, # self is a ASSOCIATION
                      y_t,  # All observations at time t
                      cousins
                      ):
        for target in self.targets:
            target.make_children(y_t,cousins)
                
    def forward(self,
                successors,   # A DB of associations for the next time step
                y_t
                ):
        """ For each plausible successor S of the ASSOCIATION self
        enter the following into the successors DB 1. A key for the
        explanatinon that S gives for the observations. 2. The
        candidate successor association S. 3. The value of
        u'(self,S,t+1).
        """
        # For each observation, make a list of plausible causes
        causes = []
        for k in xrange(len(y_t)):
            causes.append([])
            for target in self.targets:
                if target.children.has_key(k):
                    # If has_key(k) then child is plausible cause
                    causes[k].append(CAUSE(target.children[k]))
        # Initialize list of partial associations.
        old_list = [ASSOCIATION(nu=self.nu)]
        # Make list of plausible associations.  At level k, each
        # association explains the source of each y_t[i] for i<k.
        for k in xrange(len(y_t)):
            new_list = []
            for partial in old_list:
                for cause in causes[k]:
                    # Don't use same target index more than once
                    if not partial.par_tars.has_key(cause.index):
                        new_list.append(ASSOCIATION(partial,cause))
            old_list = new_list

        # Now each entry in old_list is a complete association and
        # entry.h2c[k] is a plausible CAUSE for hit y_t[k]

        # Map each association in old_list to it's successor key and
        # propose the present association as the predecessor for that
        # key.
        for association in old_list:
            successors.enter(association)
        
class ASSOCIATION3(ASSOCIATION):
    """
    New Methods:

     forward: Create plausible sucessor associations
    """

    def forward(self,
                new_perms,   # A dict of associations for the next time step
                y_t          # A list of hits at the next time step
                ):
        """ For each plausible successor S of the ASSOCIATION self,
        append the following pair of values to S.predecessor: 1. A
        pointer back to self and 2. The value of u'(self,S,t+1).
        """
        # Fetch values for calculating the utility of false alarms
        Sigma_FA_I=self.targets[0].mod.Sigma_FA_I
        norm = self.targets[0].mod.log_FA_norm
        Lambda = self.targets[0].mod.Lambda
        
        # For each observation, list the plausible causes
        causes = []
        for k in xrange(len(y_t)):
            # False alarm is plausible
            causes.append([CAUSE('FA',Sigma_I=Sigma_FA_I,y=y_t[k])])
            for i in xrange(len(self.targets)):
                if self.targets[i].children.has_key(k):
                    # If has_key(k) then child is plausible cause
                    child = self.targets[i].children[k]
                    causes[k].append(CAUSE('target',child,i))
        
        # Initialize list of partial associations.
        old_list = [PARTIAL(nu=self.nu)]
        # Make list of plausible associations.  At level k, each
        # association explains the source of each y_t[i] for i<k.
        for k in xrange(len(y_t)):
            new_list = []
            for partial in old_list:
                for cause in causes[k]:
                    # Don't use same self.targets[index] more than once
                    if not partial.dup_check.has_key(cause.index):
                        new_list.append(PARTIAL(partial,cause))
            old_list = new_list

        # For each plausible association, account for the invisible
        # targets and the number of false alarms
        for entry in old_list:
            N_FA = len(y_t) - len(entry.dup_check)
            entry.u_prime += N_FA*norm + math.log(
                Lambda**N_FA/scipy.factorial(N_FA) )
            for k in xrange(len(self.targets)):
                if not entry.dup_check.has_key(k):
                    entry.invisible(self.targets[k])

        # Now each entry in old_list is a PARTIAL and entry.h2c[k]
        # is a plausible CAUSE for hit y_t[k]

        # Initialize successors if necessary and set their predecessors
        for entry in old_list:
            key = tuple(entry.h2c)  # Key for successor
            if not new_perms.has_key(key):
                new_perms[key] = ASSOCIATION(self.N_tar,key)
            successor = new_perms[key]
            successor.predecessor_perm.append(self)
            successor.predecessor_u_prime.append(entry.u_prime)
class MV1:
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
               Ys, # Observations. Ys[t][k] is the kth hit at time t
               DASSOCIATION=ASSOCIATION,
               DTARGET=TARGET
               ):
        """Return MAP state sequence """
        T = len(Ys)

        # Initialize by making a target and cause for each of the hits
        # at t=0 and collecting them in a single association.
        target_0 = DTARGET(self,[0],[self.mu_init],[self.Sigma_init],0.0)
        partial = ASSOCIATION(nu=0.0)
        for k in xrange(self.N_tar):
            target_k = target_0.KF(Ys[0][k],k)
            partial = ASSOCIATION(partial,CAUSE(target_k))
        
        # Forward pass through time
        old_As = [partial]
        for t in xrange(1,T):
            child_targets = {}          # Dict of all targets for time t
            successors = SUCCESSOR_DB() # Just to make the while loop start
            while successors.length() is 0:
                # For each old target, collect all plausibly
                # associated hits and create a corresponding child
                # target by Kalman filtering
                for A in old_As:
                    A.make_children(Ys[t],child_targets)
                # Build u' lists for all possible successor
                # associations at time t.  Put new_As in DB so many
                # predecessors can find same successor
                successors = SUCCESSOR_DB()
                for A in old_As:
                    A.forward(successors,Ys[t])
                self.MaxD *= 2
                if successors.length() is 0 and \
                       (self.MaxD<1e-6 or self.MaxD>1e6):
                    raise RuntimeError,"""
No new associations in decode():
t=%d, len(old_As)=%d, len(child_targets)=%d, len(Ys[t])=%d,MaxD=%g
successors.length()=%d
"""%\
                (t,len(old_As), len(child_targets),len(Ys[t]),self.MaxD,
                 successors.length())
                # End of while
            self.MaxD = max(self.MaxD/4,self.MaxD_limit)
            # For each association at time t, find the best predecessor
            # and the associated targets and collect them in old_As
            new_As = successors.maxes()

            # Pass up to MaxP new_As to old_As
            if len(new_As) > self.MaxP:
                Rs = []
                for A in new_As:
                    Rs.append(A.nu)
                Rs.sort()
                limit = Rs[-self.MaxP]
                old_As = []
                for A in new_As:
                    if A.nu >= limit:
                        old_As.append(A)
            else:
                old_As = new_As
        # End of for loop over t
        
        # Find the best last association
        R = scipy.zeros(len(old_As))
        for i in xrange(len(old_As)):
            R[i] = old_As[i].nu
        A_best = old_As[R.argmax()]
        print 'nu_max=%f'%A_best.nu

        # Backtrack to get trajectories
        tracks = [] # tracks[k][t] is the x vector for target_k at time t
        y_A = []    # y_A[t] is a dict.  y_A[t][k] gives the index of
                    # y[t] associated with target k.  So y[t][A[t][k]]
                    # is associated with target k.
        for t in xrange(T):
            y_A.append({}) # Association dicts
        for k in xrange(len(A_best.targets)):
            target = A_best.targets[k]
            tracks.append(target.backtrack())
            for t in xrange(T):
                y_A[t][k] = target.m_t[t]
        return (tracks,y_A)
                
class MV2(MV1):
    def __init__(self,PV_V=[[.9,.1],[.2,.8]],**kwargs):
        MV1.__init__(self,**kwargs)
        self.PV_V = scipy.matrix(PV_V)
        self.log_det_Sig_D = scipy.linalg.det(self.Sigma_D)
        self.log_det_Sig_O = scipy.linalg.det(self.Sigma_O)
    def decode(self,
               Ys, # Observations. Ys[t][k] is the kth hit at time t
               DTARGET=TARGET2,
               DASSOCIATION=ASSOCIATION
               ):
        return MV1.decode(self,Ys,DTARGET=DTARGET, DASSOCIATION=DASSOCIATION)

    
    def simulate(self, T):
        """ Return a sequence of T observations and a sequence of T
        states."""
        x_j = []
        v_j = []
        for j in xrange(self.N_tar):
            x_j.append(util.normalS(self.mu_init,self.Sigma_init))
            v_j.append(0) # Start with all trajectories visible

        x_dim = self.mu_init.shape[1]
        zero_x = scipy.matrix(scipy.zeros(x_dim))
        y_dim = self.Sigma_O.shape[0]
        zero_y = scipy.matrix(scipy.zeros(y_dim))
        
        obs = []
        xs = []
        vs = []
        for t in xrange(T):
            x_j_new = range(self.N_tar)
            if t>0:
                i = random.randint(0,self.N_perm-1)
            else:
                i = 0 # No shuffle for t=0
            permute = util.index_to_permute(i,self.N_tar)
            state_t = []
            for j in xrange(self.N_tar):
                pv = self.PV_V[v_j[j],0]
                if pv > random.random() or t is 0:
                    v_j[j] = 0
                else:
                    v_j[j] = 1
                epsilon = util.normalS(zero_x,self.Sigma_D) # Dynamical noise
                x_j_new[j] = self.A*x_j[j] + epsilon
            obs_t = []
            for j in xrange(self.N_tar):
                if v_j[j] is not 0:
                    continue
                eta = util.normalS(zero_y,self.Sigma_O) # Observational noise
                obs_t.append(self.O * x_j[permute[j]] + eta)
            obs.append(obs_t)
            xs.append(x_j_new)
            x_j = x_j_new
        return obs,xs
    
# Test code
if __name__ == '__main__':
    import time
    random.seed(3)
    ts = time.time()
    for M in (MV2(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),MV1(N_tar=4)):
    #for M in (MV1(N_tar=4),MV2(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]])):

        print 'before simulate'
        y,s = M.simulate(5)
        print 'before decode'
        d,tmp = M.decode(y)
        print 'len(y)=',len(y), 'len(s)=',len(s),'len(d)=',len(d)
        for t in xrange(len(s)):
            print 't=%d    y         s           d'%t
            for k in xrange(len(s[t])):
                try:
                    print ' k=%d  %4.2f  '%(k,y[t][k][0,0]),
                except:
                    print ' k=%d        '%k,
                for f in (s[t][k],d[k][t]):
                    print '(%4.2f, %4.2f)  '%(f[0,0],f[1,0]),
                print ' '
        
        print '\n'
    print 'Elapsed time = %4.2f seconds.  '%(time.time()-ts)+\
          'Takes 0.30 seconds on my AMD Sempron 3000'

#---------------
# Local Variables:
# eval: (python-mode)
# End:
