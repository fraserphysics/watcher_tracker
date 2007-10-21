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
        self.children = None # Dict of targets at t+1 updated with
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
        Delta_R = R0-float(Delta_y.T*self.Sigma_y_forecast_I*Delta_y
                           +self.mod.log_det_Sig_O)/2
        # Term log_det_Sig_O makes total nu match old mv2.py but not
        # match for mv1a.py.
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
        """Add a child for invisible y to children from
        TARGET.make_children()."""
        TARGET.make_children(self,y_t,All_children)
        key = tuple(self.m_t + [-1])
        if not All_children.has_key(key):
            All_children[key] = self.update(None,-1)
        self.children[-1] = All_children[key]  # Child for invisible y

    def utility(self,y):
        """Include log_prob factors for Sig_D and visibility transitions.
        """
        if self.m_t[-1] is -1:
            v_old = 1
        else:
            v_old = 0 # Last time target was visible
        if y is None:
            v_new = 1 # This time the target is invisible
        else:
            v_new = 0
        Delta_R = math.log(self.mod.PV_V[v_old,v_new])-self.mod.log_det_Sig_D/2
        if y is None:
            Sigma_new = self.Sigma_a
            mu_new = self.mu_a
            return (Delta_R,mu_new,Sigma_new)
        return TARGET.utility(self,y,R0=Delta_R)

class CAUSE:
    def __init__(self,
                 target  # Child target that caused y
                 ):
        self.type = 'target'
        self.index = target.index # Time invariant index of target
        self.target= target
        self.R = target.R_t
class CAUSE_FA(CAUSE):
    def __init__(self,
                 y,        # To calculate R for FA
                 Sigma_I   # To calculate R for FA
                 ):
        self.type = 'FA'
        self.index = -1
        self.R = -float(y.T*Sigma_I*y/2)
        #self.R = -10.0
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
    def __init__(self,nu,mod):
        self.nu = nu        # Utility of best path ending here
        self.mod = mod      # Model, to convey parameters
        self.tar_dict = {}  # Dict of targets
        self.targets = []   # List of child targets
        self.h2c = []       # Map from hits to causes.  Will become key
        self.extra_forward = [] # Stub for extra methods to use in forward
        self.cause_checks = [self.check_targets]
        self.type='ASSOCIATION'
    def New(self, *args,**kwargs):
        return ASSOCIATION(*args,**kwargs)
    def Fork(self, # Create a child that extends self by cause
            cause,  # CAUSE of next hit in y_t
            ):
        CA = self.New(self.nu + cause.R,self.mod)
        CA.tar_dict = self.tar_dict.copy()
        CA.tar_dict[cause.index] = None # Prevent reuse of target
        CA.targets = self.targets + [cause.target]
        CA.h2c = self.h2c + [cause.index]
        return CA
    def check_targets(self,k,causes,y):
        for target in self.targets:
            if target.children.has_key(k):
                # If has_key(k) then child is plausible cause
                causes.append(CAUSE(target.children[k]))    
    def dump(self):
        print 'dumping an association of type',self.type,':'
        print '  nu=%f, len(targets)=%d'%(self.nu,len(self.targets)),
        print 'hits -> causes map=', self.h2c
        for target in self.targets:
            target.dump()
        print '\n'
    
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
            for check in self.cause_checks:
                check(k,causes[k],y_t[k])
        # Initialize list of partial associations.
        old_list = [self.New(self.nu,self.mod)]
        # Make list of plausible associations.  At level k, each
        # association explains the source of each y_t[i] for i<k.
        for k in xrange(len(y_t)):
            new_list = []
            for partial in old_list:
                for cause in causes[k]:
                    # Don't use same target index more than once
                    # tar_dict has only targets, not FAs
                    if not partial.tar_dict.has_key(cause.index):
                        new_list.append(partial.Fork(cause))
            old_list = new_list
        # For each association, do extra work work required for
        # nonstandard hit/target combinations
        for partial in old_list:
            for method in self.extra_forward:
                method(partial,y_t)
        # Now each entry in old_list is a complete association and
        # entry.h2c[k] is a plausible CAUSE for hit y_t[k]

        # Map each association in old_list to it's successor key and
        # propose the present association as the predecessor for that
        # key.
        for association in old_list:
            successors.enter(association)
class ASSOCIATION2(ASSOCIATION):
    """See ASSOCIATION above.  This differs by adding extra_forward
    method invisibles() that allows and accounts for targets that move
    from an association at time t-1 to an association at time t
    without being visible at time t.
    """
    def __init__(self,*args,**kwargs):
        ASSOCIATION.__init__(self,*args,**kwargs)
        # Extra method to use in forward
        self.extra_forward = [self.invisibles]
        self.type='ASSOCIATION2'
    def New(self, *args,**kwargs):
        return ASSOCIATION2(*args,**kwargs)
    def invisibles(self,partial,y):
        for target in self.targets:
            if not partial.tar_dict.has_key(target.index):
                try:
                    child = target.children[-1]
                except:
                    raise RuntimeError,\
        'There is no invisible child.  Perhaps MaxD is too ambitious.'
                partial.targets.append(child)
                partial.nu += child.R_t

class ASSOCIATION3(ASSOCIATION2):
    """ Add the possibility of false alarms.
    """
    def __init__(self,*args,**kwargs):
        ASSOCIATION2.__init__(self,*args,**kwargs)
        self.cause_checks.append(self.check_FAs)
        self.extra_forward.append(self.count_FAs)
        self.type='ASSOCIATION3'
        self.N_FA=0              # Count of false alarms
    def New(self, *args,**kwargs):
        return ASSOCIATION3(*args,**kwargs)
    def Fork(self, # Create a child that extends self by cause
            cause,  # CAUSE of next hit in y_t
            ):
        CA = self.New(self.nu + cause.R,self.mod)
        CA.tar_dict = self.tar_dict.copy()
        CA.h2c = self.h2c + [cause.index]
        CA.N_FA = self.N_FA
        if cause.type is 'target':
            CA.tar_dict[cause.index] = None # Prevent reuse of target
            CA.targets = self.targets + [cause.target]
            return CA
        if cause.type is 'FA':
            CA.targets = self.targets
            CA.N_FA += 1
            return CA
        raise RuntimeError,"Cause type %s not known"%cause.type
    def check_FAs(self,k,causes,y):
        causes.append(CAUSE_FA(y,self.mod.Sigma_FA_I))
    def count_FAs(self,partial,y_t):
        N_FA = partial.N_FA
        if N_FA > 0:
            partial.nu += N_FA*self.mod.log_FA_norm + math.log(
            self.mod.Lambda**N_FA/scipy.factorial(N_FA) )
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
        self.log_det_Sig_D = scipy.linalg.det(self.Sigma_D)
        self.log_det_Sig_O = scipy.linalg.det(self.Sigma_O)
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
        partial = DASSOCIATION(0.0,self)
        for k in xrange(self.N_tar):
            target_k = target_0.KF(Ys[0][k],k)
            partial = partial.Fork(CAUSE(target_k))
        
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
"""%(t,len(old_As), len(child_targets),len(Ys[t]),self.MaxD,
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
    def decode(self,
               Ys, # Observations. Ys[t][k] is the kth hit at time t
               DTARGET=TARGET2,
               DASSOCIATION=ASSOCIATION2
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
class MV3(MV2):
    """ A state consists of: Association; Locations; and Visibilities;
    (and the derivable N_FA), ie,

    s = (M,X,V,N_FA)
    
    """
    def __init__(self,Lambda=0.3,**kwargs):
        MV2.__init__(self,**kwargs)
        self.Lambda = Lambda # Average number of false alarms per frame
        Sigma_FA = self.O*self.Sigma_init*self.O.T + self.Sigma_O
        self.Sigma_FA = Sigma_FA
        self.log_FA_norm = - math.log(scipy.linalg.det(Sigma_FA))/2
        self.Sigma_FA_I = scipy.linalg.inv(Sigma_FA)
    
    def decode(self,
               Ys, # Observations. Ys[t][k] is the kth hit at time t
               DASSOCIATION=ASSOCIATION3
               ):
        return MV2.decode(self,Ys,DASSOCIATION=DASSOCIATION)
    
    def simulate(self, T):
        """ Return a sequence of T observations and a sequence of T
        states."""
        #Start with N_tar visible targets
        x_j = []
        v_j = []
        for j in xrange(self.N_tar):
            x_j.append(util.normalS(self.mu_init,self.Sigma_init))
            v_j.append(0)
        i_t = 0   # First association: No shuffle
        N_FA = 0  # No false alarms for t=0

        # Set up useful parameters
        x_dim = self.mu_init.shape[1]
        zero_x = scipy.matrix(scipy.zeros(x_dim))
        y_dim = self.Sigma_O.shape[0]
        zero_y = scipy.matrix(scipy.zeros(y_dim))

        # Lists for results
        obs = []
        xs = []
        vs = []
        for t in xrange(T):
            xs.append(x_j)        # Save X[t] part of state for return
            # Generate observations Y[t]
            obs_t = []
            permute = util.index_to_permute(i_t,self.N_tar)
            for k in xrange(N_FA):
                obs_t.append(util.normalS(zero_y,self.Sigma_FA))
            for j in xrange(self.N_tar):
                if v_j[j] is not 0:
                    continue
                eta = util.normalS(zero_y,self.Sigma_O) # Observational noise
                obs_t.append(self.O * x_j[permute[j]] + eta)
            obs.append(obs_t)     # Save Y[t] for return
            # Generate next state
            i_t = random.randint(0,self.N_perm-1)
            x_j_new = self.N_tar*[None]
            for j in xrange(self.N_tar):
                # Make a new x for each target
                epsilon = util.normalS(zero_x,self.Sigma_D) # Dynamical noise
                x_j_new[j] = self.A*x_j[j] + epsilon
                # Make a new visibility for each target
                pv = self.PV_V[v_j[j],0]
                if pv > random.random() or t is 0:
                    v_j[j] = 0
                else:
                    v_j[j] = 1
            x_j = x_j_new
            # Select N_FA for the next t
            N_FA = scipy.random.poisson(self.Lambda)
        return obs,xs

# Test code
if __name__ == '__main__':
    import time
    random.seed(3)
    ts = time.time()
    for M in (
        MV3(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),
        MV1(N_tar=4),
        MV2(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]])
        ):

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
