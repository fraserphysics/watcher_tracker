"""
FILE: mvx.py  All model variations in a single file.  I started this
after writing mv3.py which imported mv2.py and mv1a.py.  I wanted more
coherence before starting on mv4 which allows a variable number of
targets.

mv1a.py used a map from targets to observations.  I want to use a map
from observations to causes for mv4 and I modified mv3 to do that.
This file will use maps from observations to causes for all model
versions.

10/22/07: To date, I have made each successive class with more
features a sub class of a simpler class.  Ultimately, I think the code
will be easier to read if I make the subclasses simpler.

"""
import numpy, scipy, scipy.linalg, random, math, util
def log_Poisson(Lambda,N):
    return N*math.log(Lambda) - Lambda - math.log(scipy.factorial(N))
Invisible_Lifetime = 5
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
                 y,        # Useful in subclasses
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
        print '\nDump target: m_t=',self.m_t
        print '             index=%d, len(mu_t)=%d'%(self.index,len(self.mu_t))
        if self.children is not None:
            print'  len(self.children)=%d'%len(self.children), self.children
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
                self.children[k] = All_children.setdefault(key,
                                        self.update(y_t[k],k))
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
        return self.New(self.mod,m_L,mu_L,Sigma_L,Delta_R,y,
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
                             [r.Sigma_t[-1]],r.R_t,y)
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
    """ Variation on TARGET with visibility transitions.
    """
    def __init__(self,*args,**kwargs):
        TARGET.__init__(self,*args,**kwargs)
        self.invisible_count=0
    def New(self, *args,**kwargs):
        if args[-1] is None: # y is None, ie invisible
            rv = TARGET2(*args,**kwargs)
            rv.invisible_count = self.invisible_count + 1
            return rv
        else:
            return TARGET2(*args,**kwargs)
    def dump(self):
        TARGET.dump(self)
        print '   invisible_count=%d'%self.invisible_count
    def make_children(self,y_t,All_children):
        """Add a child for invisible y to children from
        TARGET.make_children()."""
        TARGET.make_children(self,y_t,All_children)
        key = tuple(self.m_t + [-1])
        self.children[-1] = All_children.setdefault(key,self.update(None,-1))
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
        suc_key = self.successors.setdefault(key,
                      {'associations':[],'u_primes':[]})
        suc_key['associations'].append(association)
        suc_key['u_primes'].append(u_prime)
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
        self.N_FA=0          # Count of false alarms for ASSOCIATION3
    def New(self, *args,**kwargs):
        return ASSOCIATION(*args,**kwargs)
    def Fork(self, # Create a child that extends self by cause
            cause  # CAUSE of next hit in y_t
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
    def check_targets(self,k,causes,y,target_0):
        """ A check method for list cause_checks called by forward().
        This method appends relevant children to plausible causes.
        """
        for target in self.targets:
            if target.children.has_key(k):
                # If has_key(k) then child is plausible cause
                causes.append(CAUSE(target.children[k]))
    def check_FAs(self,k,causes,y,target_0):
        """ A check in list of cause_checks called by forward().
        Propose that cause is false alarm
        """
        causes.append(CAUSE_FA(y,self.mod.Sigma_FA_I))
    def check_news(self,k,causes,y,target_0):
        """ A check in list of cause_checks called by forward().
        Propose that cause is false alarm or new target
        """
        causes.append(CAUSE_FA(y,self.mod.Sigma_FA_I)) # False alarm
        causes.append(CAUSE(target_0.KF(y,k)))  # New target
    def invisibles(self,partial):
        """ A method in extra_forward called by forward().  Put
        invisible targets in new association.
        """
        for target in self.targets:
            if not partial.tar_dict.has_key(target.index):
                try:
                    child = target.children[-1]
                except:
                    raise RuntimeError,\
        'There is no invisible child.  Perhaps MaxD is too ambitious.'
                partial.targets.append(child)
                partial.nu += child.R_t 
    def count_FAs(self,partial):
        """ A method in extra_forward called by forward().  Calculate
        the utility for the number of false alarms and add that
        utility to partial.nu.
        """
        N_FA = partial.N_FA
        if N_FA > 0:
            partial.nu += N_FA*self.mod.log_FA_norm + log_Poisson(
                self.mod.Lambda_FA,N_FA)
    def extra_news(self,partial):
        """ A method in extra_forward called by forward().  Put
        invisible targets in new association, and apply creation
        penalty to nu for newly created targets.
        """ 
        for target in self.targets:
            if not partial.tar_dict.has_key(target.index) and \
                   target.children.has_key(-1):
                child = target.children[-1]
                partial.targets.append(child)
                partial.nu += child.R_t
        N_new = 0
        for target in partial.targets:
            if len(target.m_t) is 1:
                N_new += 1
        if N_new > 0:
            partial.nu += N_new*self.mod.log_new_norm + log_Poisson(
                self.mod.Lambda_new,N_new)
    def dump(self):
        print 'dumping an association of type',self.type,':'
        print '  nu=%f, len(targets)=%d'%(self.nu,len(self.targets)),
        print 'hits -> causes map=', self.h2c
        for target in self.targets:
            target.dump()
        print '\n'
    
    def make_children(self,    # self is a ASSOCIATION
                      y_t,     # All observations at time t
                      cousins, # Dict of children of sibling ASSOCIATIONs
                      t        # Used by ASSOCIATION4
                      ):
        for target in self.targets:
            target.make_children(y_t,cousins)
    def forward(self,
                successors,   # A DB of associations for the next time step
                y_t,
                target_0      # Useful for subclasses
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
                check(k,causes[k],y_t[k],target_0)
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
                method(partial)
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
class ASSOCIATION3(ASSOCIATION):
    """ Add the possibility of false alarms.
    """
    def __init__(self,*args,**kwargs):
        ASSOCIATION.__init__(self,*args,**kwargs)
        self.cause_checks = [self.check_targets,self.check_FAs]
        self.extra_forward = [self.invisibles,self.count_FAs]
        self.type='ASSOCIATION3'
    def New(self, *args,**kwargs):
        return ASSOCIATION3(*args,**kwargs)
class ASSOCIATION4:
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
        self.cause_checks = [self.check_targets,self.check_news]
        self.extra_forward = [self.extra_news,self.count_FAs]
        self.dead_targets = {}
        self.type='ASSOCIATION4'
        self.N_FA=0          # Count of false alarms
    def New(self, *args,**kwargs):
        NA = ASSOCIATION4(*args,**kwargs)
        NA.dead_targets = self.dead_targets.copy()
        return NA
    def Fork(self, # Create a child that extends self by cause
            cause  # CAUSE of next hit in y_t
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
    def check_targets(self,k,causes,y,target_0):
        """ A check method for list cause_checks called by forward().
        This method appends relevant children to plausible causes.
        """
        for target in self.targets:
            if target.children.has_key(k):
                # If has_key(k) then child is plausible cause
                causes.append(CAUSE(target.children[k]))
    def check_FAs(self,k,causes,y,target_0):
        """ A check in list of cause_checks called by forward().
        Propose that cause is false alarm
        """
        causes.append(CAUSE_FA(y,self.mod.Sigma_FA_I))
    def check_news(self,k,causes,y,target_0):
        """ A check in list of cause_checks called by forward().
        Propose that cause is false alarm or new target
        """
        causes.append(CAUSE_FA(y,self.mod.Sigma_FA_I)) # False alarm
        causes.append(CAUSE(target_0.KF(y,k)))  # New target
    def invisibles(self,partial):
        """ A method in extra_forward called by forward().  Put
        invisible targets in new association.
        """
        for target in self.targets:
            if not partial.tar_dict.has_key(target.index):
                try:
                    child = target.children[-1]
                except:
                    raise RuntimeError,\
        'There is no invisible child.  Perhaps MaxD is too ambitious.'
                partial.targets.append(child)
                partial.nu += child.R_t 
    def count_FAs(self,partial):
        """ A method in extra_forward called by forward().  Calculate
        the utility for the number of false alarms and add that
        utility to partial.nu.
        """
        N_FA = partial.N_FA
        if N_FA > 0:
            partial.nu += N_FA*self.mod.log_FA_norm + log_Poisson(
                self.mod.Lambda_FA,N_FA)
    def extra_news(self,partial):
        """ A method in extra_forward called by forward().  Put
        invisible targets in new association, and apply creation
        penalty to nu for newly created targets.
        """ 
        for target in self.targets:
            if not partial.tar_dict.has_key(target.index) and \
                   target.children.has_key(-1):
                child = target.children[-1]
                partial.targets.append(child)
                partial.nu += child.R_t
        N_new = 0
        for target in partial.targets:
            if len(target.m_t) is 1:
                N_new += 1
        if N_new > 0:
            partial.nu += N_new*self.mod.log_new_norm + log_Poisson(
                self.mod.Lambda_new,N_new)
    def dump(self):
        print 'dumping an association of type',self.type,':'
        print '  nu=%f, len(targets)=%d'%(self.nu,len(self.targets)),
        print 'hits -> causes map=', self.h2c
        for target in self.targets:
            target.dump()
        print '\n'
    def make_children(self,    # self is a ASSOCIATION4
                      y_t,     # All observations at time t
                      cousins, # Dict of children of sibling ASSOCIATIONs
                      t        # Save time too
                      ):
        for target in self.targets:
            if target.invisible_count < Invisible_Lifetime:
                target.make_children(y_t,cousins)
            else:
                self.dead_targets.setdefault(target.index,[target,t])
                target.children = {}
    def forward(self,
                successors,   # A DB of associations for the next time step
                y_t,
                target_0      # Useful for subclasses
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
                check(k,causes[k],y_t[k],target_0)
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
                method(partial)
        # Now each entry in old_list is a complete association and
        # entry.h2c[k] is a plausible CAUSE for hit y_t[k]

        # Map each association in old_list to it's successor key and
        # propose the present association as the predecessor for that
        # key.
        for association in old_list:
            successors.enter(association)
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
        self.ASSOCIATION = ASSOCIATION
        self.TARGET = TARGET
        self.N_tar = N_tar
        self.A = scipy.matrix(A)
        dim,check = self.A.shape
        if dim != check:
            raise RuntimeError,\
         "A should be square but it's dimensions are (%d,%d)"%(dim,check)
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
    ############## Begin simulation methods ######################
    def step_state(self, state, zero_s):
        epsilon = util.normalS(zero_s,self.Sigma_D) # Dynamical noise
        return self.A*state + epsilon
    def observe_states(self, states_t, v_t, zero_y):
        v_states = []
        for k in xrange(len(v_t)):
            if v_t[k] is 0:
                v_states.append(states_t[k])
        y_t = []
        for state in v_states:
            eta = util.normalS(zero_y,self.Sigma_O) # Observational noise
            y_t.append(self.O * state + eta)
        return y_t
    def shuffle(self, things_i):
        N = len(things_i)
        if N is 0:
            return []
        things_o = N*[None]
        permute = util.index_to_permute(random.randint(0,N-1),N)
        for k in xrange(N):
            things_o[k] = things_i[permute[k]]
        return things_o
    def sim_zeros(self):
        """ Make two useful zero matrices """
        s_dim = self.mu_init.shape[1]
        zero_s = scipy.matrix(scipy.zeros(s_dim))
        y_dim = self.Sigma_O.shape[0]
        zero_y = scipy.matrix(scipy.zeros(y_dim))
        return (zero_s,zero_y)
    def step_vis(self,v):
        pv = self.PV_V[v,0]
        if pv > random.random():
            return 0
        else:
            return 1
    def sim_init(self,N):
        x_j = []
        v_j = []
        for j in xrange(N):
            x_j.append(util.normalS(self.mu_init,self.Sigma_init))
            v_j.append(0)
        return (x_j,v_j,0)
    def simulate(self, T):
        """ Return a sequence of T observations and a sequence of T
        states."""
        s_j,v_j,N_FA = self.sim_init(self.N_tar)
        zero_s,zero_y = self.sim_zeros()       
        obs = []
        states = []
        for t in xrange(T):
            states.append(s_j)
            obs.append(self.shuffle(self.observe_states(s_j,v_j,zero_y)))
            s_j = map(lambda x: self.step_state(x,zero_s),s_j)
        return obs,states
    ############## End simulation methods ######################

    def decode_forward(self,
               Ys # Observations. Ys[t][k] is the kth hit at time t
               ):
        """Forward pass for decoding.  Return association at final
        time with highest utility, nu."""
        T = len(Ys)

        # Initialize by making a target and cause for each of the hits
        # at t=0 and collecting them in a single association.
        target_0 = self.TARGET(self,[0],[self.mu_init],[self.Sigma_init],
                               0.0,None)
        partial = self.ASSOCIATION(0.0,self)
        
        # Forward pass through time
        old_As = [partial]
        for t in xrange(0,T):
            child_targets = {}          # Dict of all targets for time t
            successors = SUCCESSOR_DB() # Just to make the while loop start
            while successors.length() is 0:
                # For each old target, collect all plausibly
                # associated hits and create a corresponding child
                # target by Kalman filtering
                for A in old_As:
                    A.make_children(Ys[t],child_targets,t)
                # Build u' lists for all possible successor
                # associations at time t.  Put new_As in DB so many
                # predecessors can find same successor
                successors = SUCCESSOR_DB()
                for A in old_As:
                    A.forward(successors,Ys[t],target_0)
                self.MaxD *= 2
                #### Begin debugging check and print ####
                if successors.length() is 0 and \
                       (self.MaxD<1e-6 or self.MaxD>1e6):
                    raise RuntimeError,"""
No new associations in decode():
t=%d, len(old_As)=%d, len(child_targets)=%d, len(Ys[t])=%d,MaxD=%g
successors.length()=%d
"""%(t,len(old_As), len(child_targets),len(Ys[t]),self.MaxD,
     successors.length())
                #### End debugging check and print ####
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
        return (A_best,T)
    def decode_back(self,A_best,T):
        """ Backtrack from best association at final time to get MAP
        trajectories.
        """
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

    def decode(self,
               Ys # Observations. Ys[t][k] is the kth hit at time t
               ):
        """Return MAP state sequence """
        A_best,T = self.decode_forward(Ys)
        return self.decode_back(A_best,T)
    
class MV2(MV1):
    def __init__(self,PV_V=[[.9,.1],[.2,.8]],**kwargs):
        MV1.__init__(self,**kwargs)
        self.ASSOCIATION = ASSOCIATION2
        self.TARGET = TARGET2
        self.PV_V = scipy.matrix(PV_V)

    
    ############## Begin simulation methods ######################       
    def simulate(self, T):
        """ Return a sequence of T observations and a sequence of T
        states."""
        x_j,v_j,N_FA = self.sim_init(self.N_tar)
        zero_x,zero_y = self.sim_zeros()
        obs = []
        xs = []
        for t in xrange(T):
            xs.append(x_j)
            obs.append(self.shuffle(self.observe_states(x_j,v_j,zero_y)))
            x_j = map(lambda x: self.step_state(x,zero_x),x_j)
            v_j = map(self.step_vis,v_j)
        return obs,xs
    ############## End simulation methods ######################
class MV3(MV2):
    """ A state consists of: Association; Locations; and Visibilities;
    (and the derivable N_FA), ie,

    s = (M,X,V,N_FA)
    
    """
    def __init__(self,Lambda_FA=0.3,**kwargs):
        MV2.__init__(self,**kwargs)
        self.ASSOCIATION = ASSOCIATION3
        self.Lambda_FA = Lambda_FA # Average number of false alarms per frame
        Sigma_FA = self.O*self.Sigma_init*self.O.T + self.Sigma_O
        self.Sigma_FA = Sigma_FA
        self.log_FA_norm = - math.log(scipy.linalg.det(Sigma_FA))/2
        self.Sigma_FA_I = scipy.linalg.inv(Sigma_FA)
        
    ############## Begin simulation methods ######################
    def simulate(self, T):
        """ Return a sequence of T observations and a sequence of T
        states."""
        x_j,v_j,N_FA = self.sim_init(self.N_tar)
        zero_x,zero_y = self.sim_zeros()
        # Lists for results
        xs = []
        obs = []
        for t in xrange(T):
            xs.append(x_j)        # Save X[t] part of state for return
            # Generate observations Y[t]
            obs_t = self.observe_states(x_j,v_j,zero_y)
            for k in xrange(N_FA):
                obs_t.append(util.normalS(zero_y,self.Sigma_FA))
            obs.append(self.shuffle(obs_t))
            # Generate state, visibility, and N_FA for next t
            x_j = map(lambda x: self.step_state(x,zero_x),x_j)
            v_j = map(self.step_vis,v_j)
            N_FA = scipy.random.poisson(self.Lambda_FA)
        return obs,xs
    ############## End simulation methods ######################

class MV4:
    """  
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
                 MaxP = 120,
                 PV_V=[[.9,.1],[.2,.8]],
                 Lambda_new=0.05,
                 Lambda_FA=0.3
                 ):
        self.ASSOCIATION = ASSOCIATION4
        self.TARGET = TARGET2
        self.N_tar = N_tar
        self.A = scipy.matrix(A)
        dim,check = self.A.shape
        if dim != check:
            raise RuntimeError,\
         "A should be square but it's dimensions are (%d,%d)"%(dim,check)
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
        self.PV_V = scipy.matrix(PV_V)
        self.Lambda_FA = Lambda_FA # Average number of false alarms per frame
        Sigma_FA = self.O*self.Sigma_init*self.O.T + self.Sigma_O
        self.Sigma_FA = Sigma_FA
        self.log_FA_norm = - math.log(scipy.linalg.det(Sigma_FA))/2
        self.Sigma_FA_I = scipy.linalg.inv(Sigma_FA)
        self.ASSOCIATION = ASSOCIATION4
        self.Lambda_new = Lambda_new # Average number of new targets
                                     # per frame
        self.log_new_norm = 0.0 # FixMe: Is this included in target creation?

    def decode_forward(self,
               Ys # Observations. Ys[t][k] is the kth hit at time t
               ):
        """Forward pass for decoding.  Return association at final
        time with highest utility, nu."""
        T = len(Ys)

        # Initialize by making a target and cause for each of the hits
        # at t=0 and collecting them in a single association.
        target_0 = self.TARGET(self,[0],[self.mu_init],[self.Sigma_init],
                               0.0,None)
        partial = self.ASSOCIATION(0.0,self)
        
        # Forward pass through time
        old_As = [partial]
        for t in xrange(0,T):
            child_targets = {}          # Dict of all targets for time t
            successors = SUCCESSOR_DB() # Just to make the while loop start
            while successors.length() is 0:
                # For each old target, collect all plausibly
                # associated hits and create a corresponding child
                # target by Kalman filtering
                for A in old_As:
                    A.make_children(Ys[t],child_targets,t)
                # Build u' lists for all possible successor
                # associations at time t.  Put new_As in DB so many
                # predecessors can find same successor
                successors = SUCCESSOR_DB()
                for A in old_As:
                    A.forward(successors,Ys[t],target_0)
                self.MaxD *= 2
                #### Begin debugging check and print ####
                if successors.length() is 0 and \
                       (self.MaxD<1e-6 or self.MaxD>1e6):
                    raise RuntimeError,"""
No new associations in decode():
t=%d, len(old_As)=%d, len(child_targets)=%d, len(Ys[t])=%d,MaxD=%g
successors.length()=%d
"""%(t,len(old_As), len(child_targets),len(Ys[t]),self.MaxD,
     successors.length())
                #### End debugging check and print ####
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
        return (A_best,T)
    def decode_back(self,A_best,T):
        targets_times = []
        for pair in A_best.dead_targets.values():
            target = pair[0]
            t_i = pair[1]-len(target.m_t)
            t_f = pair[1]-Invisible_Lifetime
            for List in target.m_t,target.mu_t,target.Sigma_t:
                del(List[-Invisible_Lifetime:])
            targets_times.append([target,t_i,t_f])
        for target in A_best.targets:
            t_f = T
            t_i = T-len(target.m_t)
            targets_times.append([target,t_i,t_f])
        y_A = []
        for t in xrange(T):
            y_A.append({}) # Initialize association dicts
        N_tar = len(targets_times)
        d = []
        for k in xrange(N_tar):
            d.append(T*[None])# Initialize list of decoded states with Nones
            target,start,stop = targets_times[k]
            d[k][start:stop] = target.backtrack()# Set decoded values
            for t in xrange(start,stop):
                y_A[t][k] = target.m_t[t-start] # y[t][y_A[t][k]] is
                                                # associated with target k.
        return (d,y_A)
    def decode(self,
               Ys # Observations. Ys[t][k] is the kth hit at time t
               ):
        """Return MAP state sequence """
        A_best,T = self.decode_forward(Ys)
        return self.decode_back(A_best,T)
        
    ############## Begin simulation methods ######################
    def step_state(self, state, zero_s):
        epsilon = util.normalS(zero_s,self.Sigma_D) # Dynamical noise
        return self.A*state + epsilon
    def observe_states(self, states_t, v_t, zero_y):
        v_states = []
        for k in xrange(len(v_t)):
            if v_t[k] is 0:
                v_states.append(states_t[k])
        y_t = []
        for state in v_states:
            eta = util.normalS(zero_y,self.Sigma_O) # Observational noise
            y_t.append(self.O * state + eta)
        return y_t
    def shuffle(self, things_i):
        N = len(things_i)
        if N is 0:
            return []
        things_o = N*[None]
        permute = util.index_to_permute(random.randint(0,N-1),N)
        for k in xrange(N):
            things_o[k] = things_i[permute[k]]
        return things_o
    def sim_zeros(self):
        """ Make two useful zero matrices """
        s_dim = self.mu_init.shape[1]
        zero_s = scipy.matrix(scipy.zeros(s_dim))
        y_dim = self.Sigma_O.shape[0]
        zero_y = scipy.matrix(scipy.zeros(y_dim))
        return (zero_s,zero_y)
    def step_vis(self,v):
        pv = self.PV_V[v,0]
        if pv > random.random():
            return 0
        else:
            return 1
    def sim_init(self,N):
        x_j = []
        v_j = []
        for j in xrange(N):
            x_j.append(util.normalS(self.mu_init,self.Sigma_init))
            v_j.append(0)
        return (x_j,v_j,0)
    def step_count(self, x, v, c, zero_x):
        """ Step target vector x and visibilty v count of number of
        times that v != 0 (not visible) in c.
        """
        x = self.step_state(x,zero_x)
        v = self.step_vis(v)
        if v is 0:
            c = 0
        else:
            c += 1
        return (x,v,c)
    def run_state(self,x,v,zero_x,t_0,T):
        s_k = T*[None]
        v_k = T*[None]
        c = 0
        for t in xrange(t_0,T):
            s_k[t] = x
            v_k[t] = v
            x,v,c = self.step_count(x,v,c,zero_x)
            if c > Invisible_Lifetime:
                s_k[t-Invisible_Lifetime:t] = Invisible_Lifetime*[None]
                break
        return (s_k,v_k)
    def simulate(self, T): # for MV4
        """ Return a sequence of T observations and a sequence of T
        states."""
        #Start with N_tar visible targets
        x_0,v_0,N_FA = self.sim_init(self.N_tar)
        zero_x,zero_y = self.sim_zeros()
        skt = []
        vkt = []
        obs = []
        for k in xrange(self.N_tar):
            x,v = self.run_state(x_0[k],v_0[k],zero_x,0,T)
            skt.append(x)
            vkt.append(v)
        for t in xrange(T):
            obs.append([])
            N_new = scipy.random.poisson(self.Lambda_new)
            x_0,v_0,N_FA = self.sim_init(N_new)
            for k in xrange(N_new):
                x,v = self.run_state(x_0[k],v_0[k],zero_x,t,T)
                skt.append(x)
                vkt.append(v)
        stk = []
        for t in xrange(T):
            stk.append([])
            s_vis = []
            for k in xrange(len(skt)):
                s = skt[k][t]
                stk[t].append(s)
                if s is not None and vkt[k][t] is 0:
                    s_vis.append(s)
            y = self.observe_states(s_vis,len(s_vis)*[0],zero_y)
            N_FA = scipy.random.poisson(self.Lambda_FA)
            for k in xrange(N_FA):
                y.append(util.normalS(zero_y,self.Sigma_FA))
            obs[t] = self.shuffle(y)
        return obs,stk
    ############## End simulation methods ######################

# Test code
if __name__ == '__main__':
    import time
    random.seed(3)
    ts = time.time()
    for pair in (
        [MV1(N_tar=4),'MV1'],
        [MV2(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),'MV2'],
        [MV3(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),'MV3'],
        [MV4(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),'MV4']
        ):
        M=pair[0]
        print pair[1]

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
