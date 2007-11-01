"""
File mvx.py provides four model classes with the following features:

MV1: Fixed number of targets.  At each time there is a one to one map
     between targets and hits

MV2: Fixed number of targets.  Targets may fail to produce hits

MV3: Fixed number of targets.  Targets may fail to produce hits and
     hits may be false alarms

MV4: Targets die if invisible for Invisible_Lifetime.  New targets can
     appear at any time.  Targets may fail to produce hits and hits may
     be false alarms

Each model class depends on its own association class, eg, MV4 depends
on ASSOCIATION4.  There are two target classes: TARGET4 (for MV2, MV3
and MV4) which can be invisible and TARGET1 (for MV1) which is always
visible.

Here are the family trees most of the classes in this file:

CAUSE_FA -> TARGET4 -> TARGET1

ASSOCIATION4 -> ASSOCIATION3 -> ASSOCIATION2 -> ASSOCIATION1

MV4 -> MV3 -> MV2
       MV3 -> MV1
"""
import numpy, scipy, scipy.linalg, random, math, util
def log_Poisson(Lambda,N):
    return N*math.log(Lambda) - Lambda - math.log(scipy.factorial(N))
Invisible_Lifetime = 5 # After being invisible 5 times in a row targets die
Join_Time = 6          # After 6 identical associations in a row, tragets merge
close_calls = None     # Will be list of flags of likely errors
Target_Counter = 0

class CAUSE_FA: # False Alarm
    def __init__(self, y, Sigma_I ):
        self.type = 'FA'
        self.index = -1
        self.R = -float(y.T*Sigma_I*y/2)
class TARGET4(CAUSE_FA):
    """A TARGET is a possible moving target, eg, a car.  Its values
    are determined by initialization values and a sequence of
    observations that are "Kalman filtered".
    """
    def __init__(self,
                 mod,      # Parent model
                 m_t,      # History of hit indices used
                 mu_t,     # History of means
                 Sigma_t,  # History of variances
                 R,        # Most recent residual
                 y,        # Useful in subclasses
                 index=None
                 ):
        global Target_Counter
        self.type='target'
        if isinstance(index,type(0)): # Check for int
            self.index = index
        else:
            self.index = Target_Counter
            Target_Counter += 1
        self.mod = mod
        self.m_t = m_t
        self.mu_t = mu_t
        self.Sigma_t = Sigma_t
        self.R = R
        self.children = None  # Will be dict of targets at t+1 updated with
                              # hits plausibly caused by this target
        self.invisible_count=0# Number of time steps target has been invisible
    def New(self, *args,**kwargs):
        if args[-1] is None: # y is None, ie invisible
            rv = TARGET4(*args,**kwargs)
            rv.invisible_count = self.invisible_count + 1
            return rv
        else:
            return TARGET4(*args,**kwargs)
    def dump(self):
        print '\nDump target: m_t=',self.m_t
        print '             index=%d, len(mu_t)=%d'%(self.index,len(self.mu_t))
        print '   invisible_count=%d'%self.invisible_count
        if self.children is not None:
            print'  len(self.children)=%d'%len(self.children)
            for child in self.children.values():
                child.dump()
    def make_children(self,        # self is a TARGET4
                      y_t,         # list of hits at time t
                      All_children # Dict of children of all associations
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
        key = tuple(self.m_t + [-1]) # Child for invisible y
        self.children[-1] = All_children.setdefault(key,self.update(None,-1))
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
        """ Rather than the Mahalanobis distance of y from forecast
        y, I generalize to sqrt(-2 Delta_R)
        """
        return float(-2*self.utility(y)[0])**.5
    def update(self,
           y,        # The observation of the target at the current time
           m         # Index of the observation
           ):
        """ Create a new target with updated m_t, mu_t, Sigma_t and R
        for the observation, index pair (y,m)."""
        m_L = self.m_t+[m]
        Delta_R,mu_new,Sigma_new = self.utility(y)
        Sigma_L = self.Sigma_t + [Sigma_new]
        mu_L = self.mu_t + [mu_new]
        return self.New(self.mod,m_L,mu_L,Sigma_L,Delta_R,y,
                             index=self.index)
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
        Delta_y = y - self.y_forecast    # Error of forecast observation
        Sigma_new = self.Sigma_next
        mu_new = self.mu_a + self.K*Delta_y
        Delta_R += -float(Delta_y.T*self.Sigma_y_forecast_I*Delta_y
                           +self.mod.log_det_Sig_O)/2
        return (Delta_R,mu_new,Sigma_new)
    def KF(self,
           y,        # The observation of the target at the current time
           m         # Index of the observation
           ):
        """ Use a Kalman filter to create a new target with a unique
        index, fresh m_t, mu_t, Sigma_t and R for the observation y."""
        self.forecast()
        r = self.update(y,m)
        return self.New(self.mod,[r.m_t[-1]],[r.mu_t[-1]],
                             [r.Sigma_t[-1]],r.R,y)
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
class SUCCESSOR_DB:
    """For each possible successor key, collect candidate predecessors
    and the associated u_prime.
    """
    def __init__(self):
        self.successors = {}
    def enter(self,       
              association  # A candidate predecessor
              ):
        key = tuple(association.h2c) # An explanation of each component of y_t
        u_prime = association.nu     # The utility of the candidate
        suc_key = self.successors.setdefault(key,
                      {'associations':[],'u_primes':[]})
        suc_key['associations'].append(association)
        suc_key['u_primes'].append(u_prime)
    def count(self): # A method to report the number of successor keys
        count_successors = len(self.successors)
        count_predecessors = 0
        for value in self.successors.values():
            count_predecessors += len(value['associations'])
        return (count_successors,count_predecessors)
    def length(self):
        return len(self.successors)
    def maxes(self):
        """ For each key, return the association with the largest
        corresponding u_prime
        """
        rv = []
        for suc in self.successors.values():
            rv.append(suc['associations'][util.argmax(suc['u_primes'])])
        return rv
class ASSOCIATION4:
    """This is the discrete part of the state.  It gives an
    explanation for how the collection of observations at time t were
    made.

    Methods:

     __init__: Called with None or with an existing association and a
               cause to create a new association with an explanation
               for an additional observation.

     New and Fork: Create a child association that explains one more hit

     forward:  Create plausible successor associations

     make_children:  Call target.make_children() for each target
     
     check_targets, check_FAs and check_newts: Checks in list of
         cause_checks called by forward() that propose causes of hits

    invisibles, count_FAs and extra_news: Entries in list of methods
         extra_forward that are called by forward().  These methods
         modify an association to account for invisible targets, false
         alarms, and new targets.
    """
    def __init__(self,nu,mod):
        self.nu = nu        # Utility of best path ending here
        self.mod = mod      # Model, to convey parameters
        self.tar_dict = {}  # Dict of targets
        self.targets = []   # List of targets
        self.h2c = []       # Map from hits to causes.  Will become key
        self.cause_checks = [self.check_targets,self.check_FAs,
                             self.check_newts]
        self.extra_forward = [self.extra_newts,self.count_FAs]
        self.dead_targets = {}
        self.type='ASSOCIATION4'
        self.N_FA=0          # Count of false alarms
    def New(self, *args,**kwargs):
        NA = ASSOCIATION4(*args,**kwargs)
        NA.dead_targets = self.dead_targets.copy()
        return NA
    def Fork(self, # Create a child that extends association by cause
            cause  # CAUSE of next hit in y_t
            ):
        CA = self.New(self.nu + cause.R,self.mod)
        CA.tar_dict = self.tar_dict.copy()
        CA.h2c = self.h2c + [cause.index]
        CA.N_FA = self.N_FA
        if cause.type is 'target':
            CA.tar_dict[cause.index] = None # Prevent reuse of target
            CA.targets = self.targets + [cause]
            return CA
        if cause.type is 'FA':
            CA.targets = self.targets
            CA.N_FA += 1
            return CA
        raise RuntimeError,"Cause type %s not known"%cause.type
    def check_targets(self,k,causes,y):
        """ A check method for list cause_checks called by forward().
        This method appends relevant children to plausible causes.
        """
        for target in self.targets:
            if target.children.has_key(k):
                # If has_key(k) then child is plausible cause
                causes.append(target.children[k])
    def check_FAs(self,k,causes,y):
        """ A check in list of cause_checks called by forward().
        Propose that cause is false alarm
        """
        CC = CAUSE_FA(y,self.mod.Sigma_FA_I)
        if self.mod.MaxD < 1e-7 or (-2*CC.R)**.5 < self.mod.MaxD:
            causes.append(CC) # False alarm
        else:
            print 'test failed, self.mod.MaxD=',self.mod.MaxD
    def check_newts(self, # ASSOCIATION4
                    k,causes,y):
        """ A check in list of cause_checks called by forward().
        Propose that cause is false alarm or new target
        """
        CC = self.mod.newts[k]
        if self.mod.MaxD  < 1e-7 or (-2*CC.R)**.5 < self.mod.MaxD:
            causes.append(CC)  # New target
        else:
            print 'test failed, self.mod.MaxD=',self.mod.MaxD
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
        'There is no invisible child.'
                partial.targets.append(child)
                partial.nu += child.R 
    def count_FAs(self,partial):
        """ A method in extra_forward called by forward().  Calculate
        the utility for the number of false alarms and add that
        utility to partial.nu.
        """
        N_FA = partial.N_FA
        if N_FA > 0:
            partial.nu += N_FA*self.mod.log_FA_norm + log_Poisson(
                self.mod.Lambda_FA,N_FA)
    def extra_newts(self,partial):
        """ A method in extra_forward called by forward().  Put
        invisible targets in new association, and apply creation
        penalty to nu for newly created targets.
        """ 
        for target in self.targets:
            if not partial.tar_dict.has_key(target.index) and \
                   target.children.has_key(-1): # Invisible target
                child = target.children[-1]
                partial.targets.append(child)
                partial.nu += child.R
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
                      cousins, # Dict of children of sibling associations
                      t        # Save time for dead targets
                      ):
        for target in self.targets:
            if target.invisible_count < Invisible_Lifetime:
                target.make_children(y_t,cousins)
            else:
                self.dead_targets.setdefault(target.index,[target,t])
                target.children = {}
    def forward(self,       # ASSOCIATION4
                successors, # DB of associations and u's for the next time step
                y_t,        # Hits at this time
                ):
        """ For each plausible successor S of the ASSOCIATION self,
        enter the following into the successors DB 1. A key for the
        explanation that S gives for the observations. 2. The
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
                    # Don't use same target index more than once.
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
class MV4:
    """ A state consists of: Association; Locations; and Visibilities;
    (and the derivable N_FA), ie,
    s = (M,X,V,N_FA)
    """
    def __init__(self,                         # MVa1
                 N_tar = 3,                    # Number of targets
                 A = [[0.81,1],[0,.81]],       # Linear state dynamics
                 Sigma_D = [[0.01,0],[0,0.4]], # Dynamical noise
                 O = [[1,0]],                  # Observation projection
                 Sigma_O = [[0.25]],           # Observational noise
                 Sigma_init = None,            # Initial state distribution
                 mu_init = None,               # Initial state distribution
                 MaxD = 0,                     # Threshold for hits
                 MaxA = 120,                   # Max number of associations
                 PV_V=[[.9,.1],[.2,.8]],       # Visibility transition matrix
                 Lambda_new=0.05,              # Avg # of new targets per step
                 Lambda_FA=0.3                 # Avg # of false alarms per step
                 ):
        self.ASSOCIATION = ASSOCIATION4
        self.TARGET = TARGET4
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
        if mu_init == None:
            self.mu_init = scipy.matrix(scipy.zeros(dim)).T
        else:
            self.mu_init = scipy.matrix(mu_init).T
        if Sigma_init == None:
            Sigma_init = scipy.matrix(scipy.identity(dim))
            for t in xrange(100):# A clever man would simply solve
                Sigma_init = self.A*Sigma_init*self.A.T + self.Sigma_D
            self.Sigma_init = Sigma_init
        else:
            self.Sigma_init = scipy.matrix(Sigma_init)
        self.MaxD = MaxD
        self.MaxD_limit = MaxD
        self.MaxA = MaxA
        self.PV_V = scipy.matrix(PV_V)
        self.Lambda_FA = Lambda_FA # Average number of false alarms per frame
        Sigma_FA = self.O*self.Sigma_init*self.O.T + self.Sigma_O
        self.Sigma_FA = Sigma_FA
        self.log_FA_norm = - math.log(scipy.linalg.det(Sigma_FA))/2
        self.Sigma_FA_I = scipy.linalg.inv(Sigma_FA)
        self.Lambda_new = Lambda_new # Average number of new targets per frame
        self.log_new_norm = 0.0 # FixMe: Is this included in target creation?
    def make_newts(self, # MV4
                   Yts   # List of ys for current time
                   ):
        self.newts = {}
        for k in xrange(len(Yts)):
            self.newts[k] = self.target_0.KF(Yts[k],k)
    def decode_init(self,Ys): # Ys is used by subclass
        self.target_0 = self.TARGET(self,[0],[self.mu_init],
                                    [self.Sigma_init],0.0,None)
        return ([self.ASSOCIATION(0.0,self)],0) # First A and first t
    
    def decode_forward(self,   # MV4
                       Ys,     # Observations. Ys[t][k]
                       old_As, # Initial association or nub
                       t_first,# Starting t for iteration
                       analysis = False
                       ):
        """Forward pass for decoding.  Return association at final
        time with highest utility, nu."""
    
        debug_string ="""
No new associations in decode():
t=%d, len(old_As)=%d, len(child_targets)=%d, len(Ys[t])=%d,MaxD=%g
successors.length()=%d
"""
        global close_calls
        close_calls = []
        T = len(Ys)
        for t in xrange(t_first,T):
            self.make_newts(Ys[t])
            child_targets = {}          # Dict of all targets for time t
            suc_length = 0 # Just to make the while loop start
            while suc_length is 0:
                # For each old target, collect all plausibly
                # associated hits and create a corresponding child
                # target by Kalman filtering
                for A in old_As:  # For t=0 this makes no children
                    A.make_children(Ys[t],child_targets,t)
                # Build u' lists for all possible successor
                # associations at time t.  Put new associations in DB
                # with u' so many predecessors can find same successor
                successors = SUCCESSOR_DB()
                for A in old_As:
                    A.forward(successors,Ys[t])
                self.MaxD *= 1.5
                suc_length = successors.length()
                # Begin debugging check and print
                if suc_length is 0 and \
                       (self.MaxD<1e-6 or self.MaxD>1e6):
                    for A in old_As:
                        A.dump()
                    raise RuntimeError,debug_string%(t, len(old_As),
                          len(child_targets),len(Ys[t]),self.MaxD,
                          suc_length)
                     # End debugging check and print
            # End of while
            self.MaxD = max(self.MaxD/2.25,self.MaxD_limit)
            # For each association at time t, find the best predecessor
            # and the associated targets and collect them in old_As
            new_As = successors.maxes()
            if t > Join_Time:
                # Find targets that match back to Join_Time
                shorts = {}
                for key in child_targets.keys():
                    target = child_targets[key]
                    short_key = tuple(target.m_t[-Join_Time:])
                    if not shorts.has_key(short_key):
                        shorts[short_key] = [[],[]]
                    shorts[short_key][0].append(key)
                    shorts[short_key][1].append(target.R)
                condemned = {}
                for short_key in shorts.keys():
                    if len(shorts[short_key][0]) == 1:
                        continue
                    k_max = util.argmax(shorts[short_key][1])
                    del shorts[short_key][0][k_max]
                    for key in shorts[short_key][0]:
                        condemned[key] = True
                # Delete associations that have condemned targets

            # Pass up to MaxA new_As to old_As
            if len(new_As) > self.MaxA:
                Rs = []
                for A in new_As:
                    Rs.append(A.nu)
                Rs.sort()
                limit = Rs[-self.MaxA]
                old_As = []
                for A in new_As:
                    if A.nu >= limit:
                        old_As.append(A)
            else:
                old_As = new_As
            if analysis is not False:
                analysis.append(t,child_targets,new_As,old_As,successors.count())
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
        y_A = [] # y_A[t] is a dict.  y_A[t][k] gives the index of
                 # y[t] associated with target k.  So y[t][A[t][k]] is
                 # associated with target k.
        for t in xrange(T):
            y_A.append({}) # Initialize association dicts
        d = [] #If not None, d[k][t] is the x vector for target_k at time t
        for k in xrange(len(targets_times)):
            d.append(T*[None])# Initialize list of decoded states with Nones
            target,start,stop = targets_times[k]
            d[k][start:stop] = target.backtrack()# Set decoded values
            for t in xrange(start,stop):
                y_A[t][k] = target.m_t[t-start] # y[t][y_A[t][k]] is
                                                # associated with target k.
        return (d,y_A)
    def decode(self,
               Ys, # Observations. Ys[t][k] is the kth hit at time t
               analysis = False # A debugging option
               ):
        """Return MAP state sequence """
        A,t_0 = self.decode_init(Ys)
        A_best,T = self.decode_forward(Ys,A,t_0,analysis=analysis)
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
        """ Step target vector x and visibility v count of number of
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
            if c >= Invisible_Lifetime:
                s_k[t-Invisible_Lifetime:t] = Invisible_Lifetime*[None]
                break
        return (s_k,v_k)
    def simulate(self, T): # for MV4
        """ Return a sequence of T observations and a sequence of T
        states."""
        x_0,v_0,N_FA = self.sim_init(self.N_tar) # N_tar visible targets
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
class TARGET1(TARGET4):
    def New(self, *args,**kwargs):
        return TARGET1(*args,**kwargs)
    def make_children(self, y_t,   # list of hits at time t
       All_children ): # Dict of children of this and other associations
        """ Like TARGET4 make_children but no invisible targets
        """
        self.forecast()
        self.children = {}
        MD = self.mod.MaxD
        for k in xrange(len(y_t)):
            if MD < 0.01 or self.distance(y_t[k]) < MD:
                key = tuple(self.m_t+[k])
                if not All_children.has_key(key):
                    All_children[key] = self.update(y_t[k],k)
                self.children[k] = All_children[key]
    def utility(self,y,R0=0.0):
        """ Like TARGET4.utility but no visibility probabilities.
        """
        Delta_y = y - self.y_forecast
        Sigma_new = self.Sigma_next
        mu_new = self.mu_a + self.K*Delta_y
        Delta_R = R0-float(Delta_y.T*self.Sigma_y_forecast_I*Delta_y
                           +self.mod.log_det_Sig_O)/2
        return (Delta_R,mu_new,Sigma_new)
class ASSOCIATION3(ASSOCIATION4):
    """ Number of targets is fixed.  Allow false alarms and invisibles
    """
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.cause_checks = [self.check_targets,self.check_FAs]
        self.extra_forward = [self.invisibles,self.count_FAs]
        self.type='ASSOCIATION3'
    def New(self, *args,**kwargs):
        return ASSOCIATION3(*args,**kwargs)
    def make_children(self, y_t, cousins, t ):
        """ No dead targets for ASSOCIATION3. """
        for target in self.targets:
            target.make_children(y_t,cousins)
    def make_newts(self,*args,**kwargs): # No new targets for ASSOCIATION3
        pass
class ASSOCIATION2(ASSOCIATION3):
    """ Allow invisibles, ie, targets that fail to generate hits. """
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.cause_checks = [self.check_targets]
        self.extra_forward = [self.invisibles]
        self.type='ASSOCIATION2'
    def New(self, *args,**kwargs):
        return ASSOCIATION2(*args,**kwargs)
class ASSOCIATION1(ASSOCIATION3):
    """ No extra_forward methods.  Targets and only targets generate hits. """
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.extra_forward = []
        self.cause_checks = [self.check_targets]
        self.type='ASSOCIATION1'
        mod = self.mod
    def New(self, *args,**kwargs):
        return ASSOCIATION1(*args,**kwargs)
class MV3(MV4):
    """ Number of targets is fixed    
    """
    def __init__(self,**kwargs):
        MV4.__init__(self,**kwargs)
        self.ASSOCIATION = ASSOCIATION3   
        self.TARGET = TARGET4 
    def decode_init(self, Ys ):
        """Since MV4 allows generation of new targets at any time,
        MV4.decode_init() can return zero initial targets and let
        decode_forward() create targets for t=0.  This decode_init()
        must return N_tar initial targets.  It makes a target for each
        of the hits at t=0 and tells decode_forward() to start at t=1.
        """
        self.target_0 = self.TARGET(self,[0],[self.mu_init],
                                    [self.Sigma_init],0.0,None)
        T = len(Ys)
        partial = self.ASSOCIATION(0.0,self)
        for k in xrange(self.N_tar):
            target_k = self.target_0.KF(Ys[0][k],k)
            partial = partial.Fork(target_k)
        return ([partial],1)
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
class MV2(MV3):
    def __init__(self,**kwargs):
        MV4.__init__(self,**kwargs)
        self.ASSOCIATION = ASSOCIATION2
        self.TARGET = TARGET4    
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
class MV1(MV3):
    def __init__(self,**kwargs):
        MV4.__init__(self,**kwargs)
        self.ASSOCIATION = ASSOCIATION1
        self.TARGET = TARGET1
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
class analysis:
    def __init__(self):
        self.records = {}
    def append(self,t,child_targets,new_As,old_As,counts):
        """ From child_targets, I want to know how many different
        targets and how many different target.index values there are.
        I might want to know how many targets were invisible.  I want
        simple counts of new_As and old_As and of the successors and
        their predecessors.
        """
        record = {}
        index_count = {}
        for target in child_targets.values():
            if not index_count.has_key(target.index):
                index_count[target.index] = 0
            else:
                index_count[target.index] += 1
        for pair in (['index_count',index_count],
                     ['child_count',len(child_targets)],
                     ['new_count',len(new_As)],
                     ['old_count',len(old_As)],
                     ['succ_count',counts[0]],
                     ['pred_count',counts[1]]):
            record[pair[0]] = pair[1]
        self.records[t] = record
    def dump(self):
        ts = self.records.keys()
        ts.sort()
        for t in ts:
            record = self.records[t]
            if record['new_count'] != record['succ_count']:
                print 'NOTICE: new_count=%d != succ_count=%d'%(
                    record['new_count'],record['succ_count'])
            print '\nt=%2d'%t,
            for key in ('pred_count','new_count','old_count','child_count'):
                print '%s=%6d'%(key,record[key]),
        print '\n'
        for t in ts:
            print 't=%2d, index_count='%t,self.records[t]['index_count']
if __name__ == '__main__':  # Test code
    import time
    random.seed(3)
    scipy.random.seed(3)
    for pair in (
       [MV1(N_tar=4),'MV1',0.26],
       [MV2(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),'MV2',0.09],
       [MV3(N_tar=3,PV_V=[[.5,0.5],[0.5,.5]]),'MV3',0.21],
       [MV4(N_tar=3,PV_V=[[.5,0.5],[0.5,.5]]),'MV4',4.00]
       ):
        Target_Counter=0
        M=pair[0]
        print '%s: Begin simulate'%pair[1]
        y,s = M.simulate(5)
        print 'Begin decode; expect %4.2f seconds on AMD Sempron 3000'%pair[2]
        ts = time.time()
        A = analysis()
        d,tmp = M.decode(y,analysis=A)
        print 'Elapsed time = %4.2f seconds.  '%(time.time()-ts)
        print 'len(y)=',len(y), 'len(s)=',len(s),'len(d)=',len(d)
        for t in xrange(len(s)):
            print 't=%d    y         s           d'%t
            for k in xrange(max(len(s[t]),len(d))):
                try:
                    print ' k=%d  %4.2f  '%(k,y[t][k][0,0]),
                except:
                    print ' k=%d        '%k,
                try:
                    print '(%4.2f, %4.2f)  '%(s[t][k][0,0],s[t][k][1,0]),
                except:
                    print '              ',
                try:
                    print '(%4.2f, %4.2f)  '%(d[k][t][0,0],d[k][t][1,0]),
                except:
                    pass
                print ' '
        print '\n'
    A.dump()

#---------------
# Local Variables:
# eval: (python-mode)
# End:
