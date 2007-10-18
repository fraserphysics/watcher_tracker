"""
mvx.py: All model variations in a single file.  I started this after
writing mv3.py which imported mv2.py and mv1a.py.  I wanted more
coherence before starting on mv4 which allows a variable number of
targets.

mv1a.py used a map from targets to observations.  I want to use a map
from observations to causes for mv4 and I modified mv3 to do that.
This file will use maps from observations to causes for all model
versions.
"""
import numpy, scipy, scipy.linalg, random, math, util

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
                 R_t       # Most recent residual
                 ):
        self.mod = mod
        self.m_t = m_t
        self.mu_t = mu_t
        self.Sigma_t = Sigma_t
        self.R_t = R_t
        self.children = None # List of targets at t+1 updated with
                             # plausible hits and this target
    def dump(self):
        print 'Dump target: m_t=',self.m_t
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
        return (self.mod,m_L,mu_L,Sigma_L,Delta_R)
    def utility(self,y,R0=0.0):
        """ Calculates Delta_R, mu_new, and Sigma_new for both
        update() and distance().  This is the second half of Kalman
        filtering step.
        """
        Delta_y = y - self.y_forecast    # Error of forecast observation
        Sigma_new = self.Sigma_next
        mu_new = self.mu_a + self.K*Delta_y
        Delta_R = R0+(self.mod.log_det_Sig_D+self.mod.log_det_Sig_O + float(
                Delta_y.T*self.Sigma_y_forecast_I*Delta_y))/2
        return (Delta_R,mu_new,Sigma_new)
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

class TARGET2(TARGET):
    def make_children(self,y_t,All_children):
        """  Include invisibility.
        """
        TARGET.make_children(y_t,All_children)
        key = tuple(self.m_t + [-1])
        if not All_children.has_key(key):
            All_children[key] = self.update(y,k)
        self.children[k] = All_children[key]

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
            return (Delta_R,mu_new,Sigma_new)
        return TARGET.utility(y,R0=Delta_R)

class SUCCESSOR_DB:
    """
    """
    def __init__(self):
        self.successors = {}
    def enter(self,
              key,         # A tuple that explains each component of y_t
              association, # A candidate history that arrives at key
              u_prime      # The utility of the candidate
              ):
        if not self.successors.has_key(key):
            self.successors[key] = {'associations':[],'u_primes':[]}
        self.successors[key]['associations'].append(association)
        self.successors[key]['u_primes'].append(u_prime)
    def max(self,key):
        """ Return the association with the largest corresponding u_prime
        """
        successor = self.successors[key]
        return successor['associations'][util.argmax(successor['u_prime']]
class ASSOCIATION:
    """This is the discrete part of the state.  It gives an
    explanation for how the collection of observations at time t were
    made.

    Methods:

     __init__: Called with an existing association and a cause to
               create a new association with an explanation for an
               additional observation.
     
     forward:  Create plausible sucessor associations

     argmax:   Select best predecessor, evaluate self.nu, collect list
               of appropriate child targets from that predecessor,
               attach the list to self, and return the list

     make_children:  Call target.make_children() for each target

       
    """
    def __init__(self,
                 parent=None, # A partial association
                 cause=None,  # CAUSE of next hit in y_t
                 nu=None      # nu from parent association/permutation
                 ):
        if parent is None and cause is None:
            self.dup_check = {} # Dict of parent targets used
            self.perm = []      # Map from hits to sources.  Will become key
            self.u_prime = nu   # u'(self,suc,t+1)
            self.targets = []   # List of targets
            self.predecessor_perm = []  # List of predecessor associations
            self.predecessor_u_prime=[] # u' values for the predecessors
            self.nu = nu                # Utility of best path ending here
            return
        if not (parent and cause):
            raise RuntimeError,\
                  "parent and cause must both be None or both be defined"
        self.dup_check = parent.dup_check.copy()
        self.perm = parent.perm + [cause.index]
        self.u_prime = parent.u_prime + cause.R
        if not cause.type is 'target':
            raise RuntimeError,"Cause type %s not known"%cause.type
        self.dup_check[cause.index] = None # Block reuse of parent
        self.targets = parent.targets + [cause.target]
    
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
        # Create a list of associations between observations and
        # targets to consider as sucessors to the association "self".

        # Iterate over targets in the present association.  Initialize
        # with target[0]
        old_list = [] # List of partial associations.  At level k,
                      # each association says which y is associated
                      # with targets with indices smaller than k
        for child in self.targets[0].children.values():
            m_tail = child.m_t[-1]
            old_list.append({
                'dup_check':{m_tail:None}, # Hash table to ensure unique
                                           # hit associations
                'perm':[m_tail],           # Map from targets to hits
                'R':child.R_t + self.nu    # u'(self,suc,t+1)
                })
        # Initialization for k=0 done

        # Using the old list of partial associations of length k, for
        # each additional target, combine each possible (target,
        # observation) paring with each old partial association that
        # does not already have a match for that observation and
        # append it to the new list of partial associations of length
        # k+1
        for k in xrange(1,len(self.targets)):
            new_list = []
            for child in self.targets[k].children.values():
                m_tail = child.m_t[-1]
                for partial in old_list:
                    if m_tail >= 0 and partial['dup_check'].has_key(m_tail):
                        continue # Many targets mapping to invisible y OK
                    new_dict = partial['dup_check'].copy()
                    new_dict[m_tail] = None
                    new_perm = partial['perm']+[m_tail]
                    new_R = partial['R']+child.R_t
                    new_list.append({'dup_check':new_dict,'perm':new_perm,
                                     'R':new_R})
            old_list = new_list
        # y[t+1][old_list[i]['perm'][j]] is associated with target[j]

        # Make sure that for each association each observation is
        # associated with a target.  Not necessary for MV1a, but is
        # necessary for models that allow number of targets and
        # observations to differ
        final_list = []
        for candidate in old_list:
            OK = True
            for k in xrange(len(y_t)):
                if not candidate['dup_check'].has_key(k):
                    OK = False
                    continue
            if OK:
                final_list.append(candidate)
        # Initialize successors if necessary and set their predecessors
        for entry in final_list:
            key = tuple(entry['perm'])  # Dict keys can be tuples but not lists
            if not new_perms.has_key(key):
                new_perms[key] = ASSOCIATION(self.N_tar,key)
            successor = new_perms[key]
            successor.predecessor_perm.append(self)
            successor.predecessor_u_prime.append(entry['R'])

class ASSOCIATION(mv1a.ASSOCIATION):
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

        # Now each entry in old_list is a PARTIAL and entry.perm[k]
        # is a plausible CAUSE for hit y_t[k]

        # Initialize successors if necessary and set their predecessors
        for entry in old_list:
            key = tuple(entry.perm)  # Key for successor
            if not new_perms.has_key(key):
                new_perms[key] = ASSOCIATION(self.N_tar,key)
            successor = new_perms[key]
            successor.predecessor_perm.append(self)
            successor.predecessor_u_prime.append(entry.u_prime)

#---------------
# Local Variables:
# eval: (python-mode)
# End:
