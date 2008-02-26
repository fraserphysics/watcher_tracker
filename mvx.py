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

FixMe: To do:

1. Figure out why nu_max and track differ depending exhaustive vs
   Murty for MV2 and MV3.

2. Figure out occasional faliure of Murty

3. Flag likely errors correctly.  Places to think about:

   When Murty's algorithm works on big clusters
   TARGET4.make_children
   ASSOCIATION4.join
   ASSOCIATION4.check_FAs   Should I try false alarms when targets are good? 
   ASSOCIATION4.check_newts Should I try a newt when targets are good? 
   ASSOCIATION4.forward     Implement Murty's algorithm.  Prune here
                            with threshold that is independent of MaxD
   Cluster.__init__         What is the right way to apportion utility
   Cluster.Append           What is the right way to apportion utility
   Cluster_Flock.recluster  First fragment may not have best utility
   Cluster_Flock.recluster  Consistency check should include dead_targets
   Cluster_Flock.recluster  For which observations should I make newts

"""
import numpy, scipy, scipy.linalg, random, math, util, time
close_calls = None     # Will be list of flags of likely errors
Target_Counter = 0
Child_Target_Counter = 0
hungary_count = 0
hungary_time = 0
def dict_2_tuple(tk_dict): # Return tuple made from sorted list of keys
    tk_list = tk_dict.keys()
    tk_list.sort()
    return tuple(tk_list)

def list_2_flat(tk_list): # Map list of tuples to flat tuple
    tk_list.sort()
    flat_list = []
    for pair in tk_list:
        flat_list += list(pair)
    return tuple(flat_list)

class CAUSE: # Dummy for subclassing and making Fork() copy association
    def __init__(self ):
        self.type = 'void'
        self.R = 0
        self.tks = {}
class FA(CAUSE): # False Alarm
    def __init__(self, y, t, k, Sigma_I, norm ):
        self.type = 'FA'
        self.index = -1
        self.R = norm - float(y.T*Sigma_I*y/2)
        self.R_sum = self.R
        self.t = t
        self.k = k
        self.tks = {(t,k):self}
    def dump(self, #FA
             ):
        print 'Dumping %s: R=%6.2f, (t,k)=(%d,%d)'%(self.type,self.R,
                                                    self.t,self,k)
        return
class TARGET4(CAUSE):
    """A TARGET is a possible moving target, eg, a car.  Its values
    are determined by initialization values and a sequence of
    observations that are "Kalman filtered".
    """
    def __init__(self,      # TARGET4
                 mod,       # Parent model
                 m_t,       # List of hit indices used
                 mu_t,      # History of means
                 Sigma_t,   # History of variances
                 R,         # Most recent residual
                 y,         # Useful in subclasses
                 tks,       # Dict of (t,k) pairs explained
                 index=None,# Unique ID 
                 R_sum=None # Accumulation of past R values
                 ):
        global Target_Counter, Child_Target_Counter
        assert type(tks) == type({})
        self.type='target'
        if isinstance(index,type(0)): # Check for int
            self.index = index
            Child_Target_Counter += 1
        else:
            self.index = Target_Counter
            Target_Counter += 1
        self.mod = mod
        self.m_t = m_t
        self.mu_t = mu_t
        self.Sigma_t = Sigma_t
        self.R = R
        self.tks = tks
        self.children = None  # Will be dict of targets at t+1 updated with
                              # hits plausibly caused by this target.
                              # Keys are hit indices k
        self.invisible_count=0# Number of time steps target has been invisible
        if R_sum is None:
            self.R_sum = R
        else:
            self.R_sum = R_sum
        self.T0=None
        self.last=None
    def New(self,   # TARGET4
            *args,**kwargs):
        rv = TARGET4(*args,**kwargs)
        if rv.m_t[-1] < 0:
            rv.invisible_count = self.invisible_count + 1
        rv.T0 = self.T0
        return rv
    def dump(self #TARGET4
             ):
        print '\n Dump %s: m_t='%self.type,self.m_t
        tks = self.tks.keys()
        tks.sort()
        print '  tks=',tks
        print '             T0=%d, index=%d, len(mu_t)=%d t_last='%(self.T0,
               self.index,len(self.mu_t)),self.last
        print '   invisible_count=%d, R=%5.3f, R_sum=%5.3f'%(
            self.invisible_count, self.R,self.R_sum)
        #if self.children is not None: #
        if False:
            print'  len(self.children)=%d'%len(self.children)
            for child in self.children.values():
                child.dump()
    def make_children(self,        # TARGET4
                      y_t,         # list of hits at time t
                      t
                      ):
        """ For each of the hits that could plausibly be an
        observation of self, make a child target.  Collect the
        children in a dict and attach it to self.
        """
        if not self.children is None:
            return # make_children already called for this target
        self.forecast()
        self.children = {}
        mlpd = self.mod.log_min_pd
        for k in xrange(len(y_t)):
            key = tuple(self.m_t + [k])
            if mlpd > self.utility(y_t[k])[0]:
                continue                 # Candidate utility too small
            self.children[k] = self.update(y_t[k],k,t)
            # update() calls utility second time.  Possible time saving
            assert(self.children[k].m_t[-1]==k)
        # Child for invisible y
        self.children[-1] = self.update(None,-1,t)
    def forecast(self # TARGET4
                 ):
        """ Calculate forecast mean and covariance for both state and
        observation.  Also calculate K and Sigma_next.  Save all six
        for use by Update.
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
    def update(self, # Target4
           y,        # The observation of the target at the current time
           k,        # Index of the observation
           t         # Present time
           ):
        """ Create a new target with updated m_t, mu_t, Sigma_t, R and
        R_sum for the observation, index pair (y,m)."""
        m_L = self.m_t+[k]
        Delta_R,mu_new,Sigma_new = self.utility(y)
        Sigma_L = self.Sigma_t + [Sigma_new]
        mu_L = self.mu_t + [mu_new]
        tks = self.tks.copy()
        if k >= 0:
            tks[(t,k)] = True
        return self.New(self.mod,m_L,mu_L,Sigma_L,Delta_R,y,tks,
                             index=self.index,R_sum=self.R_sum+Delta_R)
    def utility(self,  # TARGET4
                y):
        """Return (log probability density, updated mean, updated
        covariance).  Include log_prob factors for Sig_D and
        visibility transitions.
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
                           -self.mod.log_det_Sig_O)/2
        return (Delta_R,mu_new,Sigma_new)
    def KF(self,     # Target4
           y,        # The observation of the target at the current time
           k,        # Index of the observation
           t         # Present time
           ):
        """ Use a Kalman filter to create a new target with a unique
        index, fresh m_t, mu_t, Sigma_t and R for the observation y."""
        self.forecast()
        r = self.update(y,k,t)
        tks = {}
        if k >= 0:
            tks[(t,k)] = True
        r = self.New(self.mod,[r.m_t[-1]],[r.mu_t[-1]],
                             [r.Sigma_t[-1]],r.R,y,tks)
        r.T0 = t
        return r
    def backtrack(self # TARGET4
                  ):
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
    def kill(self, # TARGET4
             t
             ):
        self.type = 'dead_target'
        self.last = t
        self.children = {}
def cmp_ass(A,B):
    """ Compare associations.  Want sort to make high utility come first.
    """
    if B.nu - A.nu > 0:
        return 1
    if B.nu == A.nu:
        return 0
    return -1
class SUCCESSOR_DB:
    """For each possible successor key, collect candidate predecessors
    and the associated u_prime.
    """
    def __init__(self):
        self.successors = {}
    def enter(self,       
              association  # A candidate predecessor
              ):
        key = tuple(association.h2c.values()) # Explanation of y_t components
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
        """ Make a sorted list of associations.  For each key, return
        only the association with the largest corresponding u_prime
        """
        rv = []
        for suc in self.successors.values():
            rv.append(suc['associations'][util.argmax(suc['u_primes'])])
        rv.sort(cmp_ass)
        return rv
class ASSOCIATION4:
    """This is the discrete part of the state.  It gives an
    explanation for how the collection of observations at time t were
    made.

    Methods:

     __init__:

     NewA: Create an association that has the same dead targets as self

     Fork: Create a child association that explains one more hit

     Spoon: Modify self to explain one more hit

     join: Merge two associations

     forward:  Create plausible successor associations

     make_children:  Call target.make_children() for each target
     
     check_targets, check_FAs and check_newts: Checks in list of
         cause_checks called by forward() that propose causes of hits

     extra_invisible: Entry in list of methods extra_forward that is
         called by forward().  This methods modifies an association to
         account for invisible targets.

     exhaustive: Find self.mod.Max_NA best next associations by
         exhaustive search

     Murty: Find self.mod.Max_NA best next associations by Murty's
         algorithm

     forward:  Create plausible successor associations
    """
    def __init__(self,      # ASSOCIATION4
                 nu,mod,t=None):
        self.nu = nu        # Utility of best path ending here
        self.mod = mod      # Model, to convey parameters
        self.tar_dict = {}  # Keys: unique target indices. Values: targets
        self.dead_targets={}# Key is target.index value is target
        self.FAs = {}       # Dict of FA.Rs indexed by (t,k)
        self.h2c = {}       # Map from current hits to cause indices.
        self.Atks = {}      # Dict. Key: (t,k), Value: cause
        self.cause_checks = [self.check_targets,self.check_FAs,
                             self.check_newts]
        self.extra_forward = [self.extra_invisible]
        self.t = t
        self.type='ASSOCIATION4'
    def NewA(self, # ASSOCIATION4
             *args,**kwargs):
        return ASSOCIATION4(*args,**kwargs)
    def Enter(self, # ASSOCIATION4
              cause
              ):
        #for tk in cause.tks.keys():
        #    if self.Atks.has_key(tk): # FixMe Why comment this out
        #        return (False,self)
        for tk in cause.tks.keys():
            self.Atks[tk] = cause
            # self.Atks.update(cause.tks) FixMe: use this?
        if cause.type is 'target':
            self.tar_dict[cause.index] = cause
            return (True,self)
        if cause.type is 'dead_target':
            self.dead_targets[cause.index] = cause
            return (True,self)
        if cause.type is 'FA':
            self.FAs[tk] = cause
            return (True,self)
        if cause.type is 'void':
            return (True,self)
        raise RuntimeError,"Cause type %s not known"%cause.type
    def Fork(self,  # ASSOCIATION4
             cause, # CAUSE of next hit in y_t
             k
            ):
        """ Create a child that extends association by cause
        """
        CA = self.NewA(self.nu + cause.R,self.mod) #Child Association
        CA.tar_dict = self.tar_dict.copy()
        CA.dead_targets = self.dead_targets.copy()
        CA.FAs = self.FAs.copy()
        CA.h2c = self.h2c.copy()
        CA.Atks = self.Atks.copy()
        CA.t = self.t
        if k >= 0:
            CA.h2c[k] = cause.index
        return CA.Enter(cause)
    def Spoon(self,   # ASSOCIATION4
              cause,  # CAUSE of next hit in y_t
              k
            ):
        """ Extend association by cause.  Like Fork but modify self
        rather than create child.  Called by Murty.
        """
        self.nu += cause.R
        self.h2c[k] = cause.index
        return self.Enter(cause)
    def re_nu_A(self, # ASSOCIATION4
                ):
        self.nu = 0.0
        self.Atks = self.FAs.copy()
        for FA in self.FAs.values():
            self.nu += FA.R
        for target in self.tar_dict.values() + self.dead_targets.values():
            self.nu += target.R_sum
            for tk in target.tks.keys():
                self.Atks[tk] = target
        return list_2_flat(self.Atks.keys())
        
    def verify(self,  # ASSOCIATION4
               k_list
               ):
        """ Debugging method to verify:
        1. h2c does not duplicate causes
        2. Lenghts of h2c and k_list match
        3. Number of posiitve h2c entries matches length of tar_dict???
        4. Each entry in h2c that maps to a target explains the hit
        5. None of the targets have m_t entries that conflict and
           target.tks is consistient with target.m_t
        6. Entries in Atks match entries in causes.tks (causes=targets
           + dead_targets + FAs)
        """ 
        # Do check #6
        causes = self.tar_dict.values() + self.dead_targets.values()\
            + self.FAs.values()
        Ctks = {}
        for cause in causes:
            for tk in cause.tks.keys():
                Ctks[tk] = cause
                if not self.Atks.has_key(tk):
                    cause.dump()
                    raise RuntimeError, ('cause has tk=(%d,%d) but not Atks.'\
                                             +' Dumped cause')%tk
        for tk in self.Atks:
            if not Ctks.has_key(tk):
                raise RuntimeError, 'Atks has tk=(%d,%d) but no cause has it'%tk
        # Do check #1
        c_dict = {}
        for c in self.h2c.values():
            if c < 0:
                continue
            if c_dict.has_key(c):
                print 'verify fails on c=%d.  h2c='%c,self.h2c
                self.dump()
                raise RuntimeError,'h2c has duplicate'
            c_dict[c] = True
        # Do check 5
        tk_dict = {}
        for target in self.tar_dict.values() + self.dead_targets.values():
            T0 = target.T0
            for t in xrange(len(target.m_t)):
                k = target.m_t[t]
                if k < 0:
                    continue
                tk = (T0+t,k)
                if not target.tks.has_key(tk):
                    print 'At t=%d, T0=%d, target.tks is missing '%(self.t,T0)+\
                    '(%d,%d).  Dump target'%tk
                    target.dump()
                    raise RuntimeError
                if tk_dict.has_key(tk):
                    print 'self.h2c=',self.h2c
                    self.dump()
                    raise RuntimeError,'target.m_t has duplicate (%d,%d)'%(t,k)
                else:
                    tk_dict[tk] = True
        if k_list == None:
            return
        # Do check 2
        N_k_pos = 0
        for k in k_list:
            if k >= 0:
                N_k_pos += 1
        if len(self.h2c) != N_k_pos:
            print 'self.h2c=',self.h2c
            print 'k_list=',k_list
            raise RuntimeError,'len(self.h2c)=%d != %d=n_k_pos'%(
                len(self.h2c),N_k_pos)
        # Do check 4
        for k,c in self.h2c.items(): # Check that targets explain hits
            if c < 0:    # False alarm
                continue
            target = self.tar_dict[c]
            if(target.m_t[-1] != k):
                self.dump()
                print 'k=%d, self.h2c='%k,self.h2c
                print 'k_list=',k_list
                target.dump()
                raise RuntimeError,'target %d fails to explain hit %d'%(c,k)
        return # End of verify
    def join(self, # ASSOCIATION4
            other  # ASSOCIATION4
            ):
        """ Merge other association into self
        """
        self.nu += other.nu
        self.tar_dict.update(other.tar_dict)
        self.h2c.update(other.h2c)
        self.FAs.update(other.FAs)
        self.Atks.update(other.Atks)
        self.dead_targets.update(other.dead_targets)
    # The next three methods go in the list cause_checks called by
    # forward().  Each selects causes to put in the list of plausible
    # causes.
    def check_targets(self, # ASSOCIATION4
                  k,causes,y):
        for target in self.tar_dict.values():
            if target.children.has_key(k):
                causes.append(target.children[k])
    def check_FAs(self, # ASSOCIATION4
                  k,causes,y):
        CC = FA(y,self.t,k,self.mod.Sigma_FA_I, self.mod.log_FA_norm)
        if CC.R > self.mod.log_min_pd:
            causes.append(CC) # False alarm
        else:
            print 'check_FAs rejects, self.mod.MaxD=',self.mod.MaxD
    def check_newts(self, # ASSOCIATION4.  Check for new target
                    k,causes,y):
        if not self.mod.newts.has_key(k):
            return
        CC = self.mod.newts[k]
        if self.mod.log_min_pd > CC.R:
            print ('check_newts rejects, self.mod.MaxD=%5.3f,'+\
                  'CC.R=%5.3f, t=%d, k=%d')%(self.mod.MaxD,CC.R,self.t,k)
            return
        causes.append(CC)
    def extra_invisible(self,   # ASSOCIATION4
                        partial # ASSOCIATION4
                        ):
        """ A method in extra_forward called by forward().  Put
        invisible targets in new association.
        """ 
        for target in self.tar_dict.values():
            if partial.tar_dict.has_key(target.index):
                continue # Skip if child of target in partial
            if target.children.has_key(-1): # Invisible target
                partial.Enter(target.children[-1])
    def dump(self # ASSOCIATION4
             ):
        print 'dumping an association of type',self.type,':'
        print '  nu=%f, len(tar_dict)=%d, len(dead_targets)=%d'%(
            self.nu,len(self.tar_dict),len(self.dead_targets)),
        print 'hits -> causes map=', self.h2c
        print '  Atks.keys()=',dict_2_tuple(self.Atks)
        for target in self.tar_dict.values()+self.dead_targets.values():
            target.dump()
    def make_children(self,    # self is a ASSOCIATION4
                      y_t,     # All observations at time t
                      t        # Save time for dead targets
                      ):
        for target in self.tar_dict.values():
            if target.invisible_count < self.mod.Invisible_Lifetime:
                target.make_children(y_t,t)
            else:
                #print 'target dies after %d invisible steps'%Invisible_Lifetime
                assert not self.dead_targets.has_key(target.index), \
                       'Dying target appears twice?'
                target.kill(t)
                index = target.index
                del self.tar_dict[index]
                self.dead_targets[index] = target
    def exhaustive(self,      # ASSOCIATION4
                   k_list,    # Observation indices in this cluster
                   causes,    # Plausible explanations of each observation
                   floor      # Minimum utility of associations to return
                   ):
        """ A service routine for forward that returns a list of the
        self.mod.Max_NA best associations for t+1 by exhaustive search
        """
        assert(len(k_list) > 0)
        assert(k_list != (-1,))
        # The seed association is copy of self with empty tar_dict
        OK,seed = self.Fork(CAUSE(),-1)
        seed.tar_dict = {}
        seed.h2c = {}
        old_list = [seed]
        # Make list of plausible associations.  At level j, each
        # association explains the source of each y_t[k_list[i]] for
        # i<j.
        for j in xrange(len(k_list)):
            if k_list[j] < 0:
                continue
            new_list = []    # List of partial associations at level j
            for partial in old_list:
                for cause in causes[j]:
                    # Don't use same target index more than once.
                    if not partial.tar_dict.has_key(cause.index):
                        OK,child = partial.Fork(cause,k_list[j])
                        if OK:
                            new_list.append(child)
                old_list = new_list
            if len(old_list) == 0:
                return []
        # Discard low utility associations.  FixMe: doesn't count
        # extra_invisible utilities
        new_list.sort(cmp_ass)
        if floor == None:  # Calculate threshold relative to best association
            floor = new_list[0].nu-self.mod.A_floor
        old_list = []
        for asn in new_list[:self.mod.Max_NA]:
            if asn.nu < floor:
                break
            old_list.append(asn)
        return old_list  # End of exhaustive()
    def Murty(self,      # ASSOCIATION4
              k_list,    # Observations indices in this cluster
              causes,    # Plausible explanations of each observation
              floor      # Minimum utility of associations to return
              ):
        """ A service routine for forward that finds the
        self.mod.Max_NA best associations by Murty's algorithm.
        """
        global hungary_count, hungary_time
        assert(len(k_list) == len(causes)),\
              'len(k_list)=%d, len(causes)=%d'%(len(k_list),len(causes))
        m = len(k_list) # Number of observations
        index_2_j = {}  # Map from cause index to j
        ij_2_cause = {} # Map from (i,j) to cause
        j_mult = {}     # Dict of causes that can explain many observations
        w = {}          # Dict of assignement weights
        n = 0           # Number of j values, ie, causes
        for i in xrange(m): # Loop over observations
            k = k_list[i]
            for cause in causes[i]:
                index = cause.index
                if not index_2_j.has_key(index):
                    index_2_j[index] = n
                    if index < 0:
                        j_mult[n]=True # Allow multiple assignments to FA
                    n += 1
                j = index_2_j[index]
                ij_2_cause[(i,j)] = cause
                assert(cause.type != 'target' or cause.m_t[-1] == k)
                w[(i,j)] = cause.R
        # Make each w[key] > (Max-Min) so all solutions have m links
        w_list = w.values()
        delta = 2*m*max(w_list) - (2*m+1)*min(w_list)
        for key in w.keys():
            w[key] = w[key] + delta
        #
        X = util.Hungarian(w,m,n,j_gnd=j_mult)
        util_max = 0
        for key in X.keys():
            util_max += w[key]
        if floor != None:
            if util_max < floor:
                return []
        else:  # Calculate threshold relative to best association
            floor = util_max-self.mod.A_floor
        ML = util.M_LIST(w,m,n,j_gnd=j_mult)
        # FixMe: doesn't count extra_invisible utilities
        ML.till(self.mod.Max_NA,floor)
        hungary_count += ML.H_count
        hungary_time += ML.stop_time-ML.start_time
        new_list = []
        for U,X in ML.association_list:
            # The seed assn is copy of self with empty tar_dict
            OK,new_A = self.Fork(CAUSE(),-1)
            new_A.tar_dict = {}
            new_A.h2c = {}
            for ij in X:
                i,j = ij
                k = k_list[i]
                cause = ij_2_cause[ij]
                OK, temp = new_A.Spoon(cause,k)
                if not OK:
                    break
            if not OK:
                continue
            new_list.append(new_A)
        return new_list     # End of Murty()
    def forward(self,       # ASSOCIATION4
                successors, # DB of associations and u's for the next time step
                y_t,        # Hits at this time
                t,
                k_list,     # Identify subset of y_t in this cluster
                floor
                ):
        """ For each plausible successor S of the ASSOCIATION self, enter the
        following into the successors DB 1. A key for the explanations
        that S gives for the observations. 2. The candidate successor
        association S. 3. The value of u'(self,S,t+1).  On entry,
        self.nu is correct for time t-1.
        """
        self.t = t # Make t available to other methods
        self.verify(None)
        m = len(k_list)
        assert(m > 0)       # Necessary in N_hat calulations below
        if k_list == (-1,): # No targets in self have visible children
            OK,seed = self.Fork(CAUSE(),-1)
            seed.tar_dict = {}
            seed.h2c = {}
            new_list = [seed]
        else:
            causes = []
            N_c =0    # Total number of causes for N_hat calculation
            # For each observation, make a list of plausible causes
            for k in k_list:
                if k < 0:
                    continue
                causes_k = []
                for check in self.cause_checks:
                    check(k,causes_k,y_t[k])
                causes.append(causes_k)
                N_c += len(causes_k)
            # List of plausible causes complete
            N_hat = (float(N_c)/m)**m  # Estimated number of associations
            if N_hat < self.mod.Murty_Ex:
                new_list = self.exhaustive(k_list,causes,floor)
            else:
                new_list = self.Murty(k_list,causes,floor)
            # FixMe: Made selection w/o knowing utilities required by
            # extra_forward, eg, extra_invisible
        if len(new_list) == 0:
            #print 'forward returning 0 new associations'
            return
        for asn in new_list:
            for method in self.extra_forward: # EG extra_invisible
                method(asn)
            asn.verify(k_list)  # FixMe delete this
        # Now each entry in new_list is a complete association that
        # explains k_list at time t.  Next propose the present
        # association as the predecessor for each association in
        # new_list.
        for asn in new_list:
            successors.enter(asn)
        return # End of forward, end of class ASSOCIATION4
class Cluster:
    """ A cluster is a set of targets and associations of those targets.
    Each association in a cluster should explain the same set of
    observations past and present.

    Methods:
       __init__()    Make self.As from arguments
       Append()     
       merge()
    """
    def __init__(self,            # Cluster
                 targets,         # Dict with targets as keys
                 asn              # Association that targets are from
                 ):
        """ Make first association and attach to self
        """
        self.As = []
        self.A_dict = {} # Key: mother, value: fragment.  For Append_D
        self.Append(targets,asn)
    def Append(self,            # Cluster
               targets,         # Dict with targets as keys
               asn              # Association that targets are from
               ):
        """ Create a new association in this cluster derived from the argument
        asn.  Copy targets specified in the argument 'targets'.
        """
        nu = 0.0
        for target in targets.keys():
            nu += target.R_sum
        new_a = asn.NewA(nu,asn.mod)
        for target,key in targets.items():
            assert(not new_a.tar_dict.has_key(key))
            new_a.Enter(target)
        self.As.append(new_a)
        assert(not self.A_dict.has_key(asn)),'Tried to append second'+\
            ' association from same mother association'
        self.A_dict[asn] = new_a # For Append_D
        return
    def Append_D(self,      # Fragment Cluster
                 tks,       # List of (t,k) tuples required in self
                 Mother     # Parent Cluster
                 ):
        """ Copy dead targets and FAs from As in Mother that each member of
        self.As needs in order to have a complete history.
        """
        in_tuple = list_2_flat(tks)
        for A_mother in Mother.As:
            if not self.A_dict.has_key(A_mother):
                continue
            A_fragment = self.A_dict[A_mother]
            if list_2_flat(A_fragment.Atks.keys()) == in_tuple:
                continue
            for tk in tks:
                if not A_mother.Atks.has_key(tk):
                    print 'A_mother is missing key',tk
                    print 'A_fragment.Atks.keys=',dict_2_tuple(A_fragment.Atks)
                    print '  A_mother.Atks.keys=',dict_2_tuple(A_mother.Atks)
                    print 'Dump A_mother'
                    A_mother.dump()
                    print 'Dump A_fragment'
                    A_fragment.dump()
                    raise RuntimeError
                if A_fragment.Atks.has_key(tk):
                    continue
                cause = A_mother.Atks[tk]
                if cause.type == 'FA':
                    A_fragment.FAs[tk] = cause
                    A_fragment.Atks[tk] = cause
                    continue
                if cause.type != 'target' and cause.type != 'dead_target':
                    raise RuntimeError,'Unknown cause type: %s'%cause.type
                for ttk in cause.tks.keys():
                    A_fragment.Atks[ttk] = cause
                if cause.type == 'target':
                    A_fragment.tar_dict[cause.index] = cause
                if cause.type == 'dead_target':
                    A_fragment.dead_targets[cause.index] = cause
            # Now A_fragment.tks has every tk in tks.  Check that
            # match is exact.
            if list_2_flat(A_fragment.Atks.keys()) != in_tuple:
                print 'dict_2_tuple(A_fragment.Atks)=',dict_2_tuple(
                    A_fragment.Atks)
                print '                     in_tuple=',in_tuple
                print 'Dump A_mother'
                A_mother.dump()
                print 'Dump A_fragment'
                A_fragment.dump()
                raise RuntimeError
        return # End of Append_D
    def dump(self #Cluster
             ):
        print 'Dumping a cluster with %d associations:'%len(self.As)
        for A in self.As:
            A.dump()
    def merge(self,     # Cluster
             other      # Cluster
               ):
        """ Merge other cluster into self
        """
        new_As = []
        for OA in other.As:
            for SA in self.As:
                NA = OA.NewA(OA.nu+SA.nu,SA.mod)
                NA.tar_dict = OA.tar_dict.copy()
                NA.tar_dict.update(SA.tar_dict)
                NA.dead_targets = OA.dead_targets.copy()
                NA.dead_targets.update(SA.dead_targets)
                NA.FAs = OA.FAs.copy()
                NA.FAs.update(SA.FAs)
                new_As.append(NA)
        self.As = new_As
        return # End merge()
    def re_nu_C(self,  # Cluster
                ):
        """ Call re_nu_A for each asn in self to bring asn.nu up to date, and
        verify that all associations explain the same history.
        """
        tk_tuple_0 = self.As[0].re_nu_A()
        for asn in self.As[1:]:
            tk_tuple = asn.re_nu_A()
            if __debug__:
                if tk_tuple == tk_tuple_0:
                    continue
                for tk,cause in self.As[0].Atks.items():
                    if not asn.Atks.has_key(tk):
                        print ('asn missing (%d,%d).  Dumping cause in '+\
                            'self.As[0]')%tk
                        cause.dump()
                        raise RuntimeError,'Histories fail to match'
                for tk,cause in asn.Atks.items():
                    if not self.As[0].Atks.has_key(tk):
                        print ('self.As[0].Atks missing (%d,%d).  Dumping '+\
                                   'cause in asn')%tk
                        cause.dump()
                        raise RuntimeError,'Histories fail to match'
                raise RuntimeError,'Failed to find mismatch.'
        return #End re_nu_C(), end class Cluster
class Cluster_Flock:
    """ A cluster flock is a set of clusters.  The key method,
    recluster(t), implements reorganization from clusters at time t-1
    to clusters for time t based on plausible target-observation links
    found by make_family().
    
    Variables:
       parents        Dict of targets; keys are target.index
       children       Dict of targets; keys are target.index
       k2par          Dict that maps observation index to dict of targets
       ks_and_pars    List of clusters each stored as a dict with keys
                        'ks' and 'tars'.  *['ks'] is a dict of the ks
                        and *['tars'] is a dict of the parents.
       parent_2_NI    Dict that maps parents to index of cluster in ks_and_pars
       old_clusters   List of Clusters
       dead_targets   List of dead targets that need not be in any cluster

    Methods:
       make_family()         recluster()
       find_clusters()       compat_check()
       
    """
    def __init__(self,               # Cluster_Flock
                 target_dict,        # Keys are targets
                 a                   # ASSOCIATION4
                 ):
        self.old_clusters = [Cluster(target_dict,a)]
        self.mod = a.mod
        self.dead_targets = []
    def make_family(self,  # Cluster_Flock
                    Yt,
                    t
                    ):
        """ Make each of the following:
        self.children
        self.parents
        self.k2par
        """
        self.parents = {}   # Dict of targets last time indexed by target.index
        self.k2par = {}     # Dict that maps Yt index to dict of parents
        k_list = range(len(Yt))
        self.mod.make_newts(Yt,t,k_list)
        for k in k_list:
            self.k2par[k] = {}
        for cluster in self.old_clusters:
            for a in cluster.As:
                a.make_children(Yt,t) # dead_targets made here
                for target in a.tar_dict.values():
                    self.parents[target] = target.index
                    for k in target.children.keys():
                        if k >= 0:
                            self.k2par[k][target] = target.index
        # Begin debugging code
        for k,d in self.k2par.items():
            for target,index in d.items():
                assert(target.children.has_key(k))
    def find_clusters(self,           # Cluster_Flock
                      len_y
                      ):
        """ Plausible children of each parent have been made.  Now find
        clusters of new observations using the following two maps:

        self.k2par: Map from Yt index to dict indexed by parent targets

        for each parent in self.parents.keys(), parent.children is a
              dict with keys that are indices of Yt

        Each observation that is too far from all targets is put in a
        cluster by itself even if it is close to other such
        observations.
        """
        old_pars = self.parents.copy() # Unclustered parents
        old_ks = dict(map (lambda x: (x,True),range(len_y)))
        # Need dicts of old targets and old k values so that
        # deleting doesn't change keys for remainders
        self.ks_and_pars = [] # Definitive list of clusters
        while len(old_ks) > 0:
            cluster_tar = {} # Initialize dict of targets in this cluster
            cluster_k = {old_ks.popitem()[0]:True}# Seed with a remaining k
            new_ks = cluster_k.copy()
            length = 0
            while (len(cluster_k) + len(cluster_tar)) > length:
                # Stop when iteration doesn't grow cluster
                length = len(cluster_k) + len(cluster_tar)
                new_tars = {}
                for k in new_ks.keys():
                    for parent in self.k2par[k].keys():
                        cluster_tar[parent] = parent.index
                        new_tars[parent] = True
                        if old_pars.has_key(parent):
                            del old_pars[parent]
                new_ks = {}
                for target in new_tars.keys():
                    for k in target.children.keys():
                        if k < 0:
                            continue
                        if old_ks.has_key(k):
                            cluster_k[k] = True
                            new_ks[k] = True
                            del old_ks[k]
                            continue
                        if cluster_k.has_key(k):
                            continue
                        #### Begin diagnostic printing ####
                        print '###Suspect trouble in find_clusters.'
                        print 'k=%d, self.k2par entries:'%k
                        for i,d in self.k2par.items():
                            print '  %d:'%i,d.keys()
                        print 'self.parents[*].children keys:'
                        for key,target in self.parents.items():
                            c_keys = target.children.keys()
                            print '  %d,%d:'%(key,target.index),c_keys
                        print 'Previous clusters:'
                        for kp_dict in self.ks_and_pars:
                            print kp_dict['ks'].keys(),\
                                  kp_dict['pars'].values()
                        print 'In this cluster far:'
                        print '  cluster_k=',cluster_k.keys(),\
                              'and cluster_tar=',cluster_tar.keys()
                        print ('Trouble is that neither old_ks nor cluster_k'+\
                              ' has key %d\n')%k,'old_ks=',old_ks.keys()
                        raise RuntimeError,'###Suspect trouble in find_clusters'
                        #### End diagnostic printing
            self.ks_and_pars.append({'ks':cluster_k,'pars':cluster_tar})
        # The targets that remain in old_pars have only invisible
        # children.  Put them in an additional cluster.
        if len(old_pars) > 0:
            self.ks_and_pars.append({'ks':{-1:True},'pars':old_pars})
        self.parent_2_NI = {} # Map from parents to new cluster indices NI
        for NI in xrange(len(self.ks_and_pars)):
            for parent in self.ks_and_pars[NI]['pars'].keys():
                self.parent_2_NI[parent] = NI
        return # End of find_clusters
    def compat_check(self,     # Cluster_Flock
                     asn,      # Proposed association
                     tk_2_NI,  # Map from (t,k) tuples to NI (new index)
                     tk_2_dead # lists of dead targets indexed by (t,k)
                     ):
        """ Check compatibility of asn.  Keys in tk_2_NI are historical hits
        that must end up in the new cluster NI.  Keys in tk_2_dead may
        end up in the dead targets list of the cluster self.
        """
        proposed_tk_2_NI = tk_2_NI.copy()
        proposed_tk_2_dead = tk_2_dead.copy()
        # Check that history of live targets compatible.  Check dead
        # targets later.
        for target in asn.tar_dict.values():
            NI = self.parent_2_NI[target]
            for tk in target.tks.keys():
                if not tk_2_NI.has_key(tk):
                    proposed_tk_2_NI[tk] = NI
                else:
                    if not tk_2_NI[tk] == NI:
                        return (False,tk_2_NI,tk_2_dead)
        # Begin check that no dead target connects two NIs
        #   Step 1: Update proposed_tk_2_dead with asn.dead_targets
        for target in asn.dead_targets.values():
            for tk in target.tks.keys():
                if proposed_tk_2_dead.has_key(tk):
                    proposed_tk_2_dead[tk].append(target)
                else:
                    proposed_tk_2_dead[tk] = [target]
        #   Step 2: For each target in proposed_tk_2_dead that
        #   intersects proposed_tk_2_NI, use target to augment
        #   proposed_tk_2_NI and remove the target from
        #   proposed_tk_2_dead.
        for tk,NI in proposed_tk_2_NI.items():
            if proposed_tk_2_dead.has_key(tk):
                for target in proposed_tk_2_dead[tk]:
                    for dt_tk in target.tks.keys():
                        if proposed_tk_2_NI.has_key(dt_tk):
                            if proposed_tk_2_NI[dt_tk] != NI:
                                return (False,tk_2_NI,tk_2_dead)
                        else:
                            proposed_tk_2_NI[dt_tk] = NI
                del proposed_tk_2_dead[tk]
        return (True,proposed_tk_2_NI,proposed_tk_2_dead) # End compat_check
    def recluster(self,   # Cluster_Flock
                  t,
                  Yt
                  ):
        """ On the basis of the clusters of observations and targets
        that find_clusters() has identitied, recluster() fragments
        self.old_clusters and merges the parts to form
        self.new_clusters.  Each new cluster explains a unique set of
        observations up to the present time t and all associations in
        the cluster must explain the same set of observations.  The
        targets in the associations in the new clusters are _parent
        targets_, ie, they have not incorporated observations at time
        t.
        """
        fragmentON = {} # Fragment clusters indexed by (Old index, New index)
        for OI in xrange(len(self.old_clusters)):
            cluster = self.old_clusters[OI]
            tk_2_NI = {}   # Map from (t,k) tuple to New index
            tk_2_dead = {} # Dead target candidates for self.dead_targets
            cluster.As.sort(cmp_ass) # Sort the associations by utility
            fragmentON[OI] = {}
            # Break each association into fragments if it is
            # compatible with those fragmented so far.  Put pieces in
            # fragmentON[OI]
            for asn in cluster.As:
                OK,tk_2_NI,tk_2_dead = self.compat_check(asn,tk_2_NI,tk_2_dead)
                if not OK:
                    delta = cluster.As[0].nu - asn.nu
                    if delta < 0.4: #FixMe use adjustable threshold
                        close_calls.append(('At t=%d in recluster(), dropped'+\
                          ' branch association that is off by %5.3f')%(t,delta))
                    continue # Drop this association and try the next
                # Copy live targets to cluster fragments ie, fragmentON[OI][NI]
                frag_dict = {}
                for target in asn.tar_dict.values():
                    NI = self.parent_2_NI[target]
                    if frag_dict.has_key(NI):
                        frag_dict[NI][target] = target.index
                    else:
                        frag_dict[NI] = {target:target.index}
                for NI,tar_dict in frag_dict.items():
                    if fragmentON[OI].has_key(NI):
                        fragmentON[OI][NI].Append(tar_dict,asn)
                    else:
                        fragmentON[OI][NI] = Cluster(tar_dict,asn)
            # From each old association in cluster, copy necessary
            # dead targets and FAs to appropriate associations in
            # fragments.
            NI_2_tk = {}
            for tk,NI in tk_2_NI.items():
                if not NI_2_tk.has_key(NI):
                    NI_2_tk[NI] = []
                NI_2_tk[NI].append(tk)
            for NI in fragmentON[OI].keys():
                fragmentON[OI][NI].Append_D(NI_2_tk[NI],cluster)
            # Save unused dead targets from best old association for backtrack
            for target in cluster.As[0].dead_targets.values():
                if not tk_2_NI.has_key(target.tks.keys()[0]):
                    self.dead_targets.append(target)
        # End loop over OI.  Next merge fragments into new clusters
        new_clusters = {}
        for OI in fragmentON.keys():
            for NI in fragmentON[OI].keys():
                if not new_clusters.has_key(NI):
                    new_clusters[NI] = fragmentON[OI][NI]
                else:
                    new_clusters[NI].merge(fragmentON[OI][NI])
        # Prune merged clusters
        for cluster in new_clusters.values():
            cluster.As.sort(cmp_ass) # Sort the associations by utility
            floor = cluster.As[0].nu - self.mod.A_floor
            NF = min(len(cluster.As),self.mod.Max_NA)
            for Nf in xrange(1,min(len(cluster.As),self.mod.Max_NA)):
                if cluster.As[Nf].nu < floor:
                    NF = Nf
                    break
            cluster.As = cluster.As[:NF]
        # Add clusters for observations that no target could have caused
        for NI in xrange(len(self.ks_and_pars)):
            if len(self.ks_and_pars[NI]['pars']) == 0:
                a = self.mod.ASSOCIATION(0.0,self.mod)
                new_clusters[NI] = Cluster({},a)
                assert len(self.ks_and_pars[NI]['ks']) == 1,('len(self.ks_'+\
                      'and_pars[NI]["ks"])=%d')%len(self.ks_and_pars[NI]['ks'])
        rv = [] # Return value is [[cluster,k_list],[cluster,k_list],...]
        self.old_clusters = []
        for NI in new_clusters.keys():
            k_list = self.ks_and_pars[NI]['ks'].keys()
            cluster = new_clusters[NI]
            cluster.re_nu_C()
            rv.append((cluster,tuple(k_list)))
            self.old_clusters.append(cluster)
        return rv # End of recluster.  End of class Cluster_Flock
class MV4:
    """ Model version four.  A state consists of the following for
    each cluster: Association; Locations; and Visibilities; (and the
    derivable N_FA), ie, s = (M,X,V,N_FA)
    """
    def __init__(self,                 # MV4
         N_tar = 3,                    # Number of targets
         A = [[0.81,1],[0,.81]],       # Linear state dynamics
         Sigma_D = [[0.01,0],[0,0.4]], # Dynamical noise
         O = [[1,0]],                  # Observation projection
         Sigma_O = [[0.25]],           # Observational noise
         Sigma_init = None,            # Initial state distribution
         mu_init = None,               # Initial state distribution
         MaxD = 5.0,                   # Threshold for hits
         Max_NA = 20,                  # Maximum associations per cluster
         A_floor = 25.0,               # Drop if max-ass.utility > A_floor
         Murty_Ex = 100,               # Use exhaustive, if #asn < Murty_Ex
         T_MM = 5,                     # If two targets match hits T_MM drop 1
         PV_V=[[.9,.1],[.2,.8]],       # Visibility transition matrix
         Lambda_new=0.05,              # Avg # of new targets per step
         Lambda_FA=0.3,                # Avg # of false alarms per step
         IL = 8                        # Invisible lifetime
        ):
        self.ASSOCIATION = ASSOCIATION4
        self.TARGET = TARGET4
        self.N_tar = N_tar
        self.A = scipy.matrix(A)
        dim,check = self.A.shape
        assert(dim == check),("A should be square but it's dimensions are "+\
           "(%d,%d)")%(dim,check)
        self.Id = scipy.matrix(scipy.identity(dim))
        self.Sigma_D = scipy.matrix(Sigma_D)
        self.O = scipy.matrix(O)
        dim_O,check = self.O.shape
        assert(check == dim),("Shape of O=(%d,%d) not consistent with shape "+\
          "of A=(%d,%d))")%(dim_O,check,dim,dim)
        self.Sigma_O = scipy.matrix(Sigma_O)
        self.log_det_Sig_D = math.log(scipy.linalg.det(self.Sigma_D))
        self.log_det_Sig_O = math.log(scipy.linalg.det(self.Sigma_O))
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
        self.log_min_pd = -(MaxD*MaxD + dim_O*math.log(2*math.pi) +
                            self.log_det_Sig_O) /2.0
        """ This is the log of the minimum conditional probability density for
        a single observation given some cause.  We consider
        observation/cause pairs that have probability densities lower
        than this implausible.  The control over this threshold is by
        MaxD which we think of as ''MaxD sigmas'', but to translate
        MaxD to probability density requires a scale which we take
        from Sigma_O.  Since all actual Sigmas are bigger than
        Sigma_O, the threshold accepts a smaller region than ''MaxD
        sigmas''.
        """
        self.Max_NA = Max_NA
        self.A_floor = A_floor
        self.Murty_Ex = Murty_Ex
        self.T_MM = T_MM
        self.PV_V = scipy.matrix(PV_V)
        self.Lambda_FA = Lambda_FA # Average number of false alarms per frame
        Sigma_FA = self.O*self.Sigma_init*self.O.T + self.Sigma_O
        self.Sigma_FA = Sigma_FA
        self.log_FA_norm = math.log(Lambda_FA)\
                           - math.log(scipy.linalg.det(Sigma_FA))/2
        self.Sigma_FA_I = scipy.linalg.inv(Sigma_FA)
        self.Lambda_new = Lambda_new # Average number of new targets per frame
        self.log_new_norm = math.log(Lambda_new) # For make_newts
        self.Invisible_Lifetime = IL # If invisible IL times in a row
                                     # target dies
        return # End of __init__
    def make_newts(self, # MV4.  Make new targets
                   Yts,  # List of ys for current time
                   t,    # Current time
                   k_list
                   ):
        self.newts = {}
        for k in k_list:
            self.newts[k] = self.target_0.KF(Yts[k],k,t)
            self.newts[k].R += self.log_new_norm + self.log_det_Sig_D/2
            # FixMe: Sig_D correction because KF has forecast step?
            self.newts[k].R_sum = self.newts[k].R
    def decode_prune(self,    #MV4
                     t,
                     new_As   # List of associations
                     ):
        """ Find targets that match for self.T_MM steps and remove
        associations from new_As that have inferior matching targets.
        """
        assert(len(new_As) > 0),'t=%d: decode_prune() called with empty list'%t
        if t <= self.T_MM: # Or t%self.T_NN != 0?  Could prune periodically
            return new_As[:self.Max_NA]
        # Find targets that match for self.T_MM steps
        shorts = {}
        keepers = {}
        for k in xrange(len(new_As)):
            keepers[k] = new_As[k] # copy list to dict
            for target in new_As[k].tar_dict.values():
                if len(target.m_t) < self.T_MM or \
                       target.m_t[-1] < 0 or target.m_t[-2] < 0:
                    continue # Don't kill new or invisible targets
                short_key = tuple(target.m_t[-self.T_MM:])
                if not shorts.has_key(short_key):
                    shorts[short_key] = [[],[]]
                shorts[short_key][0].append(k)
                shorts[short_key][1].append(new_As[k].nu)
        # Remove associations that have inferior matching targets
        for short in shorts.values():
            i = util.argmax(short[1])
            nu_max = short[1][i]
            del short[0][i]
            for k in short[0]:
                if keepers.has_key(k):
                    if nu_max - keepers[k].nu < 0.4: # Fixme need parameter
                        close_calls.append(
             "t=%2d: decode_prune() dropped association. nu off by %5.3f"%(
              t,nu_max - keepers[k].nu))
                    del keepers[k]
        new_As = keepers.values()
        if len(new_As) < 1:
            raise RuntimeError,'decode_prune() killed all As at t=%d'%t
        return new_As[:self.Max_NA]
    
    def decode_init(self, # MV4
                    Ys): # Ys is used by subclass
        self.target_0 = self.TARGET(self,    # Parent model
                           [0],              # m_t: history of indices
                           [self.mu_init],   # History of means
                           [self.Sigma_init],# History of coavariances
                           0.0,              # Most recent residual
                           None,             # y value
                           {}                # Dict of (t,k) pairs explained
                                    )
        return ([self.ASSOCIATION(0.0,self)],0) # First A and first t
    
    def decode_forward(self,   # MV4
                       Ys,     # Observations. Ys[t][k]
                       old_As, # [Initial association] or nub
                       t_first,# Starting t for iteration
                       analysis = False
                       ):
        global close_calls, hungary_count, hungary_time
        close_calls = []
        m_dict = {}
        for target in old_As[0].tar_dict.values():
            m_dict[target] = target.index
        flock = Cluster_Flock(m_dict,    # Dict of targets
                              old_As[0]  # Association
                              )
        TC_last = Child_Target_Counter
        Time_last = time.time()
        for t in xrange(t_first,len(Ys)):
            hungary_count = 0
            hungary_time = 0
            flock.make_family(Ys[t],t)      # Make parent and child targets
            flock.find_clusters(len(Ys[t])) # Identify clusters of observations
            for cluster,k_list in flock.recluster(t,Ys[t]):
                # k_list is current observations that cluster explains
                successors = SUCCESSOR_DB()
                floor = None
                for asn_No in xrange(len(cluster.As)):
                    asn = cluster.As[asn_No]
                    asn.forward(successors,Ys[t],t,k_list,floor)
                    # recluster makes newts and forward
                    # checks/incorporates newts and FAs
                    suc_max = successors.maxes()
                    # suc_max is sorted list of successors from best
                    # predecessors
                    if len(suc_max) == 0:
                        continue
                    if len(suc_max) > 40*self.Max_NA:
                        print ('Stop: hungary_count=%d, asn_No=%d,' + \
                          ' suc_count=%d')%(hungary_count, asn_No,len(suc_max))
                        break
                    if len(suc_max) > self.Max_NA:
                        floor = suc_max[self.Max_NA].nu
                    if suc_max[0].nu-self.A_floor > floor:
                        floor = suc_max[0].nu-self.A_floor
                if len(cluster.As) > 0 and len(suc_max) == 0:
                    print 'Ys[t]=',Ys[t],'No associations explain k_list='\
                          ,k_list
                    print ' Clusters in flock.ks_and_pars:'
                    for kp_dict in flock.ks_and_pars:
                        print kp_dict['ks'].keys(),kp_dict['pars'].values()
                    cluster.dump()
                    raise RuntimeError,('forward killed all %d associations'+\
                          ' in cluster.  Try bigger MaxD')%len(cluster.As)
                cluster.As=self.decode_prune(t,suc_max)
                #cluster.As=suc_max[:self.Max_NA] # For debug w/o prune
            print ('t %2d: %3d targets, %2d clusters, %5.2f '+\
                  'seconds, H_time %5.2f, H_count %3d')%(t,
                   Child_Target_Counter-TC_last, len(flock.old_clusters),
                   time.time()-Time_last,hungary_time,hungary_count)
            if Child_Target_Counter-TC_last > 0:
                for cluster in flock.old_clusters:
                    print '   Cluster has %d associations with %d targets'%(
                          len(cluster.As),len(cluster.As[0].tar_dict))
            TC_last = Child_Target_Counter
            Time_last = time.time()
        # Find the best associations at final time
        empty = self.ASSOCIATION(0,self)
        A_union = empty
        for cluster in flock.old_clusters:
            As = cluster.As
            if len(As) == 0:
                print 'At end of decode_forward, found an empty cluster'
                continue
            R = scipy.zeros(len(As))
            for i in xrange(len(As)):
                R[i] = As[i].nu
            A_union.join(As[R.argmax()])
        # Put dead targets from flock into A_union
        for target in flock.dead_targets:
            A_union.dead_targets[target.index] = target
        return (A_union,len(Ys),close_calls)

    def decode_back(self,   # MV4
                    A_best, # Best association at final time
                    T):
        """ Backtrack from best association at final time to get MAP
        trajectories.
        """
        targets_times = []
        for target in A_best.dead_targets.values():
            t_i = target.last-len(target.m_t)
            t_f = target.last-self.Invisible_Lifetime
            for List in target.m_t,target.mu_t,target.Sigma_t:
                del(List[-self.Invisible_Lifetime:])
            targets_times.append([target,t_i,t_f])
        for target in A_best.tar_dict.values():
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
    def decode(self, # MV4
               Ys,   # Observations. Ys[t][k] is the kth hit at time t
               analysis = False # A debugging option
               ):
        """Return MAP state sequence """
        A,t_0 = self.decode_init(Ys)
        A_best,T,close_calls = self.decode_forward(Ys,A,t_0,analysis=analysis)
        d,y_A = self.decode_back(A_best,T)
        return (d,y_A,A_best.nu,close_calls)
    ############## Begin simulation methods ######################
    def step_state(self, # MV4
                   state, zero_s):
        epsilon = util.normalS(zero_s,self.Sigma_D) # Dynamical noise
        return self.A*state + epsilon
    def observe_states(self, states_t, v_t, zero_y):
        """ Create observations for this time step from a list of
        states (states_t) and a list of visibilities (v_t).
        """
        assert(len(v_t) == len(states_t))
        v_states = []
        for k in xrange(len(v_t)):
            if v_t[k] is 0:
                v_states.append(states_t[k])
        y_t = []
        for state in v_states:
            eta = util.normalS(zero_y,self.Sigma_O) # Observational noise
            y_t.append(self.O * state + eta)
        return y_t
    def shuffle(self, # MV4
                things_i):
        """ return a random permutation of things_i 
        """
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
        """ Take observability for a single target from previous time
        step and return visibility for current time
        """
        pv = self.PV_V[v,0]
        if pv > random.random():
            return 0
        else:
            return 1
    def sim_init(self, # MV4
                 N):
        """ Initialize states and visibilities for N targets.
        """
        x_j = []
        v_j = []
        for j in xrange(N):
            x_j.append(util.normalS(self.mu_init,self.Sigma_init))
            v_j.append(0)
        return (x_j,v_j,0)
    def step_count(self, x, v, c, zero_x):
        """ Step x, v, and c forward one time step.
        x = target vector x
        v = visibility
        c = count of number of times that v != 0 (not visible)
        """
        x = self.step_state(x,zero_x)
        v = self.step_vis(v)
        if v is 0:
            c = 0
        else:
            c += 1
        return (x,v,c)
    def run_state(self,x,v,zero_x,t_0,T):
        """ Given x and for an initial time t_0, do the following:

        1. Propagate x and v forward till either the state dies or T-1
              is reached,

        2. Put the simulated x's and v'x in lists that run from 0 to T.
        """
        IL = self.Invisible_Lifetime
        s_k = T*[None]
        v_k = T*[None]
        c = 0
        for t in xrange(t_0,T):
            s_k[t] = x
            v_k[t] = v
            x,v,c = self.step_count(x,v,c,zero_x)
            if c >= IL:
                s_k[t-IL+1:t+1] = IL*[None]
                v_k[t-IL+1:t+1] = IL*[None]
                break
        return (s_k,v_k)
    def simulate(self, # MV4
                 T):
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
        rv = TARGET1(*args,**kwargs)
        rv.T0 = self.T0
        return rv
    def make_children(self, # TARGET1
           y_t,             # list of hits at time t
           t                # Present time
                      ):
        """ Like TARGET4 make_children but no invisible targets
        """
        if not self.children is None and len(self.children) > 0:
            return # make_children already called for this target
        self.forecast()
        self.children = {}
        mlpd = self.mod.log_min_pd
        for k in xrange(len(y_t)):
            if mlpd > self.utility(y_t[k])[0]:
                continue                 # Candidate utility too small
            self.children[k] = self.update(y_t[k],k,t)
            # update() calls utility second time.  Possible time saving
            assert(self.children[k].m_t[-1]==k)
    def utility(self,   # TARGET1
                y,R0=0.0):
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
        self.type='ASSOCIATION3'
    def NewA(self, *args,**kwargs):
        return ASSOCIATION3(*args,**kwargs)
    def make_children(self, # ASSOCIATION3
                      y_t, t ):
        """ No dead targets for ASSOCIATION3. """
        for target in self.tar_dict.values():
            target.make_children(y_t,t)
class ASSOCIATION2(ASSOCIATION3):
    """ Allow invisibles, ie, targets that fail to generate hits. """
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.cause_checks = [self.check_targets]
        self.type='ASSOCIATION2'
    def NewA(self, *args,**kwargs):
        return ASSOCIATION2(*args,**kwargs)
class ASSOCIATION1(ASSOCIATION3):
    """ No extra_forward methods.  Targets and only targets generate hits. """
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.extra_forward = []
        self.cause_checks = [self.check_targets]
        self.type='ASSOCIATION1'
    def NewA(self, *args,**kwargs):
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
        self.target_0 = self.TARGET(self,[0],[self.mu_init], # mod,m_t,mu_t
                             [self.Sigma_init],0.0,None,{} # Sigma_t,R,y,tks
                                    )
        T = len(Ys)
        partial = self.ASSOCIATION(0.0,self,t=0)
        for k in xrange(self.N_tar):
            target_k = self.target_0.KF(Ys[0][k],k,0)
            partial.Spoon(target_k,k)
        return ([partial],1)
    def simulate(self, #MV3
                 T):
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
    def simulate(self, # MV2
                 T):
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
    def simulate(self, # MV1
                 T):
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
    
class ASS_ABQ(ASSOCIATION4):
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.type='ASS_ABQ'
        self.cause_checks = [self.check_targets, #No self.check_FAs for ABQ
                             self.check_newts]
        # self.extra_forward = [] need invisibles to kill targets
    def NewA(self, *args,**kwargs):
        return ASS_ABQ(*args,**kwargs)
class MV_ABQ(MV4):
    """ Like MV4 but associations are ASS_ABQ.
    """
    def __init__(self,**kwargs):
        MV4.__init__(self,**kwargs)
        self.Invisible_Lifetime=1
        self.ASSOCIATION = ASS_ABQ
        self.PV_V = scipy.matrix([[.9,.1],[1e-7,1-1e-7]])
        self.Lambda_FA=0.000001 # No FAs in ABQ
        self.Lambda_new=0.5
    def dump(self  # MV_ABQ
             ):
        print 'MV_ABQ model dump: N_tar=%d, A=\n'%(
            self.N_tar),self.A
        print 'Sigma_D=\n',self.Sigma_D
        print 'O=\n',self.O
        print 'Sigma_O=\n',self.Sigma_O
        print 'mu_init=\n',self.mu_init
        print 'Sigma_init=\n',self.Sigma_init
        print 'Sigma_FA=\n',self.Sigma_FA
        print 'PV_V=\n',self.PV_V
        print 'Lambda_new=%5.3f,  Lambda_FA=%5.3f'%(self.Lambda_new,
                                                    self.Lambda_FA)
        return

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
            print '\nt=%2d'%t,
            for key in ('pred_count','succ_count','new_count','old_count',
                        'child_count'):
                print '%s=%-6d'%(key,record[key]),
        print '\n'
        for t in ts:
            print 't=%2d, index_count='%t,self.records[t]['index_count']
if __name__ == '__main__':  # Test code
    import time
    # The following parameters are for 4-d MV4 model with 2-d observation
    A = [[.95,1,0,0],[0,.95,0,0],[0,0,.95,1],[0,0,0,.95]]
    Sigma_D = [[0.01,0,0,0],[0,0.04,0,0],[0,0,0.01,0],[0,0,0,0.04]]
    O = [[1,0,0,0],[0,0,1,0]]
    Sigma_O = [[0.0025,0],[0,0.0025]]
    PV_V=[[.95,0.05],[0.2,.8]]
    # Loop over models (all except MV4_4d have 2-d states and 1-d observations)
    for pair in (
       [MV1(N_tar=4),'MV1',1.31],
       [MV2(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),'MV2',0.15],
       [MV3(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),'MV3',0.36],
       [MV4(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]]),'MV4',0.65],
       [MV4(N_tar=12,PV_V=PV_V,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,
            MaxD=6.0,Lambda_FA=.01),'MV4_4d',0.61],
       ):
        random.seed(3)
        scipy.random.seed(3)
        Target_Counter=0
        M=pair[0]
        print '%s: Begin simulate'%pair[1]
        y,s = M.simulate(10)
        print 'Begin decode; expect %4.2f seconds on AMD Sempron 3000'%pair[2]
        ts = time.time()
        A = analysis()
        d,y_A,nu,close_calls = M.decode(y,analysis=A)
        print '     Elapsed time =  %4.2f seconds.  '%(time.time()-ts)
        print 'len(y)=',len(y), 'len(s)=',len(s),'len(d)=',len(d)
        dim_s = s[0][0].shape[0]
        dim_y = y[0][0].shape[0]
        for t in xrange(len(s)):
            print 't=%d '%t+4*dim_y*' '+'y'+(1+5*dim_y+2*dim_s)*' '+\
                  's'+(2+6*dim_s)*' ' +'d'
            for k in xrange(max(len(s[t]),len(d))):
                try:
                    print ' k=%d '%k + dim_y*' %5.2f '%tuple(y[t][k][:,0]),
                except:
                    print ' k=%d '%k+dim_y*'       ',
                try:
                    print ('(%5.2f'+(dim_s-1)*' %5.2f')%tuple(
                        s[t][k][:,0])+') ',
                except:
                    print (2+6*dim_s)*' ',
                try:
                    print ('(%5.2f'+(dim_s-1)*' %5.2f')%tuple(d[k][t][:,0])+')'
                except:
                    print ' '
        print '\n'
    A.dump()

#---------------
# Local Variables:
# eval: (python-mode)
# End:
