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
on ASSOCIATION4.  There are four target classes:

TARGET5   For MV5.  Includes IMM

TARGET4   For MV4. Creation at any time.  Variable visibility.  Dies after
          invisible lifetime.

TARGET3   For MV2 and MV3.  Variable visibility.  Exists from t=0 to end.

TARGET1   For MV1.  Always visible.

Here are the family trees most of the classes in this file:

CAUSE_FA -> TARGET4 -> TARGET1

ASSOCIATION4 -> ASSOCIATION3 -> ASSOCIATION2 -> ASSOCIATION1

MV4 -> MV3 -> MV2
       MV3 -> MV1

FixMe: To do:

1. Why is re_nu_A necessary in exhaustive and Murty?  Something about Atks.

2. Flag likely errors correctly.  Places to think about:

   TARGET4.make_children
   ASSOCIATION4.join
   Cluster.__init__         What is the right way to apportion utility
   Cluster.Append           What is the right way to apportion utility
   Cluster_Flock.recluster  First fragment may not have best utility
   Cluster_Flock.recluster  Consistency check should include dead_targets
   Cluster_Flock.recluster  For which observations should I make newts

3. When I take out the argmax from SUCCESSOR_DB.maxes, Murty and
   exhaustive give different results.  I don't understand why.

4. I don't understand how serious the apporximation is of having
   argmax in SUCCESSOR_DB.maxes.

"""
import numpy, scipy, scipy.linalg, random, math, util, time
close_calls = None     # Will be list of flags of likely errors
Target_Counter = 1
Child_Target_Counter = 0
hungary_count = 0
hungary_time = 0
Murty_calls = 0
forgotten_utility = 0
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
    def __init__(self, y, t, k, Sigma_I, mod ):
        self.type = 'FA'
        self.index = -1
        QF = -float(y.T*Sigma_I*y/2)
        CC = math.log(mod.Lambda_FA) # Creation Cost
        norm = -math.log(scipy.linalg.det(mod.Sigma_FA))/2
        self.R = norm + CC + QF
        self.R_sum = self.R
        self.t = t
        self.k = k
        self.tks = {(t,k):self}
        #print 'Utility of FA is   %5.2f, QF=%5.2f, CC=%5.2f, norm=%5.2f'%(self.R,
        #              QF, CC, norm)
    def dump(self, #FA
             ):
        print 'Dumping %s: R=%6.2f, (t,k)=(%d,%d)'%(self.type,self.R,
                                                    self.t,self.k)
        return
class TARGET5(CAUSE):
    """A TARGET is a possible moving target, eg, a car.  I (Fraser)
    wrote this class after writing TARGET4 in order to implement IMM's
    (interacting multiple models) at the suggestion of Dale Klamer.
    The multiple models will be discrete states of a HMM.  After this
    class works, I may make TARGET4 a subclass.

    I use a single _ to remind myself that I should only use these
    variables inside the TARGET* classes.  If I had used __,
    subclasses could not access them.

    Private variables (t for time, i for IMM state):
        _mod              Model, eg MV4.  Provides access to parameters
        _mu_t             List of previous means. mu_t[t][i]
        _Sigma_t          List of previous covariances
        _invisible_count  Count of sequential hit failures
        _last             For dead target, time corresponding to mu_t[-1]
        _ij_max           unravel_index(D_nu.argmax(),D_nu.shape)
        _nu               Utility of best path to i is nu[i]
        _best             Best predecessor for i at t is best[t][i]
        
    The following private arrays are indexed by a tuple (ij) where i
    is the predecessor IMM state and j is the successor IMM state:
                 Calculated in forecast()
        _mu_fs            The forecast state mean
        _Sigma_fs         The forecast state covariance
        _mu_fO            The forecast observation mean
        _Sigma_fOI        The forecast observation inverse covariance
        _K                Kalman gain matrix
        _Sigma_us0        The updated state covariance if visible
                 Calculated in utility()
        _mu_us            The updated state mean
        _Sigma_us         The updated state covariance
        _D_nu             The updated utility of the step from i to j
        
    Externally accessed variables:
        T0               Time that corresponds to mu_t[0] (mostly private)
        R_sum            Max of nu
        R                R_sum(t) - R_sum(t-1)
        tks              Dict. key=(time,index), value=self
        index            Unique ID
        children         Dict indexed by hit with child targets as values
        type             'target' or 'dead_target'
        m_t              List of previous hits explained.  m_t[-1] is
                             index of hit at present time.

    Private methods:
        _forecast      Forecast state and observation
        _utility       Update and calculate utility of y
        _spawn         Create child target. Preceded by forecast() utility(y)
        _New           Create a new target of this class
        _kill          'target' --> 'dead_target' after invisible too long
        
    Public methods:
        backtrack     Calculate and return MAP trajectory
        Launch        Launch a new target for observation y
        make_children Make plausible children from list of ys
        dump          For debugging
        target_time
        
    """
    def __init__(self,             # TARGET5
                 mod=None,         # Parent model
                 par_tar=None,     # Parent target
                 copy_index=False, # Copy index from par_tar
                 nu=None,          # Vector of nu values
                 best=None,        # Vector of best predecessors
                 tk=None           # (time, hit index)
                 ):
        """ Can create a new TARGET5 instance in each of the following modes:
        1. Make target_0 from info in parent model called by MV*.decode_init
        2. New target by applying target_0 to a hit
        3. Spawn a target from an existing target with history
        """
        global Target_Counter, Child_Target_Counter
        # Determine mode, ie source of initialization information
        mode = None
        if mod != None and par_tar == None and copy_index == False:
            mode = 1
            N = len(mod.IMM.mus)
        if mod == None and par_tar != None and copy_index == False:
            mode = 2
            N = len(par_tar._mod.IMM.mus)
        if mod == None and par_tar != None and copy_index == True:
            mode = 3
            N = len(par_tar._mod.IMM.mus)
        assert (mode != None),"Inconsistent arguments in TARGET5.__init__()"
        # Do initialization that is same for all modes
        self.type = 'target'
        self._last = None
        self._invisible_count = 0
        self.tks = {}
        # The following ugly lines initialize several NxN lists of lists
        self._mu_fs, self._Sigma_fs, self._mu_fO, self._Sigma_fOI, \
                   self._K, self._mu_us, self._Sigma_us, \
                   self._Sigma_us0 = ([],[],[],[],[],[],[],[])
        for A in (self._mu_fs, self._Sigma_fs, self._mu_fO,
                  self._Sigma_fOI, self._K, self._mu_us, self._Sigma_us,
                  self._Sigma_us0):
            for i in xrange(N):
                A.append(N*[None])
        # Begin mode dependent initialization
        if mode == 1: # Make target_0
            self._mod = mod
            self._mu_t = [mod.IMM.mus]
            self._Sigma_t = [mod.IMM.Sigmas]
            self._nu = mod.IMM.nu_0
            self.index = 0
            self.m_t = []
            self.R_sum = 0
            self.R = 0
            return
        self._mod = par_tar._mod
        t,k = tk
        mu = []
        Sigma = []
        for j in xrange(N):
            mu.append(par_tar._mu_us[best[-1][j]][j])
            Sigma.append(par_tar._Sigma_us[best[-1][j]][j])
        if mode == 2: # target_0 + y => new target
            self._mu_t = [mu]
            self._Sigma_t = [Sigma]
            self.T0 = t
            self._best = []
            self.index = Target_Counter
            Target_Counter += 1
            self.m_t = [k]
        if mode == 3: # old target + y => new target
            self._mu_t = par_tar._mu_t + [mu]
            self._Sigma_t = par_tar._Sigma_t+ [Sigma]
            if k < 0:
                self._invisible_count = par_tar._invisible_count + 1
            else:
                self._invisible_count = 0
            Child_Target_Counter += 1   # Inc child counter for debugging
            self.T0 = par_tar.T0
            self._best = best
            self.tks = par_tar.tks.copy()
            self.index = par_tar.index
            self.m_t = par_tar.m_t + [k]
        # Start common code for mode 2 and 3
        self.R_sum = max(nu)
        self.R = self.R_sum - par_tar.R_sum
        self._nu = nu
        if k >= 0:                # Don't enter invisibles into tks
            self.tks[tk] = self
        self.children = None
        return # End of __init__
    def dump(self #TARGET5
             ):
        print 'Dumping TARGET5 %s #'%self.type,self.index
        print '\nm_t=',self.m_t
        tks = self.tks.keys()
        tks.sort()
        print '  tks=',tks
        print '             T0=%d, index=%d, len(mu_t)=%d t_last='%(self.T0,
               self.index,len(self._mu_t)),self._last
        print '   invisible_count=%d, R=%5.3f, R_sum=%5.3f'%(
            self._invisible_count, self.R,self.R_sum)
        #if self.children is not None: #
        if False:
            print'  len(self.children)=%d'%len(self.children)
            for child in self.children.values():
                child.dump()
    def make_children(self,        # TARGET5
                      y_t,         # list of hits at time t
                      t
                      ):
        """ For each of the hits that could plausibly be an
        observation of self, make a child target.  Collect the
        children in a dict and attach it to self.

        Return False if target killed
        """
        assert (self.type is 'target'),'self.type=%s'%self.type
        if not self.children is None:
            return True  # make_children already called for this target
        if self._invisible_count >= self._mod.Invisible_Lifetime:
            self._kill(t)
            return False # Have calling routine move this to dead_targets
        self._forecast()
        self.children = {}
        threshold = self._mod.log_min_pd
        for k in xrange(len(y_t)):
            if threshold > self._utility(y_t[k]):
                continue                 # Candidate utility too small
            self.children[k] = self._spawn(k,t)
            assert(self.children[k].m_t[-1]==k)
        # Child for invisible y
        self._utility(None)
        self.children[-1] = self._spawn(-1,t)
        return True
    def _forecast(self # TARGET5
                 ):
        """ For each ij, calculate forecast mean and covariance for
        both state and observation, the Kalman gain K, the partial
        utility of the transition, and the updated state covariance
        Sigma_us.
        """
        Id = self._mod.Id
        N = len(self._mod.IMM.A)
        for i in xrange(N): # Loop over old IMM components
            A = self._mod.IMM.A[i]
            Sigma_D = self._mod.IMM.Sigma_D[i]
            for j in xrange(N): # Loop over next IMM components
                O = self._mod.IMM.O[j]
                Sigma_O = self._mod.IMM.Sigma_O[j]
                self._mu_fs[i][j] = A*self._mu_t[-1][i]
                self._Sigma_fs[i][j] = A*self._Sigma_t[-1][i]*A.T + Sigma_D
                self._mu_fO[i][j] = O*self._mu_fs[i][j]
                Sig_y = O*self._Sigma_fs[i][j]*O.T + Sigma_O
                self._Sigma_fOI[i][j] = scipy.linalg.inv(Sig_y)
                self._K[i][j] = self._Sigma_fs[i][j]*O.T*self._Sigma_fOI[i][j]
                self._Sigma_us0[i][j]=(Id-self._K[i][j]*O)*self._Sigma_fs[i][j]
                # _Sigma_us0 is updated state covariance if y is visible
        return
    def _utility(self,  # TARGET5
                y):
        """Return the maximum (over (x(t-1),i(t-1),x(t),i(t)) of the
        utility increment given y(t).  Include log_prob factors for
        Sig_D and visibility transitions.  Save updated values for
        making child target if desired.
        """
        if len(self.m_t) == 0 or self.m_t[-1] >= 0:
            v_old = 0 # Last time target was visible or no last time
        else:
            v_old = 1
        if y is None:
            v_new = 1 # This time the target is invisible
        else:
            v_new = 0
        D_nu_v = math.log(self._mod.PV_V[v_old,v_new])
        N = len(self._mod.IMM.A)
        self._D_nu = scipy.zeros((N,N))
        for i in xrange(N): # Loop over last state
            for j in xrange(N): # Loop over next state
                D_nu_ij = math.log(self._mod.IMM.Pij[i,j])
                if y is None:
                    Sigma_new = self._Sigma_fs[i][j]
                    mu_new = self._mu_fs[i][j]
                    D_nu = D_nu_v + D_nu_ij
                else:
                    Sigma_fOI = self._Sigma_fOI[i][j]
                    norm = math.log(scipy.linalg.det(Sigma_fOI))/2
                    Delta_y = y - self._mu_fO[i][j]
                    Sigma_new = self._Sigma_us0[i][j]
                    mu_new = self._mu_fs[i][j] + self._K[i][j]*Delta_y
                    D_nu = D_nu_v + D_nu_ij + norm - float(
                        Delta_y.T*Sigma_fOI*Delta_y)/2
                self._mu_us[i][j] = mu_new
                self._Sigma_us[i][j] = Sigma_new
                self._D_nu[i,j] = D_nu
        return float(self._D_nu.max())
    def _spawn(self, # TARGET5
           k,         # Index of the observation
           t          # Present time
           ):
        """ Called by make_children if utility(y) is OK.  Creates a
        new target using information in self from preceding calls to
        forecast() and utility(y).  The new target incorporates the
        observation y into its state.
        """
        # For each IMM j, find best predecessor i and corresponding nu
        nu_ij = (self._D_nu.T+self._nu).T # NxN array of possible nu values
        temp = nu_ij.argmax(0)              # temp[j] is best i for j
        best = self._best + [temp]
        nu = scipy.choose(temp,nu_ij)  # temp[j] is util of best path to j
        # Return a new instance of self's class (could be a subclass)
        return self.__class__(par_tar=self,copy_index=True,nu=nu,best=best,
                              tk=(t,k))
    def Launch(self, # TARGET5
           y,        # The observation of the target at the current time
           k,        # Index of the observation
           t         # Present time
           ):
        """ Use a Kalman filter to create a new target with a unique
        index and lists with a single entry for observation y."""
        self._forecast()        # Calculate forecast parmeters in self
        D_nu = self._utility(y)
        CC =  math.log(self._mod.Lambda_new) # Creation cost
        # The following mimics spawn()
        nu_ij = (self._D_nu.T+self._nu).T # NxN array of possible nu values
        temp = nu_ij.argmax(0)              # temp[j] is best i for j
        best = [temp]
        nu = scipy.choose(temp,nu_ij) # temp[j] is util of best path to j
        # Return a new instance of self's class (could be a subclass)
        return self.__class__(par_tar=self,copy_index=False,nu=nu+CC,best=best,
                              tk=(t,k))
    def backtrack(self # TARGET5
                  ):
        T = len(self._mu_t)
        s_t = range(T)       # Allocate places for MAP states
        i = self._nu.argmax()
        s_t[T-1] = self._mu_t[T-1][i]
        for t in xrange(T-2,-1,-1):
            i = self._best[t][i]
            Sig_t_I = scipy.linalg.inv(self._Sigma_t[t][i])
            mu_t = self._mu_t[t][i]
            A = self._mod.IMM.A[i]
            X = A.T * scipy.linalg.inv(self._mod.IMM.Sigma_D[i])
            s_t[t]=scipy.linalg.inv(Sig_t_I + X*A)*(Sig_t_I*mu_t + X*s_t[t+1])
        return s_t
    def _kill(self, # TARGET5
             t
             ):
        self.type = 'dead_target'
        self.children = {}
        for List in self.m_t,self._mu_t,self._Sigma_t:
            del(List[-self._mod.Invisible_Lifetime:])
        self._last = t-self._mod.Invisible_Lifetime
        return
    def target_time(self, # TARGET5
                    T):
        """ To give MV*.decode_back the starting and ending times for
        self.
        """
        if self.type is 'target':
            return [self,T-len(self.m_t),T]
        if self.type is 'dead_target':
            return [self,self._last-len(self.m_t),self._last]
        else:
            raise RuntimeError,'unrecognized target type %s'%self.type
# End of class TARGET5
class TARGET4(CAUSE):
    """A TARGET is a possible moving target, eg, a car.  Its values
    are determined by initialization values and a sequence of
    observations that are "Kalman filtered".

    2008-4-21 Rewrite to match broken TARGET5.  Goal: to understand
    Target_Counter values printed during run and to understand
    __init__ and New calls.
    """
    def __init__(self,             # TARGET4
                 mod=None,         # Parent model
                 par_tar=None,     # Parent target
                 copy_index=False, # Copy index from par_tar
                 tk=None           # (time, hit index)
                 ):
        """ Can create a new TARGET4 instance in each of the following modes:
        1. Make target_0 from info in parent model called by MV*.decode_init
        2. New target by applying target_0 to a hit
        3. Spawn a target from an existing target with history
        """
        global Target_Counter, Child_Target_Counter
        # Determine mode, ie source of initialization information
        mode = None
        if mod != None and par_tar == None and copy_index == False:
            mode = 1 # Make target_0
        if mod == None and par_tar != None and copy_index == False:
            mode = 2 # Make new target from target_0 and y
        if mod == None and par_tar != None and copy_index == True:
            mode = 3 # Make child from existing target
        assert (mode != None),"Inconsistent arguments in TARGET4.__init__()"
        # Do initialization that is same for all modes
        self.type = 'target'
        self._last = None
        self._invisible_count = 0
        self.tks = {}
        # Begin mode dependent initialization
        if mode == 1: # Make target_0
            self._mod = mod
            self._mu_t = [mod.mu_init]
            self._Sigma_t = [mod.Sigma_init]
            self.index = 0
            self.m_t = []
            self.R_sum = 0
            self.R = 0
            return
        self._mod = par_tar._mod
        t,k = tk
        mu = par_tar._mu_us        # Updated state mu from _utility()
        Sigma = par_tar._Sigma_us  # Updated state Sigma from _utility()
        if mode == 2: # target_0 + y => new target
            self._mu_t = [mu]
            self._Sigma_t = [Sigma]
            self.T0 = t
            self.index = Target_Counter
            Target_Counter += 1
            self.m_t = [k]
        if mode == 3: # old target + y => new target
            self._mu_t = par_tar._mu_t + [mu]
            self._Sigma_t = par_tar._Sigma_t+ [Sigma]
            if k < 0:
                self._invisible_count = par_tar._invisible_count + 1
            else:
                self._invisible_count = 0
            Child_Target_Counter += 1   # Inc child counter for debugging
            self.T0 = par_tar.T0
            self.tks = par_tar.tks.copy()
            self.index = par_tar.index
            self.m_t = par_tar.m_t + [k]
        # Start common code for mode 2 and 3
        self.R_sum = par_tar.R_sum + par_tar._D_nu
        self.R = par_tar._D_nu
        if k >= 0:                # Don't enter invisibles into tks
            self.tks[tk] = self
        self.children = None
        return # End of __init__
    def dump(self #TARGET4
             ):
        print '\n Dump %s: m_t='%self.type,self.m_t
        tks = self.tks.keys()
        tks.sort()
        print '  tks=',tks
        print '             T0=%d, index=%d, len(mu_t)=%d t_last='%(self.T0,
               self.index,len(self._mu_t)),self._last
        print '   invisible_count=%d, R=%5.3f, R_sum=%5.3f'%(
            self._invisible_count, self.R,self.R_sum)
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

        Return False if target killed
        """
        assert (self.type is 'target'),'self.type=%s'%self.type
        if not self.children is None:
            return True  # make_children already called for this target
        if self._invisible_count >= self._mod.Invisible_Lifetime:
            self._kill(t)
            return False # Have calling routine move this to dead_targets
        self._forecast()
        self.children = {}
        threshold = self._mod.log_min_pd
        for k in xrange(len(y_t)):
            if threshold > self._utility(y_t[k]):
                continue                 # Candidate utility too small
            # Return a new instance of self's class (could be a subclass)
            self.children[k] = self.__class__(par_tar=self,copy_index=True,
                                              tk=(t,k))
            assert(self.children[k].m_t[-1]==k)
        # Child for invisible y
        self._utility(None)
        self.children[-1] = self.__class__(par_tar=self,copy_index=True,
                                           tk=(t,-1))
        return True
    def _forecast(self # TARGET4
                 ):
        """ Calculate forecast mean and covariance for both state and
        observation.  Also calculate K and Sigma_us0.  Save all six
        for use by Update.
        """
        A = self._mod.A
        O = self._mod.O
        self._mu_fs = A*self._mu_t[-1] # _fs is forecast state
        self._Sigma_fs = A*self._Sigma_t[-1]*A.T + self._mod.Sigma_D
        self._mu_fO = O*self._mu_fs # _fO is forecast observation
        Sig_y = O*self._Sigma_fs*O.T + self._mod.Sigma_O
        self._Sigma_fOI = scipy.linalg.inv(Sig_y)
        self._K = self._Sigma_fs*O.T*self._Sigma_fOI
        self._Sigma_us0 = (self._mod.Id-self._K*O)*self._Sigma_fs
        # _Sigma_us0 is updated state covariance if y is visible
        return
    def _utility(self,  # TARGET4
                y):
        """Calcualte log probability density, updated mean, updated
        covariance.  Include log_prob factors for Sig_D and visibility
        transitions.  Save updated values for making child target if
        desired.  Return D_nu.
        """
        if len(self.m_t) == 0 or self.m_t[-1] >= 0:
            v_old = 0 # Last time target was visible or no last time
        else:
            v_old = 1
        if y is None:
            v_new = 1 # This time the target is invisible
        else:
            v_new = 0
        D_nu_v = math.log(self._mod.PV_V[v_old,v_new])
        if y is None:
            Sigma_new = self._Sigma_fs
            mu_new = self._mu_fs
            D_nu = D_nu_v
        else:
            norm = math.log(scipy.linalg.det(self._Sigma_fOI))/2
            Delta_y = y - self._mu_fO
            Sigma_new = self._Sigma_us0
            mu_new = self._mu_fs + self._K*Delta_y
            D_nu = D_nu_v + norm - float(Delta_y.T*self._Sigma_fOI*Delta_y)/2
        self._mu_us = mu_new
        self._Sigma_us = Sigma_new
        self._D_nu = D_nu
        return D_nu
    def Launch(self, # TARGET4
           y,        # The observation of the target at the current time
           k,        # Index of the observation
           t         # Present time
           ):
        """ Use a Kalman filter to create a new target with a unique
        index, fresh m_t, mu_t, Sigma_t and R for the observation y."""
        self._forecast()        # Calculate forecast parmeters in self
        Delta_y = y - self._mu_fO
        QF = -float(Delta_y.T*self._Sigma_fOI*Delta_y)/2
        CC =  math.log(self._mod.Lambda_new) # Creation cost
        norm = math.log(scipy.linalg.det(self._Sigma_fOI))/2
        self._D_nu = norm + CC + QF
        self._mu_us = self._mu_fs + self._K*Delta_y
        self._Sigma_us = self._Sigma_us0
        #print 'Utility of newt is %5.2f, QF=%5.2f, CC=%5.2f, norm=%5.2f'%(self._D_nu,
        #              QF, CC, norm)
        # Return a new instance of self's class (could be a subclass)
        return self.__class__(par_tar=self,copy_index=False,tk=(t,k))
    def backtrack(self # TARGET4
                  ):
        T = len(self._mu_t)
        A = self._mod.A
        X = A.T * scipy.linalg.inv(self._mod.Sigma_D) # An intermediate
        s_t = range(T)
        s_t[T-1] = self._mu_t[T-1]
        for t in xrange(T-2,-1,-1):
            Sig_t_I = scipy.linalg.inv(self._Sigma_t[t])
            mu_t = self._mu_t[t]
            s_t[t]=scipy.linalg.inv(Sig_t_I + X*A)*(Sig_t_I*mu_t + X*s_t[t+1])
        return s_t
    def _kill(self, # TARGET4
             t
             ):
        self.type = 'dead_target'
        self.children = {}
        for List in self.m_t,self._mu_t,self._Sigma_t:
            del(List[-self._mod.Invisible_Lifetime:])
        self._last = t-self._mod.Invisible_Lifetime
        return
    def target_time(self,T):
        """ To give MV*.decode_back the starting and ending times for
        self.
        """
        if self.type is 'target':
            return [self,T-len(self.m_t),T]
        if self.type is 'dead_target':
            return [self,self._last-len(self.m_t),self._last]
        else:
            raise RuntimeError,'unrecognized target type %s'%self.type
# End of class TARGET4                
class TARGET3(TARGET4):
    """ Like TARGET4, but never kill targets, and create all at t=0
    """
    def __init__(self,             # TARGET3 
                 *args,**kwargs):
        TARGET4.__init__(self,*args,**kwargs)
        self.T0 = 0
        self._invisible_count = 0
        return
# End of class TARGET3               
class TARGET1(TARGET3):
    """ Like TARGET3, but always visible
    """
    def make_children(self, # TARGET1
           y_t,             # list of hits at time t
           t                # Present time
                      ):
        """ Like TARGET4 make_children but no invisible targets
        """
        if not self.children is None and len(self.children) > 0:
            return True # make_children already called for this target
        self._forecast()
        self.children = {}
        threshold = self._mod.log_min_pd
        for k in xrange(len(y_t)):
            D_nu = self._utility(y_t[k])
            if threshold > D_nu:
                continue                 # Candidate utility too small
            self.children[k] = self.__class__(par_tar=self,copy_index=True,
                                              tk=(t,k))
            assert(self.children[k].m_t[-1]==k)
        return True
    def _utility(self,   # TARGET1
                y):
        """ Like TARGET4.utility but no visibility probabilities.
        """
        norm = math.log(scipy.linalg.det(self._Sigma_fOI))/2
        Delta_y = y - self._mu_fO
        Sigma_new = self._Sigma_us0
        mu_new = self._mu_fs + self._K*Delta_y
        D_nu = norm - float(Delta_y.T*self._Sigma_fOI*Delta_y)/2
        self._mu_us = mu_new
        self._Sigma_us = Sigma_new
        self._D_nu = D_nu
        return D_nu
# End of class TARGET1
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
            #rv += suc['associations']
            # FixMe could be dropping good associations with argmax
            rv.append(suc['associations'][util.argmax(suc['u_primes'])])
        rv.sort(cmp_ass)
        return rv
class ASSOCIATION4:
    """This is the discrete part of the state.  It gives an
    explanation for how the collection of observations at time t were
    made.

    Methods:

     __init__:

     Enter: Put cause in association

     Fork: Create a child association that explains one more hit

     Spoon: Modify self to explain one more hit

     join: Merge two associations

     forward:  Create plausible successor associations

     make_children:  Call target.make_children() for each target
     
     check_targets, check_FAs and check_newts: Checks in list of
         cause_checks called by forward() that propose causes of hits

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
        self.vis_tran = True # Allow invisible targets
        self.t = t
        self.type='ASSOCIATION4'
    def Enter(self, # ASSOCIATION4
              cause
              ):
        """ Put the argument "cause" in the association "self" and
        update self.Atks.
        """
        for tk in cause.tks.keys():
            if self.Atks.has_key(tk):
                return (False,self)
        self.nu += cause.R
        self.Atks.update(cause.tks)
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
    def Seed(self # ASSOCIATION4
             ):
        """
        Make a new association to seed exhaustive or Murty.  Copy FAs
        and dead_targets, but don't copy targets.
        """
        CA = self.__class__(self.nu,self.mod) #Child Association
        CA.dead_targets = self.dead_targets.copy()
        CA.FAs = self.FAs.copy()
        CA.t = self.t
        CA.tar_dict = {}
        CA.h2c = {}
        CA.Atks = {}
        for old_C in CA.FAs.values() + CA.dead_targets.values():
            CA.Atks.update(old_C.tks)
        return CA
    
    def Fork(self,  # ASSOCIATION4
             cause, # CAUSE of next hit in y_t
             k
            ):
        """ Create a child that extends association by cause, or, if
        cause.type is void, create a new association that carries the
        same FAs and dead_targets as self.  Perhaps second option
        should be different method.
        """
        CA = self.__class__(self.nu,self.mod) #Child Association
        CA.dead_targets = self.dead_targets.copy()
        CA.FAs = self.FAs.copy()
        CA.t = self.t
        CA.h2c = self.h2c.copy()
        CA.tar_dict = self.tar_dict.copy()
        CA.Atks = self.Atks.copy()
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
        if k >= 0:
            self.h2c[k] = cause.index
        return self.Enter(cause)
    def re_nu_A(self, # ASSOCIATION4
                ):
        """ Calculate self.nu based on member FA's and targets.  Also
        recalculate self.Atks
        """
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
                    print 'At t=%d, T0=%d, target.tks is missing'%(self.t,T0)+\
                    ' (%d,%d).  Dump target'%tk
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
        if k < 0 and not self.vis_tran:
            return
        for target in self.tar_dict.values():
            if target.children.has_key(k):
                causes.append(target.children[k])
        return
    def check_FAs(self, # ASSOCIATION4
                  k,causes,y):
        if k < 0:
            return
        CC = FA(y[k],self.t,k,self.mod.Sigma_FA_I, self.mod)
        if CC.R > self.mod.log_min_pd:
            causes.append(CC) # False alarm
        else:
            print 'check_FAs rejects, self.mod.MaxD=',self.mod.MaxD
        return
    def check_newts(self, # ASSOCIATION4.  Check for new target
                    k,causes,y):
        if k < 0 or not self.mod.newts.has_key(k):
            return
        CC = self.mod.newts[k]
        if self.mod.log_min_pd > CC.R:
            print ('check_newts rejects, self.mod.MaxD=%5.3f,'+\
                  'CC.R=%5.3f, t=%d, k=%d')%(self.mod.MaxD,CC.R,self.t,k)
            return
        causes.append(CC)
    def dump(self # ASSOCIATION4
             ):
        print 'dumping an association of type',self.type,':'
        print '  nu=%f, len(tar_dict)=%d, len(dead_targets)=%d'%(
            self.nu,len(self.tar_dict),len(self.dead_targets)),
        print 'hits -> causes map=', self.h2c
        print '  Atks.keys()=',dict_2_tuple(self.Atks)
        for target in self.tar_dict.values()+self.dead_targets.values()+self.FAs.values():
            target.dump()
    def make_children(self,    # self is a ASSOCIATION4
                      y_t,     # All observations at time t
                      t        # Save time for dead targets
                      ):
        for target in self.tar_dict.values():
            if target.type is 'target' and target.make_children(y_t,t):
                continue # Target could have been killed in other association
            else:  # target died after too many invisible steps
                assert not self.dead_targets.has_key(target.index), \
                       'Dying target appears twice?'
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
        assert (len(causes) == len(k_list)),'len(causes)=%d, len(k_list)=%d'%(
            len(causes), len(k_list))
        # The seed association is a copy of self with an empty tar_dict
        old_list = [self.Seed()]
        # Make list of plausible associations.  At level j, each
        # association explains the source of each y_t[k_list[i]] for
        # i<j.
        for j in xrange(len(k_list)):
            k = k_list[j]
            new_list = []    # List of partial associations at level j
            for partial in old_list:
                for cause in causes[j]:
                    # Don't use same target index more than once.
                    if partial.tar_dict.has_key(cause.index):
                        continue
                    OK,child = partial.Fork(cause,k)
                    if OK:
                        new_list.append(child)
            old_list = new_list
            if len(new_list) == 0:
                return []
        if self.vis_tran:
            for p_i in xrange(len(old_list)): #Enter invisible targets
                partial = old_list[p_i]
                viable = True
                for target in self.tar_dict.values():
                    if partial.tar_dict.has_key(target.index):
                        continue # Skip if child of target in partial
                    if target.children.has_key(-1): # Invisible target
                        partial.Enter(target.children[-1])
                    else:
                        viable = False
                if not viable:
                    del old_list[p_i]
        # Discard low utility associations.
        for asn in old_list:
            asn.re_nu_A() # FixMe this should be redundant.  asn.nu
                          # does not change, but asn.Atks must because
                          # this is critical to getting reasonable
                          # tracks
        old_list.sort(cmp_ass)
        if floor == None:  # Calculate threshold relative to best association
            floor = old_list[0].nu-self.mod.A_floor
        new_list = []
        for asn in old_list[:self.mod.Max_NA]:
            if asn.nu < floor:
                break
            new_list.append(asn)
        return new_list  # End of exhaustive()
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
        dict_cause = {} # Just for counting
        ij_2_cause = {}
        j_mult = {}     # Dict of causes that can explain many observations
        w = {}          # Dict of assignement weights
        # Loop over observations i and causes j&index.  Assign
        # weights w[(i,index)] and map index_2_j.
        for i in xrange(len(k_list)):
            for cause in causes[i]:
                index = cause.index
                if not dict_cause.has_key(index):
                    dict_cause[index] = len(dict_cause)
                # If cause is FA remove restriction that it be used
                # exactly once.
                if index < 0:
                    j_mult[index]=True
                try: # Same for newts
                    if self.mod.newts[k_list[i]] == cause:
                        j_mult[index] = True
                except:
                    pass # Model class may not have newts
                assert(cause.type != 'target' or cause.m_t[-1] == k_list[i])
                w[(i,index)] = cause.R
                ij_2_cause[(i,index)] = cause
        i_mult = {} 
        if self.vis_tran: #Enter invisible targets
            for target in self.tar_dict.values():
                if target.children.has_key(-1): # Invisible target
                    index = target.index
                    if not dict_cause.has_key(index):
                        dict_cause[index] = len(dict_cause)
                    cause = target.children[-1]
                    w[(-1,index)] = cause.R
                    ij_2_cause[(-1,index)] = cause
                    i_mult[-1] = True
        m = len(k_list) + len(i_mult)
        n = len(dict_cause)
        try:
            X = util.H_cvx(w,m,n,i_gnd=i_mult,j_gnd=j_mult)
        except:
            print 'm=%d, n=%d, w='%(m,n),w
            print 'k_list=',k_list,'causes=',causes
            X = util.H_cvx(w,m,n,i_gnd=i_mult,j_gnd=j_mult)
        util_max = 0
        for key in X.keys():
            util_max += w[key]
        if floor != None:
            if util_max < floor:
                return []
        else:  # Calculate threshold relative to best association
            floor = util_max-self.mod.A_floor
        ML = util.M_LIST(w,m,n,i_gnd=i_mult,j_gnd=j_mult)
        ML.till(self.mod.Max_NA,floor)
        hungary_count += ML.H_count
        hungary_time += ML.stop_time-ML.start_time
        new_list = []
        for U,X in ML.association_list:
            # U is utility.  X is map from hits to causes.
            new_A = self.Seed()
            # The seed assn is copy of self with empty tar_dict
            for ij in X:
                cause = ij_2_cause[ij]
                i,j = ij
                if i >= 0: # Visible hit
                    OK, temp = new_A.Spoon(cause,k_list[i])
                else:
                    OK, temp = new_A.Spoon(cause,-1)
                if not OK:
                    raise RuntimeError,'FixMe when is it not OK?'
            new_A.re_nu_A() # FixMe this should not be necessary
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
        global Murty_calls
        self.t = t # Make t available to other methods
        self.verify(None)
        m = max(len(k_list), 1) # 1 for N_hat calculation with all invisible
        causes = []
        N_c =0    # Total number of causes for N_hat calculation
        # For each observation, make a list of plausible causes
        for k in k_list:
            causes_k = []
            for check in self.cause_checks:
                check(k,causes_k,y_t)
            causes.append(causes_k)
            N_c += len(causes_k)
        # List of plausible causes complete
        N_hat = (float(N_c)/m)**m  # Estimated number of associations
        if N_hat < self.mod.Murty_Ex or len(k_list) == 0:
            new_list = self.exhaustive(k_list,causes,floor)
        else: # Murty chokes on empty k_list
            Murty_calls += 1
            new_list = self.Murty(k_list,causes,floor)
        if len(new_list) == 0:
            #print 'forward returning 0 new associations'
            return
        # Now each entry in new_list is a complete association that
        # explains k_list at time t.  Next propose the present
        # association as the predecessor for each association in
        # new_list.
        for asn in new_list:
            asn.verify(k_list)  # FixMe delete this
            successors.enter(asn)
        return # End of forward
    def residual(self,   # ASSOCIATION4
                 Etks    # Dict tks explained by offspring
                 ):
        """ Calculate the utility of the causes of all the tks
        explained by self but not in Etks and collect dead targets
        used
        """
        utility = 0
        dead_list = []
        causes = {}
        for tk,cause in self.Atks.items():
            if Etks.has_key(tk):
                continue
            for ctk in cause.tks.keys():  # Ensure all tks from cause not explained
                if Etks.has_key(ctk):
                    raise RuntimeError,'First tk in Etks, subsequent tk not.'
            causes[cause] = tk
        for cause in causes.keys():
            utility += cause.R_sum
            if cause.type is 'dead_target':
                dead_list.append(cause)
        return (utility, dead_list) # End of residual, end of class ASSOCIATION4
class Cluster:
    """ A cluster is a set of targets and associations containing
    those targets and other causes.  Each association in a cluster
    should explain the same set of observations past and present.

    Methods:
       __init__()    Make self.As from arguments
       Append()     
       merge()
    """
    def __init__(self,            # Cluster
                 asn=None         # Association
                 ):
        """ Make first association and attach to self
        """
        if asn is None:
            self.As = []
        else:
            self.As = [asn]
    def Append(self,            # Cluster
               tks,             # Dict tks[tk] == True
               asn              # Association that causes are from
               ):
        """ Create a new association in this cluster derived from the argument
        asn.  Copy causes in asn that explain tks into the new association.
        """
        nu = 0.0
        new_a = asn.__class__(nu,asn.mod)
        for cause in asn.tar_dict.values() + asn.dead_targets.values() + \
                asn.FAs.values():
            if tks.has_key(cause.tks.keys()[0]):
                new_a.Enter(cause)
        # FixMe remove this check
        in_tuple = dict_2_tuple(tks)
        out_tuple = dict_2_tuple(new_a.Atks)
        assert (in_tuple == out_tuple)
        self.As.append(new_a)
        return
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
                NA = OA.__class__(OA.nu+SA.nu,SA.mod)
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
                 a                   # ASSOCIATION4
                 ):
        self.old_clusters = [Cluster(a)]
        self.mod = a.mod
        self.dead_targets = []
    def make_family(self,  # Cluster_Flock
                    Yt,
                    t
                    ):
        """ Make each of the following:
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
                if parent.children.has_key(-1): #Note invisible children
                    self.ks_and_pars[NI]['ks'][-1] = True
        return # End of find_clusters
    def compat_check(self,     # Cluster_Flock
                     asn,      # Proposed parent association
                     tk_2_NI,  # Map from (t,k) tuples to NI (new index)
                     tk_2_dead # lists of dead targets indexed by (t,k)
                     ):
        """ Check compatibility of asn.  Answers question, is
        fragmentation of this association possible that is consistent
        with fragmentations of previous (higher utility) associations.
        Keys in tk_2_NI are historical hits that must end up in the
        new cluster NI.  Keys in tk_2_dead may end up in the dead
        targets list of the cluster self.
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
        global forgotten_utility
        fragmentON = {} # Fragment clusters indexed by (Old index, New index)
        for OI in xrange(len(self.old_clusters)):
            cluster = self.old_clusters[OI]
            tk_2_NI = {}   # Map from (t,k) tuple to New index
            tk_2_dead = {} # Dead target candidates for
                           # self.dead_targets.  Keys (t,k).  Values
                           # [tara,tarb,etc]
            cluster.As.sort(cmp_ass) # Sort the associations by utility
            # Find associations with highest (past) utility that can
            # be fragmented consistently
            OK_As = []
            # Assume that asn.Atks is the same for each asn
            for asn in cluster.As:
                OK,tk_2_NI,tk_2_dead = self.compat_check(asn,tk_2_NI,tk_2_dead)
                if OK:
                    OK_As.append(asn)
                else:
                    delta = cluster.As[0].nu - asn.nu
                    if delta < 0.4: #FixMe use adjustable threshold
                        close_calls.append(('At t=%d in recluster(), dropped'+\
                          ' branch association that is off by %5.3f')%(t,delta))
            # Invert tk_2_NI to get NI_2_tk with
            # NI_2_tk[NI][tk] = True if it exists
            NI_2_tk = {}
            for tk,NI in tk_2_NI.items():
                if NI_2_tk.has_key(NI):
                    NI_2_tk[NI][tk] = True
                else:
                    NI_2_tk[NI] = {tk:True}
            # Break compatible associations into fragments.  Put
            # pieces in fragmentON[OI]
            fragmentON[OI] = {}
            for NI,tks in NI_2_tk.items():
                fragmentON[OI][NI] = Cluster()
                for asn in OK_As: # Copy necessary causes to cluster fragments
                    fragmentON[OI][NI].Append(tks,asn)
            # Save dead targets from best explanation of (all - tk_2_NI)
            util_max = None
            for asn in OK_As:
                if util_max == None:
                    util_max,dead_max = asn.residual(tk_2_NI)
                    continue
                util_try,dead_try = asn.residual(tk_2_NI)
                if util_try > util_max:
                    util_max = util_try
                    dead_max = dead_try
            if util_max != None:
                forgotten_utility += util_max
                for target in dead_max:
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
                new_clusters[NI] = Cluster(a)
                assert len(self.ks_and_pars[NI]['ks']) == 1,('len(self.ks_'+\
                      'and_pars[NI]["ks"])=%d')%len(self.ks_and_pars[NI]['ks'])
        rv = [] # Return value is [[cluster,k_list],[cluster,k_list],...]
        self.old_clusters = []
        for NI in new_clusters.keys():
            if self.ks_and_pars[NI]['ks'].has_key(-1):
                del self.ks_and_pars[NI]['ks'][-1] # FixMe: inelegant
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
                            math.log(scipy.linalg.det(self.Sigma_O))) /2.0
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
        self.target_0 = TARGET4(mod=self)
        return # End of __init__
    def make_newts(self, # MV4.  Make new targets
                   Yts,  # List of ys for current time
                   t,    # Current time
                   k_list
                   ):
        self.newts = {}
        for k in k_list:
            self.newts[k] = self.target_0.Launch(Yts[k],k,t)
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
                       target.m_t[-1] < 0 or target.m_t[-2] < 0 or \
                       target.m_t[1-self.T_MM] < 0:
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
        return ([self.ASSOCIATION(0.0,self)],0) # First A and first t
    
    def decode_forward(self,   # MV4
                       Ys,     # Observations. Ys[t][k]
                       old_As, # [Initial association] or nub
                       t_first,# Starting t for iteration
                       analysis = False
                       ):
        global close_calls, hungary_count, hungary_time, forgotten_utility
        forgotten_utility = 0
        close_calls = []
        flock = Cluster_Flock(old_As[0])
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
                        print ('Warning: hungary_count=%d, asn_No=%d,' + \
                          ' suc_count=%d')%(hungary_count, asn_No,len(suc_max))
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
            if Child_Target_Counter-TC_last > 100:
                for cluster in flock.old_clusters:
                    print '   Cluster has %d associations with %d targets'%(
                          len(cluster.As),len(cluster.As[0].tar_dict))
            for asn in cluster.As: # FixMe  This may not be necessary
                asn.re_nu_A()
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
                As[i].re_nu_A()
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
        targets_times = [] # Elements have form [target,t_i,t_f])
        for target in A_best.dead_targets.values() + A_best.tar_dict.values():
            targets_times.append(target.target_time(T))
        y_A = [] # y_A[t] is a dict.  y_A[t][j] gives the index of
                 # y[t] associated with target j.  So y[t][y_A[t][j]] is
                 # associated with target j.
        for t in xrange(T):
            y_A.append({}) # Initialize association dicts
        d = [] #If not None, d[j][t] is the x vector for target_j at time t
        for j in xrange(len(targets_times)):
            d.append(T*[None])# Initialize list of decoded states with Nones
            target,start,stop = targets_times[j]
            if (stop - start < 3):  # If this happens, I want to know
                print 'In decode_back: start, stop = ',start,stop
            d[j][start:stop] = target.backtrack()# Set decoded values
            for t in xrange(start,stop):
                y_A[t][j] = target.m_t[t-start] # y[t][y_A[t][j]] is
                                                # associated with target k.
        return (d,y_A)
        # Print for debug
        print 'returning from decode_back with:'
        N_tar = len(d)
        for t in xrange(T):
            print '%2d'%t,
            for n in xrange(N_tar):
                if d[n][t] is None:
                    print 12*' ',
                else:
                    try:
                        print '%6.2f %5.2f'%(d[n][t][0],d[n][t][1]),
                    except:
                        print '\nd[%d][%d]='%(n,t),d[n][t]
                        print '%6.2f %5.2f'%(d[n][t][0],d[n][t][1]),
            print y_A[t]
        return (d,y_A)
    def decode(self, # MV4
               Ys,   # Observations. Ys[t][k] is the kth hit at time t
               analysis = False # A debugging option
               ):
        """Return MAP state sequence """
        A,t_0 = self.decode_init(Ys)
        A_best,T,close_calls = self.decode_forward(Ys,A,t_0,analysis=analysis)
        d,y_A = self.decode_back(A_best,T)
        return (d,y_A,A_best.nu+forgotten_utility,close_calls)
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

class ASSOCIATION3(ASSOCIATION4):
    """ Number of targets is fixed.  Allow false alarms and invisibles
    """
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.cause_checks = [self.check_targets,self.check_FAs]
        self.type='ASSOCIATION3'
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
class ASSOCIATION1(ASSOCIATION3):
    """ vis_tran is false.  Targets and only targets generate hits. """
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.vis_tran = False
        self.cause_checks = [self.check_targets]
        self.type='ASSOCIATION1'
class MV3(MV4):
    """ Number of targets is fixed    
    """
    def __init__(self,**kwargs):
        MV4.__init__(self,**kwargs)
        self.ASSOCIATION = ASSOCIATION3 
        self.target_0 = TARGET3(mod=self)
        self.Invisible_Lifetime = 1 # Any number > 0
    def make_newts(self, # MV3.  Make no new targets
                   Yts,  # List of ys for current time
                   t,    # Current time
                   k_list
                   ):
        return
    def decode_init(self, # MV3
                    Ys ):
        """Since MV4 allows generation of new targets at any time,
        MV4.decode_init() can return zero initial targets and let
        decode_forward() create targets for t=0.  This decode_init()
        must return N_tar initial targets.  It makes a target for each
        of the hits at t=0 and tells decode_forward() to start at t=1.
        """
        T = len(Ys)
        partial = self.ASSOCIATION(0.0,self,t=0)
        for k in xrange(self.N_tar):
            target_k = self.target_0.Launch(Ys[0][k],k,0)
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
        MV3.__init__(self,**kwargs)
        self.ASSOCIATION = ASSOCIATION2
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
        self.target_0 = TARGET1(mod=self)
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
class IMM:
    def __init__(self, mod):
        Stop = scipy.matrix([[1,0],[0,0]])
        self.mus = [mod.mu_init, Stop*mod.mu_init]
        self.Sigmas = [mod.Sigma_init, mod.Sigma_init]
        self.A = [mod.A, Stop*mod.A*Stop]
        self.Sigma_D = [mod.Sigma_D, mod.Sigma_D]
        self.O = [mod.O,mod.O]
        self.Sigma_O = [mod.Sigma_O, mod.Sigma_O]
        self.Pij = scipy.array([[.6,.4],[.2,.8]])
        # FixMe trouble if [[.9,.1],[.05,.95]]
        # self.Pi = scipy.array([1.0,0.0])
        self.nu_0 = scipy.array([0,-100])
        return # End of IMM.__init__()
class MV5(MV4):
    """ Number of targets is fixed    
    """
    def __init__(self,**kwargs):
        MV4.__init__(self,**kwargs) 
        self.IMM = IMM(self) 
        self.target_0 = TARGET5(mod=self)
    def decode_init(self,  # MV5
                    Ys     # Dummy for subclasses of MV4
                    ):
        return ([self.ASSOCIATION(0.0,self)],0) # First A and first t
    
        
class ASS_ABQ(ASSOCIATION4):
    def __init__(self,*args,**kwargs):
        ASSOCIATION4.__init__(self,*args,**kwargs)
        self.type='ASS_ABQ'
        self.cause_checks = [self.check_targets, #No self.check_FAs for ABQ
                             self.check_newts]
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
