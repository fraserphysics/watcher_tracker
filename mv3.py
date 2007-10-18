"""
mv3.py: Models that allow both missed detections and false alarms
"""
import numpy, scipy, scipy.linalg, random, math

import util, mv2, mv1a

class PARTIAL:
    def __init__(self,parent=None,cause=None,nu=None):
        if parent is None and cause is None:
            self.dup_check = {} # Dict of parent targets used
            self.perm = []      # Map from hits to sources
            self.u_prime = nu   # u'(self,suc,t+1)
            self.targets = []   # List of targets
            return
        self.dup_check = parent.dup_check.copy()
        self.targets = []       # For cause.type 'FA'
        if cause.type is 'target':
            self.dup_check[cause.parent_key] = None # Block reuse of parent
            self.targets = parent.targets + [cause.target]
        self.perm = parent.perm + [cause.k]
        self.u_prime = parent.u_prime + cause.R
    def invisible(self,parent):
        child = parent.children[-1]
        self.targets += [child]
        self.u_prime += child.R_t
class CAUSE:
    def __init__(self, type, parent=None, k=None, Sigma_I=None, y=None):
        self.type = type    # 'target' or 'FA'
        if type is 'target':
            self.parent_key = tuple(parent.m_t)
            self.target=parent.children[k]
            self.R = self.target.R_t
            self.k = k
            return
        if self.type is 'FA':
            self.R = float(- y.T*Sigma_I*y/2)
            self.k = -1
            self.parent_key = None
            return
        raise RuntimeError,"Cause type %s not known"%self.type
class PERMUTATION(mv1a.PERMUTATION):
    """
    New Methods:

     forward: Create plausible sucessor permutations
    """

    def forward(self,
                new_perms,   # A dict of permutations for the next time step
                y_t          # A list of hits at the next time step
                ):
        """ For each plausible successor S of the PERMUTATION self,
        append the following pair of values to S.predecessor: 1. A
        pointer back to self and 2. The value of u'(self,S,t+1).
        """
        # For each observation, list the plausible causes
        causes = []
        Sigma_FA_I=self.targets[0].mod.Sigma_FA_I
        norm = self.targets[0].mod.log_FA_norm
        Lambda = self.targets[0].mod.Lambda
        for k in xrange(len(y_t)):
            causes.append([CAUSE('FA',Sigma_I=Sigma_FA_I,y=y_t[k])])
            for target in self.targets:
                if target.children.has_key(k):
                    causes[k].append(CAUSE('target',target,k))
        
        # Initialize list of partial associations.
        old_list = [PARTIAL(nu=self.nu)]
        # Make list of plausible associations.  At level k, each
        # association explains the source of each y_t[i] for i<k.
        for k in xrange(len(y_t)):
            y_t_k = y_t[k]
            new_list = []
            for partial in old_list:
                for cause in causes[k]:
                    if not partial.dup_check.has_key(cause.parent_key):
                        new_list.append(PARTIAL(partial,cause))
            old_list = new_list

        # For each plausible child association, account for the
        # invisible targets and the number of false alarms
        for entry in old_list:
            N_FA = len(y_t) - len(entry.dup_check)
            entry.u_prime += N_FA*norm + math.log(Lambda**N_FA/scipy.factorial(N_FA))
            for target in self.targets:
                key = tuple(target.m_t)
                if entry.dup_check.has_key(key):
                    continue
                entry.invisible(target)

        # Now each entry in old_list is a PARTIAL and entry.perm[k]
        # is a plausible CAUSE for hit y_t[k]

        # Initialize successors if necessary and set their predecessors
        for entry in old_list:
            key = tuple(entry.perm)  # Key for successor
            if not new_perms.has_key(key):
                new_perms[key] = PERMUTATION(self.N_tar,key)
            successor = new_perms[key]
            successor.predecessor_perm.append(self)
            successor.predecessor_u_prime.append(entry.u_prime)
    def argmax(self):
        """Select best predecessor, evaluate self.nu, collect list of
        appropriate child targets from that predecessor, and attach
        that list to self
        """
        k_max = util.argmax(self.predecessor_u_prime)
        self.nu = self.predecessor_u_prime[k_max]
        best = self.predecessor_perm[k_max]
        self.targets = []
        # For each target in the predecessor, find the child that
        # belongs in this successor and attach it here.  Note that
        # self.key[k] indicates the cause of y_t[k].
        look_up = (self.N_tar+1)*[-1]
        for i in xrange(len(self.key)):
            look_up[self.key[i]] = i
        for k in xrange(self.N_tar):
            i = look_up[k] # Index of observation caused by target k
            self.targets.append(best.targets[k].children[i])
class MV3(mv2.MV2):
    """ A state consists of: Association; Locations; and Visibilities;
    (and the derivable N_FA), ie,

    s = (M,X,V,N_FA)
    
    """
    def __init__(self,Lambda=0.3,**kwargs):
        mv2.MV2.__init__(self,**kwargs)
        self.Lambda = Lambda # Average number of false alarms per frame
        Sigma_FA = self.O*self.Sigma_init*self.O.T + self.Sigma_O
        self.Sigma_FA = Sigma_FA
        self.log_FA_norm = - math.log(scipy.linalg.det(Sigma_FA))/2
        self.Sigma_FA_I = scipy.linalg.inv(Sigma_FA)
    
    def decode(self,
               Ys, # Observations. Ys[t][k] is the kth hit at time t
               DPERMUTATION=PERMUTATION
               ):
        return mv2.MV2.decode(self,Ys,DPERMUTATION=PERMUTATION)
    
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
    M = MV3(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]])
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
    print 'Elapsed time = %4.2f seconds.  '%(time.time()-ts)

#---------------
# Local Variables:
# eval: (python-mode)
# End:
