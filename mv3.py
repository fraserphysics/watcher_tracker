"""
mv3.py: Models that allow both missed detections and false alarms
"""
import numpy, scipy, scipy.linalg, random, math

import util, mv2, mv1a
    
class PERMUTATION(mv1a.PERMUTATION):
    """
    New Methods:

     forward: Create plausible sucessor permutations
    """

    def forward(self,
                new_perms,   # A dict of permutations for the next time step
                y_t          # A list of hits at this time step
                ):
        """ For each plausible successor S of the PERMUTATION self,
        append the following pair of values to S.predecessor: 1. A
        pointer back to self and 2. The value of u'(self,S,t+1).
        """
        # Create a list of plausible successor associations between
        # targets and observations.

        old_list = [] # List of partial associations.  At level k,
                      # each association says which y is associated
                      # with targets with indices smaller than k
                      
        # Initialize with target[k=0]
        for child in self.targets[0].children.values():
            m_tail = child.m_t[-1]         # Index of hit for child
            old_list.append({
                'dup_check':{m_tail:None}, # Ensure unique hit associations
                'perm':[m_tail],           # Map from targets to hits
                'R':child.R_t + self.nu    # u'(self,suc,t+1)
                })

        for k in xrange(1,len(self.targets)):
            new_list = []
            for child in self.targets[k].children.values():
                m_tail = child.m_t[-1]
                for partial in old_list:
                    if m_tail >= 0 and partial['dup_check'].has_key(m_tail):
                        continue # Many targets mapping to invisible y OK
                                 # But no 2 targets map to same visible y 
                    new_dict = partial['dup_check'].copy()
                    new_dict[m_tail] = None
                    new_perm = partial['perm']+[m_tail]
                    new_R = partial['R']+child.R_t
                    new_list.append({'dup_check':new_dict,'perm':new_perm,
                                     'R':new_R})
            old_list = new_list
        # Each entry in old_list is a dict that has a plausible
        # association for each hit in y[t+1].  For the ith
        # association, if k=old_list[i]['perm'][j] < 0, then target[j]
        # is not visible, otherwise y[t+1][k] is associated with
        # target[j]

        # For each association apply a penalty of log(P(y)) for each
        # observation y not associated with a target.
        Sigma_I = self.targets[0].mod.Sigma_FA_I
        norm = self.targets[0].mod.log_FA_norm
        for k in xrange(len(y_t)):
            penalty = float(norm - y_t[k].T*Sigma_I*y_t[k]/2)
            for entry in old_list:
                if not entry['dup_check'].has_key(k):
                    entry['R'] += penalty # FixMe: penalties not
                                          # propagated beyond next
                                          # time.
        # Initialize successors if necessary and set their predecessors
        for entry in old_list:
            key = tuple(entry['perm'])  # Dict keys can be tuples but not lists
            if not new_perms.has_key(key):
                new_perms[key] = PERMUTATION(self.N_tar,key)
            successor = new_perms[key]
            successor.predecessor_perm.append(self)
            successor.predecessor_u_prime.append(entry['R'])

class MV3(mv2.MV2):
    def __init__(self,Lambda=0.3,**kwargs):
        mv2.MV2.__init__(self,**kwargs)
        self.Lambda = Lambda # Average number of false alarms per frame
        Sigma_FA = self.O*self.Sigma_init*self.O.T + self.Sigma_O
        self.Sigma_FA = Sigma_FA
        self.log_FA_norm = math.log(self.Lambda) - math.log(
            scipy.linalg.det(Sigma_FA))/2
        self.Sigma_FA_I = scipy.linalg.inv(Sigma_FA)
    
    def decode(self,
               Ys, # Observations. Ys[t][k] is the kth hit at time t
               DPERMUTATION=PERMUTATION
               ):
        return mv2.MV2.decode(self,Ys,DPERMUTATION=PERMUTATION)
    
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
                if t > 0: # force exactly N_tar observations at t=0
                    for k in xrange(scipy.random.poisson(
                        self.Lambda/self.N_tar)):
                        obs_t.append(util.normalS(zero_y,self.Sigma_FA))
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
