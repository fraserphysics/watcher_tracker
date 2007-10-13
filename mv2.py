"""
mv2.py: Models that allow targets to be invisible for a few frames.
"""
import numpy, scipy, scipy.linalg, random, math

import util, mv1a

class TARGET(mv1a.TARGET):
    def update(self,
           y,        # The observation of the target at the current time
           m         # Index of the observation
           ):
        """ Like mv1a.TARGET.update, but allows for missing observations

        Create a new target with updated m_t, mu_t, Sigma_t and R_t
        for the observation, index pair (y,m)."""
        m_L = self.m_t+[m]
        Delta_R,mu_new,Sigma_new = self.utility(y)
        Sigma_L = self.Sigma_t + [Sigma_new]
        mu_L = self.mu_t + [mu_new]
        
        return TARGET(self.mod,m_L,mu_L,Sigma_L,Delta_R)

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
        else:
            Delta_y = y - self.y_forecast    # Error of forecast observation
            Sigma_new = self.Sigma_next
            mu_new = self.mu_a + self.K*Delta_y
            Delta_R -= (self.mod.log_det_Sig_D+self.mod.log_det_Sig_O + float(
                Delta_y.T*self.Sigma_y_forecast_I*Delta_y))/2
        return (Delta_R,mu_new,Sigma_new)
    
    def distance(self,y):
        """ For MV1a this was the Malhalanobis distance of y from
        forecast y.  Here I generalize to sqrt(-2 Delta_R)
        """
        return float(-2*self.utility(y)[0])**.5

    def make_children(self,        # self is a TARGET
                      y_t,         # list of hits at time t
                      All_children # Dict of children of all permutations
                      ):
        """ For each of the hits that could plausibly be an
        observation of self (including missed hits), make a child
        target.  Collect the children in a dict and attach it to self.
        """
        self.forecast()     # Do forecast part of Kalman filter
        self.children = {}  # Dict with observation indices as keys
        y_plus = y_t+[None] # y_plus[-1] for invisible y
        for k in xrange(-1,len(y_t)):
            y = y_plus[k]
            if self.mod.MaxD > 0.01 and self.distance(y) > self.mod.MaxD:
                continue
                # If hit is closer than MaxD or MaxD is near zero, ie,
                # pruning is off or
            key = tuple(self.m_t+[k])
            if not All_children.has_key(key):
                All_children[key] = self.update(y,k)
            self.children[k] = All_children[key]
                
class MV2(mv1a.MV1a):
    def __init__(self,PV_V=[[.9,.1],[.2,.8]],**kwargs):
        mv1a.MV1a.__init__(self,**kwargs)
        self.PV_V = scipy.matrix(PV_V)
        self.log_det_Sig_D = scipy.linalg.det(self.Sigma_D)
        self.log_det_Sig_O = scipy.linalg.det(self.Sigma_O)
    def decode(self,
               Ys, # Observations. Ys[t][k] is the kth hit at time t
               DTARGET=TARGET,
               DPERMUTATION=mv1a.PERMUTATION
               ):
        return mv1a.MV1a.decode(self,Ys,DTARGET=TARGET, DPERMUTATION=DPERMUTATION)

    
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
    M = MV2(N_tar=4,PV_V=[[.5,0.5],[0.5,.5]])
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
