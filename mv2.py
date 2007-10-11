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
        """ Like mv1a.TARGET.update, but allows for observation "None".

        Create a new target with updated m_t, mu_t, Sigma_t and
        R_t for the observation, index pair (y,m).  This is the second
        half of Kalman filtering step."""
        vv = [self.m_L[-1],m]
        for k in [0,1]:
            if vv[k] is -1:
                vv[k] = 1
            else:
                vv[k] = 0
        Delta_R = math.log(self.mod.PV_V[vv[0],vv[1]])
        m_L = self.m_t+[m]
        if y is None:
            Sigma_L = self.Sigma_t + [self.Sigma_a]
            mu_L = self.mu_t + [self.mu_a]
            Delta_R -= self.mod.log_det_Sig_D/2
        else:
            Delta_y = y - self.y_forecast    # Error of forecast observation
            Sigma_L = self.Sigma_t + [self.Sigma_next]
            mu_L = self.mu_t + [self.mu_a + self.K*Delta_y]
            Delta_R -= (self.mod.log_det_Sig_D+self.mod.log_det_Sig_O + float(
                Delta_y.T*self.Sigma_y_forecast_I*Delta_y))/2
        R_L = self.R_t + [self.R_t[-1] + Delta_R]
        
        return TARGET(self.mod,m_L,mu_L,Sigma_L,R_L)

class PERMUTATION(mv1a.PERMUTATION):
    def make_children(self,   # self is a PERMUTATION
                      y_t,    # All observations at time t
                      cousins # Targets from all permutations
                      ):
        for target in self.targets:
            target.make_children(y_t,cousins)
            key = tuple(target.m_t+[-1])
            if not cousins.has_key(key):
                cousins[key] = target.update(None,-1)
            target.children[-1] = cousins[key]

class MV2(mv1a.MV1a):
    def __init__(self,PV_V=[[1,0],[0,1]],**kwargs):
        mv1a.MV1a.__init__(self,**kwargs)
        self.PV_V = scipy.matrix(PV_V)
        self.log_det_Sig_D = scipy.linalg.det(self.Sigma_D)
        self.log_det_Sig_O = scipy.linalg.det(self.Sigma_O)

    
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
            if t>0:
                i = random.randint(0,self.N_perm-1)
            else:
                i = 0 # No shuffle for t=0
            permute = util.index_to_permute(i,self.N_tar)
            state_t = []
            for j in xrange(self.N_tar):
                pv = self.PV_V[v_j[j],0]
                if pv > random.random():
                    v_j[j] = 0
                else:
                    v_j[j] = 1
                epsilon = util.normalS(zero_x,self.Sigma_D) # Dynamical noise
                x_j[j] = self.A*x_j[j] + epsilon
            obs_t = []
            for j in xrange(self.N_tar):
                if v_j[j] is not 0:
                    continue
                eta = util.normalS(zero_y,self.Sigma_O) # Observational noise
                obs_t.append(self.O * x_j[permute[j]] + eta)
            obs.append(obs_t)
            xs.append(x_j)
        return obs,xs

# Test code
if __name__ == '__main__':
    import time
    random.seed(3)
    ts = time.time()
    M = MV2(N_tar=4)
    y,s = M.simulate(5)
    d = M.decode(y)
    print 'len(y)=',len(y), 'len(s)=',len(s),'len(d)=',len(d)
    for t in xrange(len(y)):
        print 't=%d    y         s           d'%t
        for k in xrange(len(y[t])):
            print ' k=%d  %4.2f  '%(k,y[t][k][0,0]),
            for f in (s[t][k],d[t][k]):
                print '(%4.2f, %4.2f)  '%(f[0,0],f[1,0]),
            print ' '
    print 'Elapsed time = %4.2f seconds.  '%(time.time()-ts)+\
          'Takes 0.58 seconds on my AMD Sempron 3000'

#---------------
# Local Variables:
# eval: (python-mode)
# End:
