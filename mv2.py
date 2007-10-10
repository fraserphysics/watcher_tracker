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
        if m_L[-1] is -1:
            v_old = 1
        else:
            v_old = 0
        if m is -1:
            v_new = 1
        else:
            v_new = 0
        Delta_R = math.log(self.mod.PV_V[v_old,v_new])
        m_L = self.m_t+[m]
        if y is None:
            mu_L = self.mu_t + [self.mu_a]
            Sigma_L = self.Sigma_t + [self.Sigma_a]
            Delta_R += log_det_stuff
        else:
            Delta_y = y - self.y_forecast    # Error of forecast observation
            Sigma_L = self.Sigma_t + [self.Sigma_next]
            mu_L = self.mu_t + [self.mu_a + self.K*Delta_y]
            Delta_R += log_det_stuff  - float(
                Delta_y.T*self.Sigma_y_forecast_I*Delta_y)/2
        R_L = self.R_t + [self.R_t[-1] + Delta_R]
        
        return TARGET(self.mod,m_L,mu_L,Sigma_L,R_L)

class PERMUTATION(mv1a.PERMUTATION):
    def make_children(self, # self is a PERMUTATION
                      y_t,  # All observations at time t
                      cousins
                      ):
        for target in self.targets:
            target.make_children(y_t,cousins)
            key = tuple(target.m_t+[-1])
            if not cousins.has_key(key):
                cousins[key] = target.update(None,-1)
            target.children[-1] = cousins[key]

class MV2(mv1a.MV1A):
    def __init__(self,PV_V=[[1,0],[0,1]],**kwargs)
    mv1a.MV1A.__init__(self,**kwargs)
    self.PV_V = scipy.matrix(PV_V)

    
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
