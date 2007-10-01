"""
mv1.py unrealistic code for small simple models

"""
import numpy, scipy, scipy.linalg, random, math

import util

class MV1:
    """A simple model of observed motion with the following groups of
    methods:

    Tools for applications:
      forward()
      decode()
      simulate()

    Service methods:
      check()
      __init__()

    Debugging method: dump()

    """
    def __init__(self,
                 N_obj = 3,                      # Number of objects
                 A = [[0.81,1],[0,.81]],         # Linear state dynamics
                 Sigma_D = [[0.01,0],[0,0.4]],   # Dynamical noise
                 O = [[1,0]],                    # Observation porjection
                 Sigma_O = [[0.25]],             # Observational noise
                 Sigma_init = [[25.0,0],[0,1.0]],# Initial state distribution
                 mu_init = None                  # Initial state distribution
                 ):
        self.N_obj = N_obj
        self.A = scipy.matrix(A)
        self.Sigma_D = scipy.matrix(Sigma_D)
        self.O = scipy.matrix(O)
        self.Sigma_O = scipy.matrix(Sigma_O)
        self.N_perm = int(scipy.factorial(N_obj))
        self.Sigma_init = scipy.matrix(Sigma_init)
        if mu_init == None:
            dim = self.Sigma_init.shape[0]
            self.mu_init = scipy.matrix(scipy.zeros(dim)).T
        else:
            self.mu_init = scipy.matrix(mu_init).T

    def simulate(self, T):
        """ Return a sequence of T observations and a sequence of T
        states."""
        s_j = []
        for j in xrange(self.N_obj):
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
            permute = util.index_to_permute(i,self.N_obj)
            obs_t = range(self.N_obj) # Make list with length N_obj
            state_t = []
            for j in xrange(self.N_obj):
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
    
    def new_nu_k(self,
                 old_nu_k, # The nu_s dict for the object last time
                 y         # The observation of the object at the current time
                 ):
        """ Return the dict of nu_s for the object at the current
        time.  This is essentially Kalman filtering."""

        # Unpack necessary matrices from self and old_nu_k
        mu_t = old_nu_k['mu']
        Sig_t = old_nu_k['Sigma']
        R_t = old_nu_k['R']
        
        O = self.O
        Sig_O_I = scipy.linalg.inv(self.Sigma_O)
        Sig_D = self.Sigma_D
        A = self.A
        Id = scipy.matrix(scipy.identity(Sig_D.shape[0]))

        # Calculate intermediates
        Sig_a = A*Sig_t*A.T + Sig_D      # Covariance of state forecast
        mu_a = A*mu_t                    # Mean of state forecast
        Delta_y = y - O*mu_a             # Error of forecast observation
        Sig_y = O*Sig_a*O.T+self.Sigma_O # Covariance of forecast observation
        Sig_y_I = scipy.linalg.inv(O*Sig_a*O.T+self.Sigma_O)
        K = Sig_a*O.T*Sig_y_I            # Kalman gain
        
        # Calculate new values
        Sig_tnew = (Id-K*O)*Sig_a
        mu_tnew = mu_a + K*Delta_y
        R_tnew = R_t - float(Delta_y.T*Sig_y_I*Delta_y)/2
        
        new_nu_k = {}
        new_nu_k['mu'] = mu_tnew
        new_nu_k['Sigma'] = Sig_tnew
        new_nu_k['R'] = R_tnew
        return new_nu_k
    
    def new_nu(self,
               nu_oldt, # List of nu_s dicts for each object
               perm_new, # Perm dict for current permutation
               y_t       # List of current observations of each object
              ):
        """ Return a list consisting of a nu_s dict for each object"""
        p_vec = perm_new['p_vector']
        return_nu = []
        for k in xrange(len(nu_oldt)):
            return_nu.append(self.new_nu_k(nu_oldt[k],y_t[p_vec[k]]))
        return return_nu
    
    def decode(self,
               Ys # List of lists of observations
               ):
        """Return MAP state sequence """
        T = len(Ys)

        # Set up storage for nu's and B's
        perm_list = []
        for p in xrange(self.N_perm):
            perm = {}
            perm['p_vector'] = util.index_to_permute(p,self.N_obj)
            perm['B_p'] = []  # List of best predecessor permutations
            perm['nu_s'] = [] # List of utlity of best path to s
            for t in xrange(T):
                perm['B_p'].append(None)
                nu_s_t = []
                for k in xrange(self.N_obj):
                    nu_s_t_k = {}
                    nu_s_t_k['mu'] = None
                    nu_s_t_k['Sigma'] = None
                    nu_s_t_k['R'] = None
                    nu_s_t.append(nu_s_t_k)
                perm['nu_s'].append(nu_s_t)
            perm_list.append(perm)

        # For first time step (t=0), only consider permutation zero
        # and fudge the remainders so that permutation zero will be
        # decoded
        old_nu = []
        for k in xrange(self.N_obj):
            nu_s_t_k = {}
            nu_s_t_k['mu'] = self.mu_init
            nu_s_t_k['Sigma'] = self.Sigma_init
            nu_s_t_k['R'] = 0.0
            old_nu.append(nu_s_t_k)
        for perm_new in perm_list:
            perm_new['nu_s'][0] = self.new_nu(old_nu,perm_new,Ys[0])
        for perm_new in perm_list[1:]: # For every perm except first
            for nu_s_t_k in perm_new['nu_s'][0]:
                nu_s_t_k['R'] -= 1     # Make first permutation at
                                       # first time the best
        # Forward pass through time
        for t in xrange(1,T):
            for perm_new in perm_list:
                trial_nu_R = scipy.zeros(self.N_perm)
                trial_nu = []
                for p in xrange(self.N_perm):
                    old_nu = perm_list[p]['nu_s'][t-1]
                    trial_nu.append(self.new_nu(old_nu,perm_new,Ys[t]))
                    R = 0.0
                    for nu_s_t_k in trial_nu[p]:
                        R = R + nu_s_t_k['R']
                    trial_nu_R[p] = R
                b = trial_nu_R.argmax()
                perm_new['nu_s'][t] = trial_nu[b]
                perm_new['B_p'][t] = b

        # Find the best last permutation and state
        R = scipy.zeros(self.N_perm)
        for p in xrange(self.N_perm):
            nu_last = perm_list[p]['nu_s'][T-1]
            R_p = 0.0
            for nu_last_k in nu_last:
                R_p = R_p + nu_last_k['R']
        b = R.argmax()
        
        s_all = range(T)
        s_old = []
        for k in xrange(self.N_obj):
            best_last_k = perm_list[b]['nu_s'][t][k]['mu']
            s_old.append(best_last_k)
        s_all[T-1] = s_old
        
        # Backtrack to get trajectories
        A = self.A
        X = A.T * scipy.linalg.inv(self.Sigma_D) # An intermediate
        for t in xrange(T-2,-1,-1):
            b = perm_list[b]['B_p'][t+1]
            perm_t = perm_list[b]
            s_t = []
            for k in xrange(self.N_obj):
                nu_t_k = perm_t['nu_s'][t][k]
                Sig_t_I = scipy.linalg.inv(nu_t_k['Sigma'])
                mu_t = nu_t_k['mu']
                
                s_t.append(scipy.linalg.inv(Sig_t_I + X*A)*
                (Sig_t_I*mu_t + X*s_old[k]))
            
            s_all[t] = s_t
            s_old = s_t
        return s_all
            
#---------------
# Local Variables:
# eval: (python-mode)
# End:
