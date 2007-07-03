"""
filter1.py code for doing filtering and tracking

"""
import numpy, scipy, scipy.linalg, random, math

def column(a):
    """
    A utility that forces argument 'a' to be numpy column vector.
    """
    m = numpy.mat(a)
    r,c = m.shape
    if min(r,c) > 1:
        raise RuntimeError,"(r,c)=(%d,%d)"%(r,c)
    if c == 1:
        return m
    else:
        return m.T

def normalS(mu,sigma,C=None):
    """
    A utility that returns a sample from a vector normal distribtuion.
    """
    if C == None:
        C = numpy.mat(scipy.linalg.cholesky(sigma))
    r,c = C.shape  # rows and columns
    z = []
    for i in xrange(r):
        z.append(random.gauss(0,1))
    z =  numpy.matrix(z)
    return column(mu) + C*z.T
    
def normalP(mu,sigma, x):
    """
    A utility that returns the density of N(mu,sigma) at x.
    """
    from scipy.linalg import solve, det
    r,c = sigma.shape
    d = x-mu
    Q = d.T*solve(sigma,d)
    D = det(sigma)
    norm = 1/((2*math.pi)**(r/2.0)*D)
    P = norm*math.exp(-Q)
    return P

class model1:
    """ My first try at a linear Gaussian model class.  Basic model
    is:
    x(t+1) = F*x(t) + eta(t)         eta ~ Normal(0,Sigma_D)
    y(t+1) = G*x(t+1) + epsilon(t)   epsilon ~ Normal(0,Sigma_O)
    """
    def __init__(self, F,G,Sigma_D,Sigma_O,mu,Sigma):
        self.F = numpy.mat(F)
        self.G = numpy.mat(G)
        self.Sigma_D = numpy.mat(Sigma_D)
        self.Sigma_O = numpy.mat(Sigma_O)
        self.mu = column(mu)
        self.Sigma = numpy.mat(Sigma)
        self.LogLike = 0.0

    def dump(self):
        print 20*'='+'model1.dump()'+20*'='
        print 'F=\n',self.F
        print 'G=\n',self.G
        print 'Sigma_D=\n',self.Sigma_D
        print 'Sigma_O=\n',self.Sigma_O
        print 'mu=\n',self.mu
        print 'Sigma=\n',self.Sigma
        print 60*'='

    def simulate(self, xp):
        """
        Propagate x stochastically, draw y stochastically, and return
        both
        """
        xn = self.F*xp + normalS(xp*0,self.Sigma_D)
        y = self.G*xn
        zero = y*0
        yn = y + normalS(zero,self.Sigma_O)
        return yn,xn

    def forward(self, y):
        """
        Do a Kalman filter iteration.
        """
        F = self.F
        G = self.G

        # Calculate the forecast *_f distribution
        mu_f = F*self.mu
        Sigma_f = F*self.Sigma*F.T + self.Sigma_D
        # Calculate the forecast observation *_g distribution
        mu_g = G*mu_f
        Sigma_g = G*Sigma_f*G.T + self.Sigma_O
        # Calculate the Kalman gain matrix K
        K = Sigma_f *G.T*numpy.linalg.inv(G*Sigma_f*G.T + self.Sigma_O)
        # Calculate the update *_u distribution
        mu_u = mu_f + K*(y-G*mu_f)
        Sigma_u = Sigma_f - K*G*Sigma_f
        self.mu = mu_u
        self.Sigma = Sigma_u
        P = normalP(mu_g,Sigma_g,y)
        self.LogLike = self.LogLike + math.log(P)

# Code for testing

def test():
    random.seed(3)
    O_noise = 1.0
    F = numpy.array([[1,1],[-.1,0.9]])    # Dynamical map
    G = numpy.array([[1,0]])              # Observation map
    D = numpy.array([[1e-6, 0],[0, 1e-6]])# Dynamical noise covariance
    O = numpy.array([[O_noise**2]])       # Observation noise covariance
    mu = numpy.array([4,0])               # Mean of initial distribution
    Sigma = numpy.array([[1,0],[0,1]])    # Covariance of inital distribution
    M = model1(F,G,D,O,mu,Sigma)
    M.dump()
    x = normalS(M.mu,M.Sigma)
    Ys = []
    Xs = []
    for t in xrange(100):
        y_mat,x = M.simulate(x)
        Ys.append(y_mat)
        Xs.append(x)

    for t in xrange(len(Ys)):
        yt = Ys[t]
        M.forward(yt)
        print 'y[%2d]=%5.2f, x=(%5.2f,%5.2f), u=(%5.2f,%5.2f), +/- %5.2f'%(t,Ys[t][0,0],Xs[t][0,0],Xs[t][1,0],M.mu[0,0],M.mu[1,0],(M.Sigma[0,0])**.5)

#---------------
# Local Variables:
# eval: (python-mode)
# End:
