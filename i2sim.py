import numpy, random

class REGION(object):
    def __init__(self,_pass,period,n,limits):
        self.pass = _pass
        self.period = period
        self.n = n
        self.limits
        return
    def limit(self,t):
        return self.limits[int(self.n*t/self.period)%self.n]
   def Pass(self):
       return self.pass
class GEOMETRY(object):
    """
    Describes road
    """
    time = 0
    boundaries = numpy.array([0,100,200,300],numpy.float64)
    regions = {0:REGION(False,     # No passing
                        1,1,[2]    # One speed limit of 2.0
                        ),
               1:REGION(True,      # Passing OK
                        50,2,[2,1] # Speed limit switches between 2
                                   # and 1 with period 50
                        ),
               2:REGION(True,      # Passing OK
                        1,1,[3])   # Speed limit is always 3
               }
    def __init__(self):
        return
    def region(self,pos):
        assert pos >= self.boundaries[0]
        for r in xrange(len(self.boundaries)-1):
            if pos < boundaries[r+1]:
                return r
        print('pos=%f is beyond last boundary'%pos)
        return r
    def limit(self,pos):
        return self.regions[self.region(pos)].limit(t)
    def Pass(self,pos):
        return self.regions[self.region(pos)].Pass()
class TARGET(object):
    """
    A class whose instances have invariant characteristics and
    position and velocity
    """
    accel_var = 0.01
    close = 1.0
    speed_dev = 0.2  # Deviation of preferred speed relative to limit
    relax = 0.05
    def __init__(self,G):
        self.G = G
        self.pos = 0.0
        self.albedo = random.random()
        self.mean = random.gauss(1,TARGET.speed_dev**2) # Preferred rvel
        self.rvel = random.gauss(self.mean,TARGET.accel_var/TARGET.relax)
        return
    def step(self, limit, other, Pass):
        old_limit = self.G.limit(self.pos)
        self.rvel = random.gauss(0,TARGET.accel) + TARGET.relax*self.mean + (
            1-TARGET.relax)*self.revel
        self.vel = old_limit*self.rvel
        self.pos += self.vel
        if (not G.Pass(self.pos)) and (other.pos + TARGET.close < self.pos):
            #If passing forbidden and next target too close
            self.pos = other.pos - TARGET.close
            self.vel = other.vel
        new_limit = self.G.limit(self.pos)
        self.rvel = self.vel/new_limit
        return
    
if __name__ == '__main__': # Test code
#---------------
# Local Variables:
# eval: (python-mode)
# End:
