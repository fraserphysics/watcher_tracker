import numpy, random

class REGION(object):
    def __init__(self,_pass,period,n,limits):
        self._pass = _pass
        self.period = period
        self.n = n
        self.limits = limits
        return
    def limit(self,t):
        return self.limits[int(self.n*t/self.period)%self.n]
    def _Pass(self):
        return self._pass
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
    def region(self,   # GEOMETRY
               pos):
        """ Return the index of the region in which 'pos' lies.
        """
        assert pos >= self.boundaries[0]
        for r in xrange(len(self.boundaries)-1):
            # For many regions should avoid this linear search
            if pos < self.boundaries[r+1]:
                return r
        return r
    def limit(self,pos,t):
        return self.regions[self.region(pos)].limit(t)
    def _Pass(self,pos):
        return self.regions[self.region(pos)]._Pass()
    def bounds(self,pos):
        if pos < self.boundaries[0] or pos > self.boundaries[-1]:
            return False
        return True
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
    def step(self, t, other):
        old_limit = self.G.limit(self.pos,t)
        self.rvel = random.gauss(0,TARGET.accel_var) + TARGET.relax*self.mean +(
            1-TARGET.relax)*self.rvel
        self.vel = old_limit*self.rvel
        self.pos += self.vel
        if (other != None and
            (not G._Pass(self.pos)) and
            (other.pos + TARGET.close < self.pos)):
            #If passing forbidden and next target too close
            self.pos = other.pos - TARGET.close
            self.vel = other.vel
        new_limit = self.G.limit(self.pos,t)
        self.rvel = self.vel/new_limit
        return self.G.bounds(self.pos)
    
if __name__ == '__main__': # Test code
    import operator
    T = 1000
    G = GEOMETRY()
    targets = []
    for t in xrange(T):
        if t%50 == 0:
            targets.append(TARGET(G))
        targets.sort(key=operator.attrgetter('pos'))
        for i in xrange(len(targets)-1,-1,-1):
            if i == len(targets)-1:
                other = None
            else:
                other = targets[i+1]
            if targets[i].step(t,other):
                print('t=%3d pos[%2d]=%f'%(t,i, targets[i].pos))
            else:
                del targets[i]
                print('targets[%d] rolled off the end'%i)

#---------------
# Local Variables:
# eval: (python-mode)
# End:
