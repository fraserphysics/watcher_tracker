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
            (not self.G._Pass(self.pos)) and
            (other.pos + TARGET.close < self.pos)):
            #If passing forbidden and next target too close
            self.pos = other.pos - TARGET.close
            self.vel = other.vel
        new_limit = self.G.limit(self.pos,t)
        self.rvel = self.vel/new_limit
        return self.G.bounds(self.pos)

class animator(object):
    def __init__(self,canvas,G):
        self.canvas = canvas
        self.t = 0
        self.G = G
        self.targets = []
        return
    def update(self,ptr):
        A = self.canvas._image
        if self.t%5 == 0:
            self.targets.append(TARGET(self.G))
        self.targets.sort(key=operator.attrgetter('pos'))
        for i in xrange(len(self.targets)-1,-1,-1):
            if i == len(self.targets)-1:
                other = None
            else:
                other = self.targets[i+1]
            if self.targets[i].step(self.t,other):
                pass
            else:
                del self.targets[i]
        X,Y,D = A.shape
        y = self.t%Y
        A[y,:,0] *= 0
        for target in self.targets:
            x = int(target.pos)
            #A[x,y,0] = int(target.albedo)%256
            A[y,x,0] = 255
        self.canvas.redraw()
        self.t += 1
        return    

if __name__ == '__main__': # Test code
    import operator, sys, fltk, numpy, utilities
    def slider_cb(slider,args): # Call back for sliders
        global slide_dict
        key = args[0]
        slide_dict[key]['value'] = int(slider.value())
        for act in args[1]:
            act()
        return
    # Set up GUI
    keys = [
        'key',      'value','min','max','step','acts']
    slide_list = [
        ['rate',      0.01,  0,   .1,    0.001, []],
        ['accel_dev', 0.03,  0,   .1,    0.001, []],
        ['speed_dev', 0.2,   0,   .2,    0.002, []],
        ['relax',     0.05,  0,   .1,    0.001, []]]
    slide_dict = {}
    for slide in slide_list:
        t_dict = {}
        for i in xrange(1,len(keys)):
            t_dict[keys[i]]=slide[i]
        slide_dict[slide[0]] = t_dict
    N_s = len(slide_list)
    HEIGHT =  400     # Height of window
    VPH=HEIGHT-20     # Height of V_Pack
    S_LENGTH=VPH-10   # Length of sliders
    R_HEIGHT= S_LENGTH# Height of each row
    VPS     = 10      # Space between elements in V_Pack
    V_SPACE = 40      # Vertical space between rows of sliders
    WIDTH =   700     # Width of window
    SWIDTH  = 300     # Width of slider region
    H_SPACE = 20      # Horizontal space between sliders
    VPW = int((SWIDTH)/N_s - H_SPACE) # Width of V_Pack
    Y_      = 5       # Starting y position in window
    X_row   = 5       # Gap from left edge of window to first slider
    BS      = 10      # Button size.  I don't understand
    def Slide(key,V_Pack=None):  # from pi_track/support.py
        sd = slide_dict[key]
        if V_Pack == None:
            V_Pack = fltk.Fl_Pack(0,0,VPW,VPH)
            V_Pack.align(fltk.FL_ALIGN_BOTTOM)
            V_Pack.type(fltk.FL_VERTICAL)
            V_Pack.spacing(VPS)
            V_Pack.children = []
        s = fltk.Fl_Value_Slider(0,0,0,S_LENGTH,key)
        s.range(sd['min'],sd['max'])
        s.step(sd['step'])
        s.callback(slider_cb,(key,sd['acts']))
        s.value(sd['value'])
        V_Pack.children.append(s)
        V_Pack.end()
        return
    
    X,Y = (0,0)       # Position on screen
    window = fltk.Fl_Window(X,Y,WIDTH,HEIGHT)
    X,Y = (X_row,Y_)
    W,H = (0,R_HEIGHT)
    H_Pack = fltk.Fl_Pack(X,Y,W,H)
    H_Pack.type(fltk.FL_HORIZONTAL)
    H_Pack.spacing(H_SPACE)
    H_Pack.children = []
    for slide in slide_list:
        H_Pack.children.append(Slide(slide[0]))
    H_Pack.end()
    A = numpy.zeros((WIDTH-SWIDTH,HEIGHT),numpy.uint8)
    canvas = utilities.FltkCanvas(SWIDTH,0,SWIDTH,HEIGHT,A)
    canvas.new_image(A)
    window.end()
    window.show(len(sys.argv), sys.argv)
    anim = animator(canvas,GEOMETRY())
    fltk.Fl.add_idle(anim.update)
    fltk.Fl.run()
    
    # T = 1000
    # G = GEOMETRY()
    # targets = []
    # for t in xrange(T):
    #     if t%50 == 0:
    #         targets.append(TARGET(G))
    #     targets.sort(key=operator.attrgetter('pos'))
    #     for i in xrange(len(targets)-1,-1,-1):
    #         if i == len(targets)-1:
    #             other = None
    #         else:
    #             other = targets[i+1]
    #         if targets[i].step(t,other):
    #             print('t=%3d pos[%2d]=%f'%(t,i, targets[i].pos))
    #         else:
    #             del targets[i]
    #             print('targets[%d] rolled off the end'%i)

#---------------
# Local Variables:
# eval: (python-mode)
# End:
