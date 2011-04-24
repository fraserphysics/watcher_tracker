import numpy, random, fltk

class FltkCanvas(fltk.Fl_Widget):
    """ A widget that has an array of type uint8 in self._image.  For
    displaying images.  Variant of class in pushbroom/utilities.py
    """
    def __init__(self, #FltkCanvas
                 x,y,w,h,A):
        fltk.Fl_Widget.__init__(self, x, y, w, h, "canvas")
        self._draw_overlay = False
        self._button = None
        self._key = None
        self.new_image(A)
    def new_image(self, #FltkCanvas
                 A,MAX=None):
        if A.ndim == 2:
            W,H = A.shape
            D = 1
            A = A.reshape(W,H,D)
	self._image=A.copy()
        return
    def draw(self #FltkCanvas
                 ):
        newsize=(self.w(),self.h())
        H,W,D = self._image.shape # fl_draw_image assumes different axes order
        fltk.fl_draw_image(self._image,self.x(),self.y(),W,H,D,0)
        self.redraw()
        return

class REGION(object):
    """ A component of the geometry of a simulation.
    """
    def __init__(self,_pass,period,limits):
        self._pass = _pass    # True/False Is passing allowed?
        self.period = period  # Period of cycle of sequence of speed limits 
        self.limits = limits  # Array of limits
        self.n = len(limits)  # Number of different speed limits
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
    boundaries = numpy.array([0,200,400,600],numpy.float64)
    regions = {0:REGION(False,     # No passing
                        1,[2]      # One speed limit of 2.0
                        ),
               1:REGION(True,      # Passing OK
                        300,[4,4,1]# Speed limit switches between 4
                                   # and 1.  300 steps per interval
                        ),
               2:REGION(True,      # Passing OK
                        1,[2])     # Speed limit is always 8
               }
    def __init__(self):
        return
    def region(self,   # GEOMETRY
               pos):
        """ Return the index of the region in which 'pos' lies.
        """
        assert pos >= self.boundaries[0]
        for r in xrange(len(self.boundaries)-1):
            # If many regions, need code to avoid this linear search
            if pos < self.boundaries[r+1]:
                return r
        return r
    def limit(self,pos,t):
        """ Return speed limit for pos and t
        """
        return self.regions[self.region(pos)].limit(t)
    def _Pass(self,pos):
        return self.regions[self.region(pos)]._Pass()
    def bounds(self,pos):
        """ Check if pos is in the boundaries defined by self
        """
        if pos < self.boundaries[0] or pos > self.boundaries[-1]:
            return False
        return True
class TARGET(object):
    """
    A class whose instances have constant characteristics and position
    and velocity
    """
    def __init__(self,G,accel_var,close,speed_var,relax):
        self.G = G
        self.accel_var = accel_var
        self.close = close
        self.relax = relax
        self.pos = 0.0
        albedo = numpy.array([random.randint(1,255) for x in xrange(3)],
                             numpy.int16)
        albedo *= 512.0/albedo.sum()
        self.albedo = albedo
        self.mean = max(.1,random.gauss(1,speed_var)) # Preferred rvel
        self.rvel = max(.1,random.gauss(self.mean,accel_var/relax))
        # rvel, the "relative velocity", is the fraction of the speed
        # limit this target tries to use
        return
    def step(self, t, other):
        old_limit = self.G.limit(self.pos,t)
        self.rvel = max(.1, random.gauss(0,self.accel_var) +
                         self.relax*self.mean + (1-self.relax)*self.rvel)
        self.vel = old_limit*self.rvel
        self.pos += self.vel
        #Follow leader if passing forbidden and next target too close
        if (other != None and (not self.G._Pass(self.pos)) and
                (other.pos < self.close + self.pos)):
            self.pos = max(.1,other.pos - self.close)
            self.vel = other.vel
        new_limit = self.G.limit(self.pos,t)
        self.rvel = self.vel/new_limit
        return self.G.bounds(self.pos)

class animator(object):
    """ Keeps track of targets and pixels in display
    """
    def __init__(self,canvas,G,sliders):
        self.canvas = canvas
        self.t = 0
        self.G = G
        self.sliders = sliders
        self.targets = []
        return
    def update(self,ptr):
        import time
        
        rate = self.sliders['rate']['value']
        accel_var = self.sliders['accel_dev']['value']**2
        speed_var = self.sliders['speed_dev']['value']**2
        relax = self.sliders['relax']['value']
        delay = self.sliders['delay']['value']
        close = 5.0

        if random.random() < rate:
            self.targets.append(TARGET(self.G,accel_var,close,speed_var,relax))
        self.targets.sort(key=operator.attrgetter('pos'),reverse=False)
        for i in xrange(len(self.targets)-1,-1,-1):
            if i == len(self.targets)-1:
                other = None
            else:
                other = self.targets[i+1]
            if self.targets[i].step(self.t,other):
                pass
            else:
                del self.targets[i]
        A = self.canvas._image
        X,Y,D = A.shape
        B = 20 # Width of upper band
        A[B:,:,:] = A[(B-1):-1,:,:]
        A[0:B,:,:] *= 0
        for target in self.targets:
            x = int(target.pos)
            #A[x,y,0] = int(target.albedo)%256
            A[0:B,max(0,x-close):x,:] = target.albedo
        self.canvas.redraw()
        self.t += 1
        time.sleep(delay)
        return    

if __name__ == '__main__': # Test code
    import operator, sys, fltk, numpy
    def slider_cb(slider,args): # Call back for sliders
        global slide_dict
        key = args[0]
        slide_dict[key]['value'] = slider.value()
        for act in args[1]:
            act()
        return
    # Set up GUI
    keys = [
        'key',      'value','min','max','step','acts']
    slide_list = [
        ['rate',      0.03,  0,   .5,    0.01, []],
        ['accel_dev', 0.03,  0,   .1,    0.001, []],
        ['speed_dev', 0.2,   0,   .5,    0.002, []],
        ['relax',     0.05, .001, .5,    0.001, []],
        ['delay',     0.02,  0,   .5,    0.005, []]
        ]
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
    WIDTH =   1000    # Width of window
    SWIDTH  = 390     # Width of slider region
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
    A = numpy.zeros((HEIGHT,WIDTH-SWIDTH,3),numpy.uint8)
    canvas = FltkCanvas(SWIDTH,0,WIDTH-SWIDTH,HEIGHT,A)
    canvas.new_image(A)
    window.end()
    window.show(len(sys.argv), sys.argv)
    anim = animator(canvas,GEOMETRY(),slide_dict)
    fltk.Fl.add_idle(anim.update)
    fltk.Fl.run()
    
#---------------
# Local Variables:
# eval: (python-mode)
# End:
