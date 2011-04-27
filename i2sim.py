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
	self._image=A
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
    close = 5 # Minimum following distance
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
    def __init__(self,G,accel_var,speed_var,relax):
        self.G = G
        self.accel_var = accel_var
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
        close = self.G.close
        if (other != None and (not self.G._Pass(self.pos)) and
                (other.pos < close + self.pos)):
            self.pos = max(.1,other.pos - close)
            self.vel = other.vel
        new_limit = self.G.limit(self.pos,t)
        self.rvel = self.vel/new_limit
        return self.G.bounds(self.pos)

class animator(object):
    """ Keeps track of targets and pixels in display
    """
    def __init__(self,CA,GA,G):
        self.CA = CA
        self.GA = GA
        self.t = 0
        self.G = G
        self.record = False
        self.targets = []
        return
    def set_canvas(self,canvas):
        self.canvas = canvas
        return
    def set_buttons(self,buttons):
        self.buttons = buttons
        return
    def set_sliders(self,sliders):
        self.sliders = sliders
        return
    def start_record(self,button):
        self.history = []
        self.record = True
        return
    def end_record(self,button):
        analyze(self.history)
        self.record = False
        return
    def update(self,ptr):
        import time
        
        time_start = time.time()
        rate = self.sliders['new_rate']['value']
        accel_var = self.sliders['accel_dev']['value']**2
        speed_var = self.sliders['speed_dev']['value']**2
        relax = self.sliders['relax']['value']
        fps = self.sliders['view_fps']['value']
        # Display before update to allow gray/color switching during pause
        CA = self.CA  # Color array
        GA = self.GA  # Grey array
        if self.buttons['color']['button'].value():
            self.canvas.new_image(CA)
        else:
            self.canvas.new_image(GA)
        self.canvas.redraw()
        if self.buttons['pause']['button'].value():
            time.sleep(1.0/fps)
            return 

        if random.random() < rate:
            self.targets.append(TARGET(self.G,accel_var,speed_var,relax))
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
        X,Y,D = CA.shape
        B = 20 # Width of upper band
        CA[B:,:,:] = CA[(B-1):-1,:,:]
        CA[0:B,:,:] *= 0
        GA[0:B,:,:] *= 0
        close = self.G.close
        for target in self.targets:
            x = int(target.pos)
            GA[0:B,max(0,x-close):x,:] = 128
            CA[0:B,max(0,x-close):x,:] = target.albedo
        delay = time_start + 1.0/fps - time.time()
        if self.record:
            self.history.append((self.t,self.targets))
        if delay > 0:
            time.sleep(delay)
        self.t += 1
        return
class analyzer(animator):
    """ Does grouping for Ried
    """
    def __init__(self,CA,GA,G,history):
        self.CA = CA
        self.GA = GA
        self.t = 0
        self.G = G
        self.history = history
        return
def slider_cb(slider,args):
    """ Call back for sliders
    slider  An fltk slider object
    args    The tuple (slider_dict, key)
    """
    slider_dict_key = args[0][args[1]]
    slider_dict_key['value'] = slider.value()
    for act in slider_dict_key['acts']:
        act()
    return
def Slide(slide_dict,key,Pack,cb=slider_cb,x=0,y=0,width=30,height=100):
    """ Function to put slider into GUI.  Derived from pi_track/support.py
    """
    sd = slide_dict[key]
    s = fltk.Fl_Value_Slider(x,y,width,height,key)
    s.range(sd['min'],sd['max'])
    s.step(sd['step'])
    s.value(sd['value'])
    s.callback(cb,(slide_dict,key))
    Pack.children.append(s)
    return
def button_cb(button,args):
    """ Call back for buttons
    button  An fltk button object
    args    The tuple (b_dict, key)
    b_dict[key][string] is a list of functions to call to do the action
    """
    for act in args[0][args[1]][button.label()]:
        act(button)
    return
def Button(b_dict,key,Pack,cb=button_cb,x=0,y=0,width=60,height=20):
    b = fltk.Fl_Button(x,y,width,height)
    b.label(key)
    b.callback(cb,(b_dict,key))
    Pack.children.append(b)
    b_dict[key]['button'] = b
    return
analysis_window = None
def analyze(history):
    """ Open new window to support analyst exploitation of data
    """
    global analysis_window
    if analysis_window != None:
        print("Can only do one analysis at a time")
        return
    def quit(button):
        global analysis_window
        analysis_window.thisown = 1
        analysis_window = None
    X,Y = (100,100)           # Position on screen
    WIDTH,HEIGHT = (1000,400) # Shape of window
    BHEIGHT = 30     # Height of button row
    SHEIGHT = HEIGHT-BHEIGHT - 50     # Height of slider row
    CWIDTH  = 390     # Width of control region
    CA = numpy.zeros((HEIGHT,WIDTH-CWIDTH,3),numpy.uint8) # Color array
    GA = numpy.zeros((HEIGHT,WIDTH-CWIDTH,1),numpy.uint8) # Gray array
    grouper = analyzer(CA,GA,GEOMETRY(),history)
    H_SPACE = 20      # Horizontal space between sliders
    X_row   = 10      # Gap from left edge of window to first slider
    Y_      = 5       # Starting y position in window
    s_dict = {'t':{'value':0,'min':0,'max':len(history),'step':1,'acts':[]}}
    b_dict = {
        'quit':  {'quit':(quit,)},
        'color': {'color':(lambda button : button.label('gray'),
                           lambda button : button.value(True)),
                  'gray':(lambda button : button.label('color'),
                            lambda button : button.value(False))
                  }
        }
    window = fltk.Fl_Window(X,Y,WIDTH,HEIGHT)
    window.color(fltk.FL_WHITE)
    X,Y = (X_row,Y_)
    W,H = (0,BHEIGHT)
    H_Pack = fltk.Fl_Pack(X,Y,W,H)
    H_Pack.type(fltk.FL_HORIZONTAL)
    H_Pack.spacing(30)   # Gap between buttons
    H_Pack.children = []
    for key in b_dict.keys():
        H_Pack.children.append(
            Button(b_dict,key,H_Pack,width=65,height=20))
    H_Pack.end()
    Y_ += BHEIGHT + H_SPACE
    X,Y = (X_row,Y_)
    W,H = (0,SHEIGHT)
    H_Pack = fltk.Fl_Pack(X,Y,W,H)
    H_Pack.type(fltk.FL_HORIZONTAL)
    SW = int((CWIDTH - 2*X_row)/(len(slide_list)+2)) # Spacing of sliders
    H_Pack.spacing(SW)
    H_Pack.children = []
    for key in ['t']:
        H_Pack.children.append(Slide(s_dict,key,H_Pack,width=SW/2))
    H_Pack.end()
    window.show(3,['A','B','C'])
    analysis_window = window
    return

if __name__ == '__main__': # Test code
    import operator, sys, fltk, numpy
    # Set up GUI

    WIDTH,HEIGHT = (1000,400) # Shape of window
    X,Y = (0,0)               # Position on screen
    BHEIGHT = 30     # Height of button row
    SHEIGHT = HEIGHT-BHEIGHT - 50     # Height of slider row
    CWIDTH  = 390     # Width of control region
    CA = numpy.zeros((HEIGHT,WIDTH-CWIDTH,3),numpy.uint8) # Color array
    GA = numpy.zeros((HEIGHT,WIDTH-CWIDTH,1),numpy.uint8) # Gray array
    anim = animator(CA,GA,GEOMETRY())
    H_SPACE = 20      # Horizontal space between sliders
    X_row   = 10      # Gap from left edge of window to first slider
    Y_      = 5       # Starting y position in window

    keys = [
        'key',      'value','min','max','step','acts']
    slide_list = [
        ['new_rate',  0.10,  0,   .5,    0.01,  []],
        ['accel_dev', 0.03,  0,   .1,    0.001, []],
        ['relax',     0.05, .001, .5,    0.001, []],
        ['speed_dev', 0.40,  0,   .5,    0.002, []],
        ['view_fps',  30.0,  5.0, 100.0, 1.0,   []]
        ]
    button_list = [
        ('quit',  {'quit':(lambda button : sys.exit(0),)}),
        ('pause', {'pause':(lambda button : button.label('continue'),
                           lambda button : button.value(True)),
                  'continue':(lambda button : button.label('pause'),
                            lambda button : button.value(False))
                  }),
        ('color', {'color':(lambda button : button.label('gray'),
                           lambda button : button.value(True)),
                  'gray':(lambda button : button.label('color'),
                            lambda button : button.value(False))
                  }),
        ('record', {'record':(lambda button : button.label('analyze'),
                             anim.start_record),
                  'analyze':(lambda button : button.label('record'),
                             anim.end_record)
                  })
        ] # Did list first to enable control of layout order
    button_dict = dict(button_list)

    window = fltk.Fl_Window(X,Y,WIDTH,HEIGHT)
    window.color(fltk.FL_WHITE)
    X,Y = (X_row,Y_)
    W,H = (0,BHEIGHT)
    H_Pack = fltk.Fl_Pack(X,Y,W,H)
    H_Pack.type(fltk.FL_HORIZONTAL)
    H_Pack.spacing(30)   # Gap between buttons
    H_Pack.children = []
    for key,value in button_list:
        H_Pack.children.append(
            Button(button_dict,key,H_Pack,width=65,height=20))
    H_Pack.end()
    Y_ += BHEIGHT + H_SPACE
    X,Y = (X_row,Y_)
    W,H = (0,SHEIGHT)
    H_Pack = fltk.Fl_Pack(X,Y,W,H)
    H_Pack.type(fltk.FL_HORIZONTAL)
    SW = int((CWIDTH - 2*X_row)/(len(slide_list)+2)) # Spacing of sliders
    H_Pack.spacing(SW)
    H_Pack.children = []
    slide_dict = {}
    for slide in slide_list:
        t_dict = {}
        for i in xrange(1,len(keys)):
            t_dict[keys[i]]=slide[i]
        slide_dict[slide[0]] = t_dict
        H_Pack.children.append(Slide(slide_dict,slide[0],H_Pack,width=SW/2))
    H_Pack.end()
    canvas = FltkCanvas(CWIDTH,0,WIDTH-CWIDTH,HEIGHT,CA)
    anim.set_canvas(canvas)
    anim.set_buttons(button_dict)
    anim.set_sliders(slide_dict)
    window.end()
    window.show(len(sys.argv), sys.argv)
    fltk.Fl.add_idle(anim.update)
    fltk.Fl.run()
    
#---------------
# Local Variables:
# eval: (python-mode)
# End:
