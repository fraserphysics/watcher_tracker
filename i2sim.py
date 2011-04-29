"""http://www.velocityreviews.com/forums/t325406-generated-code-that-is-exec-ed-to-simulate-import-cannot-importos-path.html
"""
import numpy, random, fltk

##################### Begin Simulation Code ##########################
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
##################### End of simulation code #########################
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

class animator(object):
    """ Keeps track of targets and pixels in display
    """
    B = 20 # Height of upper band
    def __init__(self,win,G):
        self.CA = win.CA
        self.GA = win.GA
        self.canvas = win.canvas
        self.buttons = win.b_dict
        self.sliders = win.s_dict
        self.t = 0
        self.G = G
        self.record = False
        self.targets = []
        return
    def start_record(self,button):
        self.history = []
        self.record = True
        return
    def end_record(self,button):
        analyze(self.history)
        self.record = False
        return
    def update(self, # animator
               ptr):
        import time,copy
        
        time_start = time.time()
        rate = self.sliders['new_rate']['value']
        t_samp = self.sliders['t_samp']['value']
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
        B = self.B # Height of upper band
        CA[B:,:,:] = CA[(B-1):-1,:,:]
        CA[0:B,:,:] *= 0
        GA[0:B,:,:] *= 0
        close = self.G.close
        for target in self.targets:
            x = int(target.pos)
            GA[0:B,max(0,x-close):x,:] = 128
            CA[0:B,max(0,x-close):x,:] = target.albedo
        delay = time_start + 1.0/fps - time.time()
        if self.record and self.t%t_samp==0:
            self.history.append((self.t,copy.deepcopy(self.targets)))
        if delay > 0:
            time.sleep(delay)
        self.t += 1
        return
class analyzer(animator):
    """ Does grouping for Ried
    """
    def __init__(self,win,G,history):
        self.CA = win.CA
        self.GA = win.GA
        self.canvas = win.canvas
        self.buttons = win.b_dict
        self.sliders = win.s_dict
        self.t = 0
        self.G = G
        self.history = history
        return
    def update(self, # analyzer
               ptr):
        """
        Display part of self.history that fits on screen.  History is
        a list of pairs (time, targets).  Time is an intiger, and each
        element of targets is a TARGET.
        """
        self.t = int(self.sliders['-t']['value'])
        CA = self.CA  # Color array
        GA = self.GA  # Grey array
        CA *= 0
        GA *= 0
        X,Y,D = CA.shape
        B = self.B        # Vehicle height
        C = self.G.close  # Vehicle width
        y_ = 0
        t = self.t
        while y_ < Y + B and t < len(self.history):
            time,targets = self.history[-1-t]
            for target in targets:
                x = int(target.pos)
                GA[y_:y_+B,max(0,x-C):x,:] = 128
                GA[(y_+2):(y_+B-2),max(0,x-C/2-1),:] = 255
                CA[y_:y_+B,max(0,x-C):x,:] = target.albedo
            y_ += B
            t += 1
        if self.buttons['color']['button'].value():
            self.canvas.new_image(CA)
        else:
            self.canvas.new_image(GA)
        self.canvas.redraw()
def slider_cb(slider,args):
    """ Call back for sliders
    slider  An fltk slider object
    args    The tuple (slider_dict, key)
    """
    slider_dict_key = args[0][args[1]]
    slider_dict_key['value'] = slider.value()
    for act in slider_dict_key['acts']:
        act(slider)
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
    """ Function to put button into GUI
    """
    b = fltk.Fl_Button(x,y,width,height)
    b.label(key)
    b.callback(cb,(b_dict,key))
    Pack.children.append(b)
    b_dict[key]['button'] = b
    return
def menu_button_cb(menu,args):
    """ Call back for menu_buttons
    menu    An fltk menu_button object
    args    The tuple (b_dict, key)
    b_dict[key][string] is a list of functions to call to do the action
    """
    print('In menu_button_cb, menu.mvalue()=',menu.mvalue().label(),
          'key=',args[1])
    return
    for act in args[0][args[1]][button.label()]:
        act(button)
    return
def Menu_Button(b_dict,key,Pack,cb=menu_button_cb,x=0,y=0,width=60,height=20):
    """ Function to put button into GUI
    See /usr/share/pyshared/fltk/test/menubar.py
    """
    entry = b_dict[key]
    b = fltk.Fl_Menu_Button(x,y,width,height,key)
    b.text = key
    b.copy(entry['pulldown'])
    b.callback(cb,(b_dict,key))
    Pack.children.append(b)
    b_dict[key]['button'] = b
    return
class My_win(object):
    """
    Class to make fltk window.  Methods:
    __init__    Initializes entire appearance of window and calls show()
    pack_row    Places a row of buttons or sliders
    close       Clears references to all items so that they disappear
    """
    BHEIGHT = 30     # Height of button row
    V_SPACE = 20      # Vertical space between buttons and sliders
    def __init__(self,Title,X,Y,b_list,s_list,
                 WIDTH=1100,
                 HEIGHT=400,
                 CWIDTH=490,
                 Button=Button
                 ):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CWIDTH = CWIDTH
        self.SHEIGHT = HEIGHT-self.BHEIGHT - 50     # Height of slider row
        self.CA = numpy.zeros((self.HEIGHT,self.WIDTH-self.CWIDTH,3),
                              numpy.uint8) # Color array
        self.GA = numpy.zeros((self.HEIGHT,self.WIDTH-self.CWIDTH,1),
                              numpy.uint8) # Gray array
        self._Y = 5       # Starting y position in window
        window = fltk.Fl_Window(X,Y,self.WIDTH,self.HEIGHT)
        window.color(fltk.FL_WHITE)
        self.b_dict = self.pack_row(b_list, self.CWIDTH, self.BHEIGHT, 90, 20,
                                    Button)
        self.s_dict = self.pack_row(s_list, self.CWIDTH, self.SHEIGHT, 30, 50,
                                    Slide)
        self.canvas = FltkCanvas(self.CWIDTH,0,self.WIDTH-self.CWIDTH,
                                 self.HEIGHT, self.CA)
        window.end()
        window.show(len(Title),Title)
        self.window = window
        return
    def pack_row(self,_list, # Describes each item
                 W,          # Width of entire row
                 H,          # Height of entire row
                 width,      # Width of each item
                 height,     # Height of each item
                 init        # Function to make an item
                 ):
        N = len(_list)
        total_space = W - N*width
        space = int(total_space/N)
        start_x = int(space/2)
        _dict = dict(_list)
        keys = [item[0] for item in _list]
        H_Pack = fltk.Fl_Pack(start_x,self._Y,W,H)
        H_Pack.type(fltk.FL_HORIZONTAL)
        H_Pack.spacing(space)
        H_Pack.children = []
        for key in keys:
            H_Pack.children.append(init(
                _dict,key,H_Pack,width=width,height=height))
        H_Pack.end()
        self._Y += self.BHEIGHT + self.V_SPACE
        return _dict
    def close(self):
        self.window = None
        return

def dummy(*args):
    print('Here in dummy() len(args)=%d\nargs=%s'%(len(args),args.__str__()))
Awin = None
def analyze(history):
    """ Open new window to support analyst exploitation of data
    """
    global Awin
    if Awin != None:
        print("Can only do one analysis at a time")
        return
    def quit(button):
        global Awin
        Awin.close()
        Awin = None
    s_list = [
        ('-t',{'value':0,'min':0,'max':len(history),'step':1,'acts':[]}),
        ]
    b_list = [
        ('Action', {'pulldown':(('New View',0,dummy,5),('New Instance',),
                                ('New Relation',),('Close',))
                  })
        ]
    X,Y = (100,100)           # Position on screen
    Awin = My_win(['Analysis Window'],X,Y,b_list,s_list,WIDTH=800,
                  CWIDTH=190,Button=Menu_Button)
    analyze = analyzer(Awin,GEOMETRY(),history)
    Awin.s_dict['-t']['acts'] = [analyze.update]
    #Awin.b_dict['color']['color'] += (analyze.update,)
    #Awin.b_dict['color']['gray'] += (analyze.update,)
    #analyze.update(None)
    return

if __name__ == '__main__': # Test code
    import operator, sys, fltk, numpy
    # Set up GUI

    keys = [
        'key',      'value','min','max','step','acts']
    slide_list = [
        ['new_rate',  0.10,  0,   .5,    0.01,  []],
        ['accel_dev', 0.03,  0,   .1,    0.001, []],
        ['relax',     0.05, .001, .5,    0.001, []],
        ['speed_dev', 0.40,  0,   .5,    0.002, []],
        ['view_fps',  60.0,  5.0, 100.0, 1.0,   []],
        ['t_samp',       5,  5,   50,       1,  []],
        ]
    s_list = []
    for slide in slide_list:
        t_dict = {}
        for i in xrange(1,len(keys)):
            t_dict[keys[i]]=slide[i]
        s_list.append((slide[0],t_dict))
    slide_list = s_list
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
        ('record', {})  # Specify after anim defined
        ] # Did list first to enable control of layout order
    X,Y = (0,0) # Window postion
    win = My_win(sys.argv,X,Y,button_list,slide_list)
    anim = animator(win,GEOMETRY())
    win.b_dict['record'] = {
        'record':(lambda button : button.label('analyze'), anim.start_record),
        'analyze':(lambda button : button.label('record'), anim.end_record)}
    fltk.Fl.add_idle(anim.update)
    fltk.Fl.run()
    
#---------------
# Local Variables:
# eval: (python-mode)
# End:
