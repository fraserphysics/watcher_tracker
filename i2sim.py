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
    """ Keeps track of targets and pixels in simulation display
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
def Menu_Button(b_dict,key,Pack,x=0,y=0,width=60,height=20):
    """ Function to put button into GUI
    See /usr/share/pyshared/fltk/test/menubar.py
    """
    entry = b_dict[key]
    b = fltk.Fl_Menu_Button(x,y,width,height,key)
    b.text = key
    b.copy(entry['pulldown'])
    Pack.children.append(b)
    b_dict[key]['button'] = b
    return
class My_win(fltk.Fl_Window):
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
        fltk.Fl_Window.__init__(self,X,Y,WIDTH,HEIGHT)
        self.color(fltk.FL_WHITE)
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CWIDTH = CWIDTH
        self.SHEIGHT = HEIGHT-self.BHEIGHT - 50     # Height of slider row
        self.CA = numpy.zeros((self.HEIGHT,self.WIDTH-self.CWIDTH,3),
                              numpy.uint8) # Color array
        self.GA = numpy.zeros((self.HEIGHT,self.WIDTH-self.CWIDTH,1),
                              numpy.uint8) # Gray array
        self._Y = 5       # Starting y position in window
        self.b_dict = self.pack_row(b_list, self.CWIDTH, self.BHEIGHT, 90, 20,
                                    Button)
        self.s_dict = self.pack_row(s_list, self.CWIDTH, self.SHEIGHT, 30, 50,
                                    Slide)
        self.canvas = FltkCanvas(self.CWIDTH,0,self.WIDTH-self.CWIDTH,
                                 self.HEIGHT, self.CA)
        self.end()
        self.show(len(Title),Title)
        return
    def pack_row(self,       # My_win
                 _list,      # Describes each item
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
        H_Pack = fltk.Fl_Pack(start_x,self._Y,W,H)
        H_Pack.type(fltk.FL_HORIZONTAL)
        H_Pack.spacing(space)
        H_Pack.children = []
        for key in [item[0] for item in _list]: # Preserves order of keyskeys:
            H_Pack.children.append(init(
                _dict,key,H_Pack,width=width,height=height))
        H_Pack.end()
        self._Y += self.BHEIGHT + self.V_SPACE
        return _dict
class HIT(object):
    C = 5
    B = 20
    def __init__(self,t,x,parent=None):
        self.t = t
        self.x = x
        self._parent = parent
        return
    def display(self, # HIT
                CA,   # Color array
                t=0,  # Time offset 
                color=[127,127,127]):
        t -= self.t
        if (t < 0) or (t > 20):
            return
        C = 5
        B = 20
        x = self.x
        y_ = t*B
        CA[y_:y_+B,max(0,x-C):x,:] = color
        CA[(y_+2):(y_+B-2),max(0,x-C/2-1),:] = 255
    def parent(self, parent=None):
        if parent == None:
            return self.parent
        self._parent = parent
        return
    def display_parent(self, # HIT
                CA,   # Color array
                t=0,  # Time offset  
                color=[127,127,127]):
        if self.parent == None:
            self.display(CA,t=t,color=color)
            return
        self._parent.display_parent(CA,t=t,color=color)
        return
class SET(set,HIT):
    """ 
    """
    def __init__(self,*args,**kwargs):
        set.__init__(self,*args[:1])
        self.title = args[1]
        self._parent = None
        if kwargs.has_key('parent'):
            self._parent = kwargs['parent']
    def display(self, # HIT
                CA,   # Color array
                t=0,  # Time offset  
                color=[127,127,127]):
        for x in self:
            x.display(CA,t=t,color=color)
        return
class SEQ(list,HIT):
    """ 
    """
    def __init__(self,*args,**kwargs):
        list.__init__(self,*args[:1])
        if len(args) > 1:
            self.title = args[1]            
        self._parent = None
        if kwargs.has_key('parent'):
            self._parent = kwargs['parent']
    def display(self, # HIT
                CA,   # Color array
                t=0,  # Time offset  
                color=[127,127,127]):
        for x in self:
            x.display(CA,t=t,color=color)
        return

class VIEWER(object):
    def __init__(self, collection, analyzer):
        self.collection = collection
        self.analyzer = analyzer
        s_list = [
            ('-t',{'value':analyzer.t_max,
                   'min':(analyzer.t_min-1),
                   'max':(analyzer.t_max+1),
                   'step':1,'acts':[self.time]}),
            ]
        b_list = [
            ('Action', {'pulldown':(
                        ('New View',0,analyzer.new_view),
                        ('New Instance',0,analyzer.new_instance),
                        ('New Relation',0,analyzer.new_relation),
                        ('Close',0,analyzer.close,self))
                        })
            ]
        X,Y = (100,100)           # Position on screen
        self.win = My_win([collection.title],X,Y,b_list,s_list,WIDTH=800,
                          CWIDTH=190,Button=Menu_Button)
        self.canvas = self.win.canvas
        self._image = self.canvas._image
        self.s_dict = self.win.s_dict
        self.b_dict = self.win.b_dict
        return
    def time(self,slider):
        CA = self._image
        CA *= 0
        t = self.s_dict['-t']['value']
        for item in self.collection:
            item.display(CA,t=t)
        self.canvas.draw()
class ANALYZER(object):
    """ Holds all of the data and analysis results.  Also is parent of
    all viewers.
    """ 
    def __init__(self, history):
        assert(len(history) > 0)
        # Note:  Discard simulation time and use index of history
        self.t_min = 0
        self.t_max = len(history)+1
        self.history = history
        hits = SET([],'hits')
        traj_dict = {}
        for t in xrange(len(history)):
            dum,targets = history[t]
            for target in targets:
                hit = HIT(t,int(target.pos))
                if not traj_dict.has_key(target):
                    traj_dict[target] = SEQ([hit])
                else:
                    traj_dict[target].append(hit)
                hits.add(hit)
        tracks = SEQ([],'tracks')
        for key in traj_dict.keys():
            tracks.append(traj_dict[key])
        self.collections = {'hits':hits,'tracks':tracks}
        self.viewers = [VIEWER(self.collections['hits'],self)]
        return
    def new_view(self,           # ANALYZER
                 swig_menu_item  # Mysterious
                 ):
        global block
        block = True
        choose_win = fltk.Fl_Window(451, 190, 192, 162)
        browser = fltk.Fl_Select_Browser(5, 5, 180, 150)
        keys = self.collections.keys()
        for key in keys:
            browser.add(key)
        def choose_cb(ptr):
            global block
            self.viewers.append(VIEWER(
                    self.collections[keys[browser.value()-1]],self))
            block = False
        browser.callback(choose_cb)
        browser.end()
        choose_win.pyChildren = [browser]
        choose_win.end()
        Title = ['New View']
        choose_win.show(len(Title),Title)
        choose_win.set_modal()  # Blocks events to other windows
        while block:
            fltk.Fl.wait()      # Waits till block cleared in choose_cb()
    def new_instance(self,swig_menu_item):
        print('new_instance')
    def new_relation(self,swig_menu_item):
        input = fltk.fl_input("Name:", None)
        if input != None:
            print('new_relation named %s'%input)
            return
        else:
            print('Cancel new_relation')
            return
    def close(self,swig_menu_item,viewer):
        global analysis
        if len(self.viewers) == 1:
            if fltk.fl_choice("Closing last view kills analysis.  Continue?",
                              "No", "Yes", None) != 1:
                return
            analysis = None
        #viewer.win.delete_widget()
        viewer.win = None
        self.viewers.remove(viewer)
        return
def dummy(*args):
    print('Here in dummy() len(args)=%d'%(len(args),))
    for arg in args:
        print('  %s'%arg.__str__())
    return
         
analysis = None
def analyze(history):
    """ Open new window to support analyst exploitation of data
    """
    global analysis
    if analysis != None:
        print("Can only do one analysis at a time")
        return
    analysis = ANALYZER(history)
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
