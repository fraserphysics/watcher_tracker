import wx, mvx, demo, random, scipy, time
import matplotlib
#matplotlib.interactive(False)
#matplotlib.interactive(True)
#Use the WxAgg back end. The Wx one takes too long to render
#matplotlib.use('WXAgg')

T = 20
foo_t = 0
N_obj = 4
a_x = 0.98
a_v = 0.98
sig_x = 0.1
sig_v = 0.2
sig_O = 0.3
MaxD   = 5.0   # Maximum Malhabonobis distance from forecast to y
Max_NA = 20    # Maximum number of associations per cluster
A_floor = 4.0  # Drop associations with utility less than max - A_floor
Murty_Ex = 50  # If there are less that Murty_Ex associations use exhaustive()
T_MM = 5       # If two targets match hits more than T_MM in a row, kill one
Analyze = False
Model_No = 0

class FloatSlider(wx.Slider):
    def __init__(self, parent, ID, Min, Max, Step, Value, call_back, **kwargs):
        self.FMin = Min
        self.FMax = Max
        self.FStep = Step
        self.Fvalue = Value
        self.Call_Back = call_back
        V = int((Value-Min)/Step)
        wx.Slider.__init__(self, parent, id=ID, value=V, minValue=0,
                           maxValue=int((Max-Min)/Step), **kwargs)
        parent.Bind(wx.EVT_SLIDER, self.scale, self)
    def scale(self,event):    
        Ivalue = self.GetValue()
        self.Fvalue = self.FMin + Ivalue*self.FStep
        self.Call_Back(event)

class PlotPanelA(demo.PlotPanel):
    """A variation on DemoPlotPanel for panel A that plots different
    numbers of observations at each time"""
    def __init__(self, parent, **kwargs):
        demo.PlotPanel.__init__(self, parent, **kwargs)
        self.y = None  # y[t] is a list of observations
        self.A = None  # A[t] is a dict.  y[t][A[t][k]] is associated
                       # with target k
        self.t = 0     # t is index of y to mark
    def draw(self):
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        if self.y is None:
            return
        self.subplot.hold(True)
        self.subplot.clear()
        # Draw dots for each observation/hit
        for t in xrange(len(self.y)):
            if len(self.y[t]) is 0:   # y[t] is a list of hits at time t
                continue
            x_t = len(self.y[t])*[t]
            y_t = []
            for i in xrange(len(self.y[t])):
                y_t.append(float(self.y[t][i]))
            self.subplot.plot(x_t,y_t,markerfacecolor='black',marker='o',
                              linewidth=0)
            if self.t is t:
                self.subplot.plot(x_t,y_t,markerfacecolor='black',
                 markeredgecolor='black', marker='x',linewidth=0,
                 markeredgewidth=2,markersize=25)
        if self.A is None:
            self.subplot.set_ylim(-80,80)
            self.subplot.set_title(self.title, fontsize = 12)
            self.subplot.set_ylabel(self.ylabel, fontsize = 10)
            self.subplot.set_xlabel(self.xlabel, fontsize = 10)
            return
        # If associations are available, use them to connect dots.
        # A[t] is a dict; keys are targets; y[t][A[t][key]] is an
        # observation of the target indexed by key at time t
        linesx = {}
        linesy = {}
        for t in xrange(len(self.A)):
            for key in self.A[t].keys():
                if self.A[t][key] < 0:
                    continue # Not visible
                if not linesx.has_key(key):
                    linesx[key] = []
                    linesy[key] = []
                linesx[key].append(t)
                linesy[key].append(float(self.y[t][self.A[t][key]]))
        for key in linesx.keys():
            self.subplot.plot(linesx[key],linesy[key], lw=2,color='black',
                           linestyle='-')
        self.subplot.set_title(self.title, fontsize = 12)
        self.subplot.set_ylabel(self.ylabel, fontsize = 10)
        self.subplot.set_xlabel(self.xlabel, fontsize = 10)
        self.subplot.set_ylim(-80,80)

    def _forceDraw(self):
        self.draw()
        self._SetSize()
    def save(self,file_name):
        self.figure.savefig(file_name)

class PlotPanelB(PlotPanelA):
    """A variation on PlotPanelA for plotting state space trajectories"""
    def __init__(self, parent, **kwargs):
        demo.PlotPanel.__init__(self, parent, **kwargs)
        self.s = None  # states (s[t][k][0,0],s[t][k][1,0]) is a point
        self.d = None  # decoded states (d[k][t][0,0],d[k][t][1,0]) is a point
        self.t = 0     # t is index of s and d to mark
    def draw(self):
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        if self.s is None:
            return
        self.subplot.hold(True)
        self.subplot.clear()
        T = len(self.s)
        # Draw dots for each state at each time
        for k in xrange(len(self.s[0])):
            color = self.colors[k%len(self.colors)]
            x = []
            y = []
            for t in xrange(T):
                if self.s[t][k] is not None:
                    x.append(self.s[t][k][1,0])
                    y.append(self.s[t][k][0,0])
            self.subplot.plot(x,y,markerfacecolor=color,marker='o',
                              linewidth=0)
            t = self.t
            if self.s[t][k] is not None:
                self.subplot.plot([self.s[t][k][1,0]],[self.s[t][k][0,0]],
                 markerfacecolor=color, markeredgecolor=color, marker='x',
                 linewidth=0, markeredgewidth=2, markersize=25)
        if self.d is None:
            self.subplot.set_title(self.title, fontsize = 12)
            self.subplot.set_ylabel(self.ylabel, fontsize = 10)
            self.subplot.set_xlabel(self.xlabel, fontsize = 10)
            self.subplot.set_ylim(-80,80)
            return
        # Draw lines for each decoded state trajectory
        for k in xrange(len(self.d)):
            x = []
            y = []
            for t in xrange(T):
                if self.d[k][t] is not None:
                    x.append(self.d[k][t][1,0])
                    y.append(self.d[k][t][0,0])
            color = self.colors[k%len(self.colors)]
            self.subplot.plot(x,y, lw=2, color=color, linestyle='-')
            t = self.t
            if self.d[k][t] is not None:
                self.subplot.plot([self.d[k][t][1,0]],[self.d[k][t][0,0]],
                 markerfacecolor=color, markeredgecolor=color, marker='+',
                 linewidth=0, markeredgewidth=2, markersize=25)
        self.subplot.set_title(self.title, fontsize = 12)
        self.subplot.set_ylabel(self.ylabel, fontsize = 10)
        self.subplot.set_xlabel(self.xlabel, fontsize = 10)
        self.subplot.set_ylim(-80,80)

def layout(panel, compList, orient=wx.VERTICAL):
    box = wx.BoxSizer(orient)
    for c in compList:
        box.Add(c, 0, wx.EXPAND, 4)
    panel.SetSizer(box)
    box.Fit(panel)

def Vlab_slider(binder, parent, label, call_back, **kwargs):
    frame = wx.Panel(parent,-1)
    Label = wx.StaticText(frame,-1,' '+label+' ')
    Slider = wx.Slider(frame, id=-1, style=wx.SL_VERTICAL, **kwargs)
    layout(frame,[Label,Slider])
    binder.Bind(wx.EVT_SLIDER, call_back, Slider)
    return frame,Slider

def VFlab_slider(binder, parent, label, Min, Max, Step, Value, call_back,
                 style=0, **kwargs):
    # Vertical Float Slider
    style += wx.SL_VERTICAL
    frame = wx.Panel(parent,-1)
    Label = wx.StaticText(frame,-1,' '+label+' ')
    Slider = FloatSlider(frame, -1, Min, Max, Step, Value, call_back,
                         style=style, **kwargs)
    layout(frame,[Label,Slider])
    return frame,Slider

class view_mv1_frame(wx.Frame):
    def __init__(self, parent):
        self.title = "MV* Viewer"
        wx.Frame.__init__(self, parent, -1, self.title)
        self.initStatusBar()
        self.controlPanel = ControlPanel(self, -1)
        self.plot_panelA = PlotPanelA(self,ylabel="Position",
               xlabel='Time',title='Observations', size=(400,800))
        self.plot_panelB = PlotPanelB(self,ylabel="Position",
               xlabel='Velocity',title='State Space', size=(400,800))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.controlPanel,0,wx.EXPAND,4)
        sizer.Add(self.plot_panelA,1,wx.EXPAND,4)
        sizer.Add(self.plot_panelB,1,wx.EXPAND,4)
        self.SetSizer(sizer)
        sizer.Fit(self)
        
    def initStatusBar(self):
        self.statusbar = self.CreateStatusBar()

    def OnNew(self, event): pass

    def OnCloseWindow(self, event):
        self.Destroy()

    def ModelClicked(self, event):
        global Model_No, Model_Class
        Model_Classes = [mvx.MV1,mvx.MV2,mvx.MV3,mvx.MV4,mvx.MV5,mvx.MV_ABQ]
        label = ['MV1','MV2','MV3','MV4','MV5','ABQ']
        random.seed(3)
        scipy.random.seed(3)
        Model_No = (Model_No+1)%len(label)
        Model_Class = Model_Classes[Model_No]
        self.controlPanel.ModelButton.SetLabel(label[Model_No])
        
    def AnalClicked(self, event):
        global Analyze
        if Analyze is False:
            Analyze = mvx.analysis()
            self.controlPanel.AnalButton.SetLabel('Analyze is on')
        else:
            Analyze = False
            self.controlPanel.AnalButton.SetLabel('Analyze is off')
        
    def OnSimClicked(self, event):
        global T,N_obj,a_x,a_v,sig_x,sig_v,sig_O,M,yo,s,MaxD,Max_NA
        global A_floor, Murty_Ex, T_MM, Model_Class

        M = Model_Class(N_tar=N_obj,A = [[a_x,1],[0,a_v]],Sigma_O=[[sig_O**2]],
            Sigma_D = [[sig_x**2,0],[0,sig_v**2]],MaxD=MaxD, Max_NA=Max_NA,
            A_floor=A_floor, Murty_Ex=Murty_Ex, T_MM=T_MM)
        yo,s = M.simulate(T)
        self.plot_panelA.A=None
        self.plot_panelA.y=yo
        self.plot_panelA._forceDraw()
        self.plot_panelB.s=s
        self.plot_panelB.d=None
        self.plot_panelB._forceDraw()

    def SaveClicked(self, event):
        global T
        self.plot_panelA.save('figA.png')
        self.plot_panelB.save('figB.png')
        print "Save clicked"
        
    def OnTrackClicked(self, event):
        global T,N_obj,M,yo,s,a_x,a_v,sig_O,sig_x,sig_v,MaxD
        global Model_Class, Analyze

        M = Model_Class(N_tar=N_obj,A = [[a_x,1],[0,a_v]],Sigma_O=[[sig_O**2]],
            Sigma_D = [[sig_x**2,0],[0,sig_v**2]],MaxD=MaxD, Max_NA=Max_NA,
            A_floor=A_floor, Murty_Ex=Murty_Ex, T_MM=T_MM)
        t_start = time.time()
        d,yd,nu,close_calls = M.decode(yo,analysis=Analyze)
        print 'decode time = %f  Number of close calls=%d'%(
            time.time()-t_start, len(close_calls))
        print 'nu_max=%f'%nu
        if Analyze:
            for call in close_calls:
                print call
            Analyze.dump()
        self.plot_panelA.A=yd
        self.plot_panelA._forceDraw()
        self.plot_panelB.d=d
        self.plot_panelB._forceDraw()
    
    def a_x_sliderUpdate(self, event):
        global a_x
        a_x = self.controlPanel.a_x_Slider.Fvalue
        self.statusbar.SetStatusText('a_x=%4.2f'%a_x)

    def a_v_sliderUpdate(self, event):
        global a_v
        a_v = self.controlPanel.a_v_Slider.Fvalue
        self.statusbar.SetStatusText('a_v=%4.2f'%a_v)

    def sig_x_sliderUpdate(self, event):
        global sig_x
        sig_x = self.controlPanel.sig_x_Slider.Fvalue
        self.statusbar.SetStatusText('sig_x=%5.3f'%sig_x)

    def sig_v_sliderUpdate(self, event):
        global sig_v
        sig_v = self.controlPanel.sig_v_Slider.Fvalue
        self.statusbar.SetStatusText('sig_v=%5.3f'%sig_v)

    def sig_O_sliderUpdate(self, event):
        global sig_O
        sig_O = self.controlPanel.sig_O_Slider.Fvalue
        self.statusbar.SetStatusText('sig_O=%5.3f'%sig_O)

    def MaxD_sliderUpdate(self, event):
        global MaxD
        V = self.controlPanel.MaxD_Slider.Fvalue
        if V > 0:
            MaxD = 1.0/V
        else:
            MaxD = 0
        self.statusbar.SetStatusText('Max D in sigmas, MaxD=%5.3f'%MaxD)

    def Max_NA_sliderUpdate(self, event):
        global Max_NA
        Max_NA = self.controlPanel.Max_NA_Slider.GetValue()
        self.statusbar.SetStatusText(
            'Max number of associations per cluster, Max_NA=%d'%Max_NA)

    def A_floor_sliderUpdate(self, event):
        global A_floor
        A_floor = self.controlPanel.A_floor_Slider.Fvalue
        self.statusbar.SetStatusText(
            'Threshold for utility of assocaition in cluster, A_Floor=%5.3f'\
            %A_floor)

    def Murty_Ex_sliderUpdate(self, event):
        global Murty_Ex
        Murty_Ex = self.controlPanel.Murty_Ex_Slider.GetValue()
        self.statusbar.SetStatusText(
            'Threshold for exhaustive search not Murty, Murty_Ex=%d'%Murty_Ex)

    def T_MM_sliderUpdate(self, event):
        global T_MM
        T_MM = self.controlPanel.T_MM_Slider.GetValue()
        self.statusbar.SetStatusText(
            'Time match max for two trajectories, T_MM=%d'%T_MM)

    def t_sliderUpdate(self, event):
        global N_obj,s,yo,foo_t
        Ft = self.controlPanel.t_Slider.Fvalue
        t = max(0,min(T-1,int(T*Ft)))
        if t == foo_t:
            return
        foo_t = t
        self.statusbar.SetStatusText('t=%d'%t)
        self.plot_panelA.t=t
        self.plot_panelA._forceDraw()
        self.plot_panelB.t=t
        self.plot_panelB._forceDraw()
        
    def T_sliderUpdate(self, event):
        global T
        T = self.controlPanel.T_Slider.GetValue()
        self.statusbar.SetStatusText('T=%d'%T)
        
    def N_sliderUpdate(self, event):
        global N_obj
        N_obj = self.controlPanel.N_Slider.GetValue()
        self.statusbar.SetStatusText('N=%d'%N_obj)

class ControlPanel(wx.Panel):

    BMP_SIZE = 16
    BMP_BORDER = 3
    NUM_COLS = 4
    SPACING = 4

    maxThickness = 16

    def __init__(self, parent, ID):
        global T,N_obj,a_x,a_v,sig_x,sig_v,sig_O,MaxD,Max_NA,A_floor
        global Murty_Ex,T_MM
        global Model_No, Model_Class
        wx.Panel.__init__(self, parent, ID, style=wx.RAISED_BORDER)
        buttonSize = (self.BMP_SIZE + 2 * self.BMP_BORDER,
                      self.BMP_SIZE + 2 * self.BMP_BORDER)

        saveButton = wx.Button(parent=self, id=-1, label='Save')
        self.Bind(wx.EVT_BUTTON, parent.SaveClicked, saveButton)

        ModelButton = wx.Button(parent=self, id=-1, label='MV1')
        self.Bind(wx.EVT_BUTTON, parent.ModelClicked, ModelButton)
        self.ModelButton = ModelButton
        Model_Class = mvx.MV1
        Model_No = 0

        AnalButton = wx.Button(parent=self, id=-1, label='Analyze is off')
        self.Bind(wx.EVT_BUTTON, parent.AnalClicked, AnalButton)
        self.AnalButton = AnalButton

        sim_frame = wx.Panel(self,-1)
        
        simButtonA = wx.Button(parent=sim_frame, id=-1, label='Simulate')

        self.Bind(wx.EVT_BUTTON, parent.OnSimClicked, simButtonA)

        row_A = wx.Panel(sim_frame,-1)

        a_x_frame,self.a_x_Slider = VFlab_slider(self,row_A,"a_x",
            0.6, 1.0, 0.01, a_x, parent.a_x_sliderUpdate, size=(-1, 200))
        #  from,  to, step, init
        a_v_frame,self.a_v_Slider = VFlab_slider(self,row_A,"a_v",
            0.8, 1.0, 0.01, a_v, parent.a_v_sliderUpdate, size=(-1, 200))
        N_frame,self.N_Slider = Vlab_slider(self, row_A, "N",
                                parent.N_sliderUpdate, value=N_obj, minValue=1,
                                maxValue=100, size=(-1, 200))
        T_frame,self.T_Slider = Vlab_slider(self, row_A, "T",
                                parent.T_sliderUpdate, value=T, minValue=1,
                                maxValue=100, size=(-1, 200))
        t_frame,self.t_Slider = VFlab_slider(self, row_A, "t", 0.0, 1.0,
                    0.005, 0, parent.t_sliderUpdate, size=(-1, 200))
        layout(row_A,[a_x_frame,a_v_frame,N_frame,T_frame,t_frame],
               orient=wx.HORIZONTAL)

        row_B = wx.Panel(sim_frame,-1)

        sig_x_frame,self.sig_x_Slider = VFlab_slider(self,row_B,"sig_x",
            0.01, 1.0, 0.005, sig_x, parent.sig_x_sliderUpdate, size=(-1, 200))
        sig_v_frame,self.sig_v_Slider = VFlab_slider(self,row_B,"sig_v",
            0.01, 1.0, 0.005, sig_v, parent.sig_v_sliderUpdate, size=(-1, 200))
        sig_O_frame,self.sig_O_Slider = VFlab_slider(self,row_B,"sig_O",
            0.01, 1.0, 0.005, sig_O, parent.sig_O_sliderUpdate, size=(-1, 200))
        
        layout(row_B,[sig_x_frame,sig_v_frame,sig_O_frame],
               orient=wx.HORIZONTAL)

        layout(sim_frame,[simButtonA,row_A,row_B])
        #---------------------------
        track_frame = wx.Panel(self,-1)
        
        trackButtonA = wx.Button(parent=track_frame, id=-1, label='Track')
        self.Bind(wx.EVT_BUTTON, parent.OnTrackClicked, trackButtonA)

        row_C = wx.Panel(track_frame,-1)
        MaxD_frame,self.MaxD_Slider = VFlab_slider(self, row_C,"MD",
            0.0, 1.0, 0.002, 1/MaxD, parent.MaxD_sliderUpdate, size=(-1, 200),
                                                   style=wx.SL_INVERSE)
        Max_NA_frame,self.Max_NA_Slider = Vlab_slider(self, row_C, "MA",
                     parent.Max_NA_sliderUpdate, value=Max_NA, minValue=1,
                                maxValue=100, size=(-1, 200))
        A_floor_frame,self.A_floor_Slider = VFlab_slider(self, row_C,"Af",
            0.0, 20.0, 0.1, 4, parent.A_floor_sliderUpdate, size=(-1, 200))
        Murty_Ex_frame,self.Murty_Ex_Slider = Vlab_slider(self, row_C,
                 "MX", parent.Murty_Ex_sliderUpdate,
                 value=Murty_Ex, minValue=0, maxValue=199, size=(-1, 200))
        T_MM_frame,self.T_MM_Slider = Vlab_slider(self, row_C, "TM",
                     parent.T_MM_sliderUpdate, value=T_MM, minValue=2,
                                maxValue=10, size=(-1, 200))
        layout(row_C,
               [MaxD_frame,Max_NA_frame,A_floor_frame,
                Murty_Ex_frame,T_MM_frame],
               orient=wx.HORIZONTAL)
        
        layout(track_frame,[trackButtonA,row_C])
        #--------------------
        layout(self,[saveButton,ModelButton,AnalButton,sim_frame,track_frame])

class view_mv1_app(wx.App):
    def OnInit(self):
        frame = view_mv1_frame(None)  
        frame.Show()
        return True

app = view_mv1_app(False)
app.MainLoop()
   
#---------------
# Local Variables:
# eval: (python-mode)
# End:
