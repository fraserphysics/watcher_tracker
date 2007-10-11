import wx, mv1a, mv2, demo, random, scipy, time
import matplotlib
#matplotlib.interactive(False)
matplotlib.interactive(True)
#Use the WxAgg back end. The Wx one takes too long to render
matplotlib.use('WXAgg')

random.seed(3)

T = 100
N_obj = 4
a_x = 0.9
a_v = 0.9
sig_x = 0.1
sig_v = 0.2
sig_O = 0.3
MaxD   = 1/3. # Inverse of maximum Malhabonobis distance from forecast to y
MaxP = 120    # Number of permutations allowed
Model_Class = mv1a.MV1a
#M = mv1a.MV1a(N_tar=N_obj,A = [[a_x,1],[0,a_v]],Sigma_O=[[sig_O**2]],
#            Sigma_D = [[sig_x**2,0],[0,sig_v**2]])
foo_t = 0

class FloatSlider(wx.Slider):
    def __init__(self, parent, ID, Min, Max, Step, Value, call_back, **kwargs):
        self.FMin = Min
        self.FMax = Max
        self.FStep = Step
        self.Fvalue = Value
        self.Call_Back = call_back
        MV = int((Max-Min)/Step)
        V = int((Value-Min)/Step)
        wx.Slider.__init__(self, parent, id=ID, value=V, minValue=0,
                           maxValue=MV, **kwargs)
        parent.Bind(wx.EVT_SLIDER, self.scale, self)
    def scale(self,event):
        Ivalue = self.GetValue()
        self.Fvalue = self.FMin + Ivalue*self.FStep
        self.Call_Back(event)
    
class DemoPlotPanel(demo.PlotPanel):
    """An example plotting panel. The only method that needs 
    overriding is the draw method"""
    def draw(self):
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        #Now draw it
        if self.x is None:
            return
        self.subplot.hold(True)
        self.subplot.clear()
        N = len(self.x)
        for k in xrange(N_obj):
            self.subplot.plot(self.x[k],self.y[k],
                 markerfacecolor=self.colors[k%len(self.colors)],
                              marker='o',linewidth=0)
            foo_x = [self.x[k][foo_t]]
            foo_y = [self.y[k][foo_t]]
            self.subplot.plot(foo_x,foo_y,
                  markerfacecolor=self.colors[k%len(self.colors)],
                  markeredgecolor=self.colors[k%len(self.colors)], marker='x',
                              markeredgewidth=2,markersize=25)
        if N > N_obj:
            for k in xrange(N_obj,N):
                self.subplot.plot(self.x[k],self.y[k], lw=2,
                           color=self.colors[k%len(self.colors)],
                           linestyle=self.linestyle)
                foo_x = [self.x[k][foo_t]]
                foo_y = [self.y[k][foo_t]]
                self.subplot.plot(foo_x,foo_y,
                  markerfacecolor=self.colors[k%len(self.colors)],
                  markeredgecolor=self.colors[k%len(self.colors)], marker='+',
                  markeredgewidth=2, markersize=25)
        #Set some plot attributes
        if self.title != None:
            self.subplot.set_title(self.title, fontsize = 12)
        if self.ylabel != None:
            self.subplot.set_ylabel(self.ylabel, fontsize = 10)
        if self.xlabel != None:
            self.subplot.set_xlabel(self.xlabel, fontsize = 10)

    def _forceDraw(self,x=None,y=None):
        if x != None:
            self.x=x
        if y != None:
            self.y=y
        self.draw()
        self._SetSize()

def layout(panel, compList, orient=wx.VERTICAL):
    box = wx.BoxSizer(orient)
    for c in compList:
        #box.Add(c, 0, wx.ALL, panel.SPACING)
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
                 **kwargs):
    # Vertical Float Slider
    frame = wx.Panel(parent,-1)
    Label = wx.StaticText(frame,-1,' '+label+' ')
    Slider = FloatSlider(frame, -1, Min, Max, Step, Value, call_back,
                         style=wx.SL_VERTICAL, **kwargs)
    layout(frame,[Label,Slider])
    return frame,Slider

class view_mv1_frame(wx.Frame):
    def __init__(self, parent):
        self.title = "MV* Viewer"
        wx.Frame.__init__(self, parent, -1, self.title)
        self.initStatusBar()
        self.controlPanel = ControlPanel(self, -1)
        self.plot_panelA = DemoPlotPanel(self,ylabel="Position",
               xlabel='Time',title='Observations', size=(400,800))
        #self.plot_panelA.linestyle = ':'
        self.plot_panelA.colors = N_obj*['black']
        self.plot_panelB = DemoPlotPanel(self,ylabel="Position",
               xlabel='Velocity',title='State Space', size=(400,800))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.controlPanel,0,wx.EXPAND,4)
        sizer.Add(self.plot_panelA,1,wx.EXPAND,4)
        sizer.Add(self.plot_panelB,1,wx.EXPAND,4)
        self.SetSizer(sizer)
        sizer.Fit(self)
        
    def initStatusBar(self):
        self.statusbar = self.CreateStatusBar()
        #self.statusbar.SetFieldsCount(3)
        #self.statusbar.SetStatusWidths([-1, -2, -3])

    def OnNew(self, event): pass

    def OnCloseWindow(self, event):
        self.Destroy()

    def DummyClicked(self, event):
        print 'DummyClicked'
        
    def Mv1Clicked(self, event):
        global Model_Class
        Model_Class = mv1a.MV1a
        
    def Mv2Clicked(self, event):
        global Model_Class
        Model_Class = mv2.MV2
        
    def OnSimClicked(self, event):
        global T,N_obj,a_x,a_v,sig_x,sig_v,sig_O,M,yo,s,MaxD,Model_Class

        M = Model_Class(N_tar=N_obj,A = [[a_x,1],[0,a_v]],Sigma_O=[[sig_O**2]],
            Sigma_D = [[sig_x**2,0],[0,sig_v**2]],MaxD=MaxD,MaxP=MaxP)
        yo,s = M.simulate(T)
        x = scipy.zeros((N_obj,T))
        y = scipy.zeros((N_obj,T))
        ts_x = []
        ts_y = []
        x = []
        y = []

        for k in xrange(N_obj):
            for List in (ts_x,ts_y,x,y):
                List.append(scipy.zeros(T))
            for t in xrange(T):
                x[k][t] = s[t][k][0,0]
                y[k][t] = s[t][k][1,0]
                if yo[t][k] is not None:
                    ts_x[k][t] = t+0.5
                    ts_y[k][t] = yo[t][k][0,0]
        self.plot_panelA._forceDraw(x=ts_x,y=ts_y)
        self.plot_panelB._forceDraw(x=y,y=x)

    def OnTrackClicked(self, event):
        global T,N_obj,M,yo,s,a_x,a_v,sig_O,sig_x,sig_v,MaxD,MaxP,Model_Class

        M = Model_Class(N_tar=N_obj,A = [[a_x,1],[0,a_v]],Sigma_O=[[sig_O**2]],
            Sigma_D = [[sig_x**2,0],[0,sig_v**2]],MaxD=MaxD,MaxP=MaxP)
        t_start = time.time()
        d = M.decode(yo)
        print 'decode time = %f'%(time.time()-t_start)

        x = scipy.zeros((N_obj,T))
        y = scipy.zeros((N_obj,T))
        ts_x = []
        ts_y = []
        x = []
        y = []

        for k in xrange(2*N_obj):
            for List in (x,y):
                List.append(scipy.zeros(T))
        for k in xrange(N_obj):
            for t in xrange(T):
                x[k][t] = s[t][k][0,0]
                y[k][t] = s[t][k][1,0]
                x[N_obj+k][t] = d[t][k][0,0]
                y[N_obj+k][t] = d[t][k][1,0]
        self.plot_panelB._forceDraw(x=y,y=x)
    
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
        self.statusbar.SetStatusText('MaxD=%5.3f'%MaxD)

    def MaxP_sliderUpdate(self, event):
        global MaxP
        MaxP = self.controlPanel.MaxP_Slider.GetValue()
        self.statusbar.SetStatusText('MaxP=%d'%MaxP)

    def t_sliderUpdate(self, event):
        global foo_t
        Ft = self.controlPanel.t_Slider.Fvalue
        foo_t = max(0,min(T-1,int(T*Ft)))
        self.plot_panelA._forceDraw()
        self.plot_panelB._forceDraw()
        self.statusbar.SetStatusText('t=%d'%foo_t)
        
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
        global T,N_obj,a_x,a_v,sig_x,sig_v,sig_O,MaxD,MaxP
        wx.Panel.__init__(self, parent, ID, style=wx.RAISED_BORDER)
        buttonSize = (self.BMP_SIZE + 2 * self.BMP_BORDER,
                      self.BMP_SIZE + 2 * self.BMP_BORDER)

        mv1Button = wx.Button(parent=self, id=-1, label='MV1')
        self.Bind(wx.EVT_BUTTON, parent.Mv1Clicked, mv1Button)

        mv2Button = wx.Button(parent=self, id=-1, label='MV2')
        self.Bind(wx.EVT_BUTTON, parent.Mv2Clicked, mv2Button)

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
                                maxValue=20, size=(-1, 200))
        T_frame,self.T_Slider = Vlab_slider(self, row_A, "T",
                                parent.T_sliderUpdate, value=T, minValue=1,
                                maxValue=100, size=(-1, 200))
        layout(row_A,[a_x_frame,a_v_frame,N_frame,T_frame],
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
        t_frame,self.t_Slider = VFlab_slider(self, row_C, "t", 0.0, 1.0,
                    0.005, 0, parent.t_sliderUpdate, size=(-1, 200))
        MaxD_frame,self.MaxD_Slider = VFlab_slider(self, row_C,"MaxD",
            0.0, 2.0, 0.002, MaxD, parent.MaxD_sliderUpdate, size=(-1, 200))
        MaxP_frame,self.MaxP_Slider = Vlab_slider(self, row_C,"MaxP",
            parent.MaxP_sliderUpdate, value=MaxP, minValue=1, maxValue=120,
            size=(-1, 200))
        layout(row_C,[t_frame,MaxD_frame,MaxP_frame],orient=wx.HORIZONTAL)
        
        layout(track_frame,[trackButtonA,row_C])
        #--------------------
        layout(self,[mv1Button,mv2Button,sim_frame,track_frame])

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
