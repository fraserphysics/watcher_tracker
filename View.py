import wx, mv1a, demo, random, scipy, time
import matplotlib
#matplotlib.interactive(False)
matplotlib.interactive(True)
#Use the WxAgg back end. The Wx one takes too long to render
matplotlib.use('WXAgg')

random.seed(3)

T = 20
N_obj = 4
a_x = 0.81
a_v = 0.95
sig_x = 0.01
sig_v = 0.31
sig_O = 0.1
#M = mv1a.MV1a(N_tar=N_obj,A = [[a_x,1],[0,a_v]],Sigma_O=[[sig_O**2]],
#            Sigma_D = [[sig_x**2,0],[0,sig_v**2]])
foo_t = 0

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
                 markerfacecolor=self.colors[k], marker='o',linewidth=0)
            foo_x = [self.x[k][foo_t]]
            foo_y = [self.y[k][foo_t]]
            self.subplot.plot(foo_x,foo_y,markerfacecolor=self.colors[k],
                              markeredgecolor=self.colors[k], marker='x',
                              markeredgewidth=2,markersize=25)
        if N > N_obj:
            for k in xrange(N_obj,N):
                self.subplot.plot(self.x[k],self.y[k], lw=2,
                           color=self.colors[k], linestyle=self.linestyle)
                foo_x = [self.x[k][foo_t]]
                foo_y = [self.y[k][foo_t]]
                self.subplot.plot(foo_x,foo_y,markerfacecolor=self.colors[k],
                     markeredgecolor=self.colors[k], marker='+',
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

def Vlab_slider(parent, label, **kwargs): # Vertical slider and
                                             # label in frame
    frame = wx.Panel(parent,-1)
    Label = wx.StaticText(frame,-1,' '+label+' ')
    Slider = wx.Slider(frame, id=-1, style=wx.SL_VERTICAL, **kwargs)
    layout(frame,[Label,Slider])
    return frame

def Vlab_sliderA(binder, parent, label, call_back, **kwargs):
    frame = wx.Panel(parent,-1)
    Label = wx.StaticText(frame,-1,' '+label+' ')
    Slider = wx.Slider(frame, id=-1, style=wx.SL_VERTICAL, **kwargs)
    layout(frame,[Label,Slider])
    binder.Bind(wx.EVT_SLIDER, call_back)
    return frame,Slider

class view_mv1_frame(wx.Frame):
    def __init__(self, parent):
        self.title = "MV1 Viewer"
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

    def OnSimClicked(self, event):
        global T,N_obj,a_x,a_v,sig_x,sig_v,sig_O,M,yo,s

        M = mv1a.MV1a(N_tar=N_obj,A = [[a_x,1],[0,a_v]],Sigma_O=[[sig_O**2]],
            Sigma_D = [[sig_x**2,0],[0,sig_v**2]])
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
                ts_x[k][t] = t+0.5
                ts_y[k][t] = yo[t][k][0,0]
        self.plot_panelA._forceDraw(x=ts_x,y=ts_y)
        self.plot_panelB._forceDraw(x=y,y=x)

    def OnTrackClicked(self, event):
        global T,N_obj,M,yo,s

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
        a_x = self.controlPanel.a_x_Slider.GetValue()
        print 'a_x=%f'%a_x
        self.statusbar.SetStatusText('a_x=%f'%a_x)

    def t_sliderUpdate(self, event):
        global foo_t
        foo_t = self.controlPanel.t_Slider.GetValue()/2
        self.plot_panelA._forceDraw()
        self.plot_panelB._forceDraw()
        self.statusbar.SetStatusText('t=%d'%foo_t)

class ControlPanel(wx.Panel):

    BMP_SIZE = 16
    BMP_BORDER = 3
    NUM_COLS = 4
    SPACING = 4

    maxThickness = 16

    def __init__(self, parent, ID):
        wx.Panel.__init__(self, parent, ID, style=wx.RAISED_BORDER)
        buttonSize = (self.BMP_SIZE + 2 * self.BMP_BORDER,
                      self.BMP_SIZE + 2 * self.BMP_BORDER)

        sim_frame = wx.Panel(self,-1)
        
        simButtonA = wx.Button(parent=sim_frame, id=-1, label='Simulate')

        self.Bind(wx.EVT_BUTTON, parent.OnSimClicked, simButtonA)

        row_A = wx.Panel(sim_frame,-1)

        a_x_frame,self.a_x_Slider = Vlab_sliderA(self,row_A,"a_x",
            parent.a_x_sliderUpdate, value=0.8, minValue=0.6,maxValue=1.0,
            size=(-1, 200))
        a_v_frame = Vlab_slider(row_A,"a_v",value=0, minValue=0,
                                maxValue=2*(T-1), size=(-1, 200))
        N_frame = Vlab_slider(row_A,"N",value=0, minValue=0,
                                maxValue=2*(T-1), size=(-1, 200))
        T_frame = Vlab_slider(row_A,"T",value=0, minValue=0,
                                maxValue=2*(T-1), size=(-1, 200))
        layout(row_A,[a_x_frame,a_v_frame,N_frame,T_frame],
               orient=wx.HORIZONTAL)

        row_B = wx.Panel(sim_frame,-1)

        sig_x_frame = Vlab_slider(row_B,"sig_x",
                     value=0, minValue=0, maxValue=2*(T-1), size=(-1, 200))
        sig_v_frame = Vlab_slider(row_B,"sig_v", value=0, minValue=0,
                                  maxValue=2*(T-1), size=(-1, 200))
        sig_O_frame = Vlab_slider(row_B,"sig_O",
                     value=0, minValue=0, maxValue=2*(T-1), size=(-1, 200))
        
        layout(row_B,[sig_x_frame,sig_v_frame,sig_O_frame],
               orient=wx.HORIZONTAL)

        layout(sim_frame,[simButtonA,row_A,row_B])
        #---------------------------
        track_frame = wx.Panel(self,-1)
        
        trackButtonA = wx.Button(parent=track_frame, id=-1, label='Track')
        self.Bind(wx.EVT_BUTTON, parent.OnTrackClicked, trackButtonA)

        row_C = wx.Panel(track_frame,-1)
        t_frame,self.t_Slider = Vlab_sliderA(self, row_C, "t",
                                parent.t_sliderUpdate, value=0, minValue=0,
                                maxValue=2*(T-1), size=(-1, 200))
        MaxD_frame = Vlab_slider(row_C,"MaxD",value=0, minValue=0,
                                maxValue=2*(T-1), size=(-1, 200))
        MaxP_frame = Vlab_slider(row_C,"MaxP",value=0, minValue=0,
                                maxValue=2*(T-1), size=(-1, 200))
        layout(row_C,[t_frame,MaxD_frame,MaxP_frame],orient=wx.HORIZONTAL)
        
        layout(track_frame,[trackButtonA,row_C])
        #--------------------
        layout(self,[sim_frame,track_frame])

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
