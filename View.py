import wx, mv1a, demo, random, scipy, time
import matplotlib
#matplotlib.interactive(False)
matplotlib.interactive(True)
#Use the WxAgg back end. The Wx one takes too long to render
matplotlib.use('WXAgg')

random.seed(3)

T = 20
N_obj = 4
M = mv1a.MV1a(N_tar=N_obj,A = [[0.81,1],[0,.95]],Sigma_O=[[0.01]],
            Sigma_D = [[0.0001,0],[0,0.1]])
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

class view_mv1_frame(wx.Frame):
    def __init__(self, parent):
        self.title = "MV1 Viewer"
        wx.Frame.__init__(self, parent, -1, self.title)
        self.initStatusBar()
        self.controlPanel = ControlPanel(self, -1)
        self.plot_panelA = DemoPlotPanel(self,ylabel="Position",
               xlabel='Time',title='Observations', size=(500,1000))
        #self.plot_panelA.linestyle = ':'
        self.plot_panelA.colors = N_obj*['black']
        self.plot_panelB = DemoPlotPanel(self,ylabel="Position",
               xlabel='Velocity',title='State Space', size=(500,1000))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.controlPanel,0,wx.EXPAND,4)
        sizer.Add(self.plot_panelA,1,wx.EXPAND,4)
        sizer.Add(self.plot_panelB,1,wx.EXPAND,4)
        self.SetSizer(sizer)
        sizer.Fit(self)
        
    def initStatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        self.statusbar.SetStatusWidths([-1, -2, -3])

    def OnNew(self, event): pass

    def OnCloseWindow(self, event):
        self.Destroy()

    def OnSimClicked(self, event):
        global T,N_obj,M,yo,s

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

    def sliderUpdate(self, event):
        global foo_t
        foo_t = self.controlPanel.fooSlider.GetValue()/2
        self.plot_panelA._forceDraw()
        self.plot_panelB._forceDraw()

    def t_sliderUpdate(self, event):
        global foo_t
        foo_t = self.controlPanel.t_Slider.GetValue()/2
        self.plot_panelA._forceDraw()
        self.plot_panelB._forceDraw()

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
        
        a_x_frame = wx.Panel(row_A,-1)
        a_x_Label = wx.StaticText(a_x_frame,-1,"a_x")
        a_x_Slider = wx.Slider(a_x_frame, id=-1, style=wx.SL_VERTICAL,
                     value=0, minValue=0, maxValue=2*(T-1), size=(-1, 200))
        a_x_Slider.SetTickFreq(T, 1)
        layout(a_x_frame,[a_x_Label,a_x_Slider])

        a_v_frame = wx.Panel(row_A,-1)
        a_v_Label = wx.StaticText(a_v_frame,-1,"a_v")
        a_v_Slider = wx.Slider(a_v_frame, id=-1, style=wx.SL_VERTICAL,
                     value=0, minValue=0, maxValue=2*(T-1), size=(-1, 200))
        a_v_Slider.SetTickFreq(T, 1)
        layout(a_v_frame,[a_v_Label,a_v_Slider])
        
        layout(row_A,[a_x_frame,a_v_frame],orient=wx.HORIZONTAL)

        row_B = wx.Panel(sim_frame,-1)
        
        sig_x_frame = wx.Panel(row_B,-1)
        sig_x_Label = wx.StaticText(sig_x_frame,-1,"sig_x")
        sig_x_Slider = wx.Slider(sig_x_frame, id=-1, style=wx.SL_VERTICAL,
                     value=0, minValue=0, maxValue=2*(T-1), size=(-1, 200))
        sig_x_Slider.SetTickFreq(T, 1)
        layout(sig_x_frame,[sig_x_Label,sig_x_Slider])

        sig_v_frame = wx.Panel(row_B
                               ,-1)
        sig_v_Label = wx.StaticText(sig_v_frame,-1,"sig_v")
        sig_v_Slider = wx.Slider(sig_v_frame, id=-1, style=wx.SL_VERTICAL,
                     value=0, minValue=0, maxValue=2*(T-1), size=(-1, 200))
        sig_v_Slider.SetTickFreq(T, 1)
        layout(sig_v_frame,[sig_v_Label,sig_v_Slider])
        
        layout(row_B,[sig_x_frame,sig_v_frame],orient=wx.HORIZONTAL)

        layout(sim_frame,[simButtonA,row_A,row_B])
        #---------------------------
        track_frame = wx.Panel(self,-1)
        
        trackButtonA = wx.Button(parent=track_frame, id=-1, label='Track')
        self.Bind(wx.EVT_BUTTON, parent.OnTrackClicked, trackButtonA)

        row_C = wx.Panel(track_frame,-1)
        
        t_frame = wx.Panel(row_C,-1)
        t_Label = wx.StaticText(t_frame,-1,"t")
        t_Slider = wx.Slider(t_frame, id=-1, style=wx.SL_VERTICAL,
                     value=0, minValue=0, maxValue=2*(T-1), size=(-1, 200))
        t_Slider.SetTickFreq(T, 1)
        self.t_Slider = t_Slider
        self.Bind(wx.EVT_SLIDER, parent.t_sliderUpdate)
        layout(t_frame,[t_Label,t_Slider])
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
