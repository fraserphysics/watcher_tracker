import wx, mv1, random, scipy, time
import matplotlib
#matplotlib.interactive(False)
matplotlib.interactive(True)
#Use the WxAgg back end. The Wx one takes too long to render
matplotlib.use('WXAgg')
import demo
from matplotlib.numerix import arange, sin, cos, pi

random.seed(3)

T = 20
N_obj = 4
M = mv1.MV1(N_obj=N_obj,A = [[0.81,1],[0,.95]],Sigma_O=[[0.01]],Sigma_D = [[0.0001,0],[0,0.1]])

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
        for k in xrange(4):
            #self.subplot.plot(self.x[k],self.y[k], color=self.colors[k], marker='o',linewidth=0)
            self.subplot.plot(self.x[k],self.y[k],markerfacecolor=self.colors[k], marker='o',linewidth=0)
        if N > 4:
            for k in xrange(4,N):
                self.subplot.plot(self.x[k],self.y[k], lw=2, color=self.colors[k], linestyle=self.linestyle)
        #Set some plot attributes
        self.subplot.set_title("A title", fontsize = 12)
        self.subplot.set_xlabel("An xlabel", fontsize = 8)
        #self.subplot.set_xlim([-10, 10])
        #self.subplot.set_ylim([-5, 5])
        #self.Refresh()
        #self.subplot.hold(False)

    def _forceDraw(self,x,y):
        self.x=x
        self.y=y
        self.draw()
        self._SetSize()

class view_mv1_frame(wx.Frame):
    def __init__(self, parent):
        self.title = "MV1 Viewer"
        wx.Frame.__init__(self, parent, -1, self.title)
        self.initStatusBar()
        self.controlPanel = ControlPanel(self, -1)
        self.plot_panelA = DemoPlotPanel(self)
        #self.plot_panelA.linestyle = ':'
        self.plot_panelA.colors = 4*['black']
        self.plot_panelB = DemoPlotPanel(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        #plot_panel.SetSizer(sizer)
        sizer.Add(self.controlPanel)
        sizer.Add(self.plot_panelA,1,wx.EXPAND)
        sizer.Add(self.plot_panelB,1,wx.EXPAND)
        self.SetSizer(sizer)
        
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
        ts_x = scipy.zeros((N_obj,T))
        ts_y = scipy.zeros((N_obj,T))
        x = scipy.zeros((N_obj,T))
        y = scipy.zeros((N_obj,T))

        for t in xrange(T):
            for k in xrange(N_obj):
                x[k,t] = s[t][k][0,0]
                y[k,t] = s[t][k][1,0]
                ts_x[k,t] = t
                ts_y[k,t] = yo[t][k][0,0]
        self.plot_panelA._forceDraw(ts_x,ts_y)
        self.plot_panelB._forceDraw(x,y)

    def OnTrackClicked(self, event):
        global T,N_obj,M,yo,s

        x = scipy.zeros((2*N_obj,T))
        y = scipy.zeros((2*N_obj,T))
        t_start = time.time()
        d = M.decode(yo)
        print 'decode time = %f'%(time.time()-t_start)

        for t in xrange(T):
            for k in xrange(N_obj):
                x[k,t] = s[t][k][0,0]
                y[k,t] = s[t][k][1,0]
                x[N_obj+k,t] = d[t][k][0,0]
                y[N_obj+k,t] = d[t][k][1,0]
        self.plot_panelB._forceDraw(x,y)

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
        simButton = wx.Button(parent=self, id=-1, label='Simulate')
        self.Bind(wx.EVT_BUTTON, parent.OnSimClicked, simButton)
        trackButton = wx.Button(parent=self, id=-1, label='Track')
        self.Bind(wx.EVT_BUTTON, parent.OnTrackClicked, trackButton)

        fooLabel = wx.StaticText(self,-1,"foo")
        fooSlider = wx.Slider(parent=self, id=-1, style=wx.SL_VERTICAL, value=0, minValue=0, maxValue=100, size=(-1, 100))
        fooSlider.SetTickFreq(20, 1)
        self.fooSlider = fooSlider

        self.Bind(wx.EVT_SLIDER, self.sliderUpdate)

        self.layout(None, None, [simButton, trackButton, fooLabel, fooSlider])

    def sliderUpdate(self, event):
        pass

    def layout(self, colorGrid, thicknessGrid, compList):
        box = wx.BoxSizer(wx.VERTICAL)
#        box.Add(colorGrid, 0, wx.ALL, self.SPACING)
#        box.Add(thicknessGrid, 0, wx.ALL, self.SPACING)
        for c in compList:
            box.Add(c, 0, wx.ALL, self.SPACING)
        self.SetSizer(box)
        box.Fit(self)

    def OnSetColour(self, event):
        color = self.colorMap[event.GetId()]

    def OnSetThickness(self, event):
        thickness = self.thicknessIdMap[event.GetId()]


class view_mv1_app(wx.App):
    def OnInit(self):
        global frame
        #Initialise a frame ...
        #frame = wx.Frame(None, -1, 'WxPython and Matplotlib')
        frame = view_mv1_frame(None)
        #And we are done ...    
        frame.Show()
        return True

app = view_mv1_app(False)
app.MainLoop()
   
#---------------
# Local Variables:
# eval: (python-mode)
# End:
