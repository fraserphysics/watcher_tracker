import wx, mv1, random, scipy
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
yo,s = M.simulate(T)
#d = M.decode(yo)

x_plot = scipy.zeros((2*N_obj,T))
y_plot = scipy.zeros((2*N_obj,T))

for t in xrange(T):
    for k in xrange(N_obj):
        x_plot[k,t] = s[t][k][0,0]
        y_plot[k,t] = s[t][k][1,0]
    """
    for k in xrange(len(y[t])):
        print ' k=%d  %4.2f  '%(k,y[t][k][0,0]),
        for f in (s[t][k],d[t][k]):
            print '(%4.2f, %4.2f)  '%(f[0,0],f[1,0])
    """

#redraw = True
colors = ['red','blue','green','magenta']
class DemoPlotPanel(demo.PlotPanel):
    """An example plotting panel. The only method that needs 
    overriding is the draw method"""
    def draw(self):
        global x_plot,y_plot,redraw
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        #Now draw it
        self.subplot.hold(True)
        #self.subplot.clear()
        for k in xrange(N_obj):
            self.subplot.plot(x_plot[k],y_plot[k], lw=6, color=colors[k])
        if x_plot[N_obj,0] != 0.0:
            for k in xrange(N_obj):
                self.subplot.plot(x_plot[N_obj+k],y_plot[N_obj+k], lw=3, color=colors[k])
#        if redraw:
#            self.draw()
#            redraw = False
        #Set some plot attributes
        self.subplot.set_title("A title)", fontsize = 12)
        self.subplot.set_xlabel("An xlabel", fontsize = 8)
        self.subplot.set_xlim([-10, 10])
        self.subplot.set_ylim([-5, 5])
        #self.Refresh()
        #self.subplot.hold(False)

class view_mv1_frame(wx.Frame):
    def __init__(self, parent):
        self.title = "MV1 Viewer"
        wx.Frame.__init__(self, parent, -1, self.title)
        self.initStatusBar()
        self.controlPanel = ControlPanel(self, -1)
        self.plot_panel = DemoPlotPanel(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        #plot_panel.SetSizer(sizer)
        sizer.Add(self.controlPanel)
        sizer.Add(self.plot_panel,1,wx.EXPAND)
        self.SetSizer(sizer)
        
    def initStatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        self.statusbar.SetStatusWidths([-1, -2, -3])

    def OnNew(self, event): pass

    def OnCloseWindow(self, event):
        self.Destroy()

    def OnPlotClicked(self, event):
        global x_plot,y_plot,T,N_obj,M,yo,s

        d = M.decode(yo)

        for t in xrange(T):
            for k in xrange(N_obj):
                x_plot[k,t] = s[t][k][0,0]
                y_plot[k,t] = s[t][k][1,0]
                x_plot[N_obj+k,t] = d[t][k][0,0]
                y_plot[N_obj+k,t] = d[t][k][1,0]
        redraw = False
        self.plot_panel._forceDraw()

class ControlPanel(wx.Panel):

    BMP_SIZE = 16
    BMP_BORDER = 3
    NUM_COLS = 4
    SPACING = 4

    colorList = ('Black', 'Yellow', 'Red', 'Green', 'Blue', 'Purple',
              'Brown', 'Aquamarine', 'Forest Green', 'Light Blue',
              'Goldenrod', 'Cyan', 'Orange', 'Navy', 'Dark Grey',
              'Light Grey')
    maxThickness = 16

    def __init__(self, parent, ID):
        wx.Panel.__init__(self, parent, ID, style=wx.RAISED_BORDER)
        buttonSize = (self.BMP_SIZE + 2 * self.BMP_BORDER,
                      self.BMP_SIZE + 2 * self.BMP_BORDER)
        plotButton = wx.Button(parent=self, id=-1, label='Plot')
        self.Bind(wx.EVT_BUTTON, parent.OnPlotClicked, plotButton)
        runButton = wx.Button(parent=self, id=-1, label='Run')
        self.Bind(wx.EVT_BUTTON, parent.OnPlotClicked, runButton)
        viewButton = wx.Button(parent=self, id=-1, label='Decode')
        self.Bind(wx.EVT_BUTTON, parent.OnPlotClicked, viewButton)

        fooLabel = wx.StaticText(self,-1,"foo")
        fooSlider = wx.Slider(parent=self, id=-1, style=wx.SL_VERTICAL, value=0, minValue=0, maxValue=100, size=(-1, 100))
        fooSlider.SetTickFreq(20, 1)
        self.fooSlider = fooSlider

        self.Bind(wx.EVT_SLIDER, self.sliderUpdate)

        self.layout(None, None, [plotButton, runButton, viewButton, fooLabel, fooSlider])

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
