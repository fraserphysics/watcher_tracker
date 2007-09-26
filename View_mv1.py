import wx
import matplotlib
matplotlib.interactive(False)
#matplotlib.interactive(True)
#Use the WxAgg back end. The Wx one takes too long to render
matplotlib.use('WXAgg')
import demo
from matplotlib.numerix import arange, sin, cos, pi


class DemoPlotPanel(demo.PlotPanel):
    """An example plotting panel. The only method that needs 
    overriding is the draw method"""
    def draw(self):
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        theta = arange(0, 45*2*pi, 0.02)
        rad = (0.8*theta/(2*pi)+1)
        r = rad*(8 + sin(theta*7+rad/1.8))
        x = r*cos(theta)
        y = r*sin(theta)
        #Now draw it
        self.subplot.plot(x,y, '-r')
        #Set some plot attributes
        self.subplot.set_title("A polar flower (%s points)"%len(x), fontsize = 12)
        self.subplot.set_xlabel("Flower is from  http://www.physics.emory.edu/~weeks/ideas/rose.html", fontsize = 8)
        self.subplot.set_xlim([-400, 400])
        self.subplot.set_ylim([-400, 400])

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
        #global plot_panel, frame
        print "Plot"
        #plot_panel.draw()
        #frame.Show()

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
