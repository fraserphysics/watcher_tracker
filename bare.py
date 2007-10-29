import matplotlib.figure, matplotlib.backends.backend_wxagg, wx

class BarePlotPanel(wx.Panel):
    """
    I've cut http://new.scipy.org/Matplotlib_figure_in_a_wx_panel down
    to a bare minimum
    """
    def __init__(self, parent, id = -1, style = wx.NO_FULL_REPAINT_ON_RESIZE,
                 **kwargs):
        wx.Panel.__init__(self, parent, id = id, style = style, **kwargs)
        self.figure = matplotlib.figure.Figure()
        self.canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            self, -1, self.figure)
        self.axes = self.figure.add_subplot(111)
        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

    def _onIdle(self, evt):
        if self._resizeflag:
            pixels = self.GetClientSize()
            self.canvas.SetSize(pixels)
            self.figure.set_size_inches(pixels[0]/self.figure.get_dpi(),
                                        pixels[1]/self.figure.get_dpi())
            self.draw()
            self._resizeflag = False
    def _onSize(self, event):
        self._resizeflag = True
    def draw(self):
        x = [0,1,2,3,4,5,6,7,8,9]
        y = [0,1,4,9,16,25,36,49,64,81]
        self.axes.plot(x,y, '-r')

app = wx.PySimpleApp(0)
#Initialise a frame ...
frame = wx.Frame(None, -1, 'WxPython and Matplotlib')
#Make a child plot panel...
panel = BarePlotPanel(frame)
#Put it in a sizer ...   
sizer = wx.BoxSizer(wx.HORIZONTAL)
panel.SetSizer(sizer)
sizer.SetItemMinSize(panel, 300, 300)
panel.Fit()   
frame.Show()
panel._resizeflag = True
panel._onIdle(None) # Force initial draw
app.MainLoop()
