import wx

import random

import mv1, time

N_obj=4
M = mv1.MV1(N_obj=N_obj)
T = 20
y,s = M.simulate(T)
d = None

# Adapted from Rappin's book.

class ViewDotsWindow(wx.Window):
    def __init__(self, parent, ID):
        wx.Window.__init__(self, parent, ID)
        self.SetBackgroundColour("White")
        self.color = "Black"
        self.thickness = 1
        self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)
        self.lines = []
        self.curLine = []
        self.pos = (0, 0)
        self.dotCenters = []
        self.dotColors = []
        self.availableColors = ['Red', 'Black', 'Green', 'Blue', 'Purple']
        self.RandomizeDots()
        self.InitBuffer()
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def InitBuffer(self):
        size = self.GetClientSize()
        self.buffer = wx.EmptyBitmap(size.width, size.height)
        dc = wx.BufferedDC(None, self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        self.DrawLines(dc)
        self.DrawDots(dc, self.dotCenters, self.dotColors)
        self.reInitBuffer = False

    def GetLinesData(self):
        return self.lines[:]

    def SetLinesData(self, lines):
        self.lines = lines[:]
        self.InitBuffer()
        self.Refresh()

    def OnSize(self, event):
        self.reInitBuffer = True

    # Call this with
    #
    # SetDots([(x1, y1), (x2, y2), ..., (xn yn)], ['Black', 'Green', 'Blue', ...]
    def SetDots(self, cntrs, clrs):
        self.dotCenters = cntrs
        self.dotColors = clrs
        self.reInitBuffer = True

    def RandomizeDots(self):
        #sz = self.GetClientSizeTuple()
        self.dotColors = self.availableColors
        self.dotCenters = []
        for i in xrange(0,4):
            self.dotCenters.append((random.randint(0,800), random.randint(0,600)))

    def OnIdle(self, event):
        if self.reInitBuffer:
            self.InitBuffer()
            self.Refresh(False)

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self, self.buffer)

    def DrawDots(self, dc, dot_centers, dot_colors):
        for i in xrange(0,len(dot_centers)):
            c = dot_centers[i]
            clr = dot_colors[i]
            pen = wx.Pen(clr, 10, wx.SOLID)
            dc.SetPen(pen)
            dc.DrawCircle(c[0], c[1], 5)

    def DrawLines(self, dc):
        for colour, thickness, line in self.lines:
            dc.SetPen(pen)
            for coords in line:
                dc.DrawLine(*coords)

    def SetColor(self, color):
        self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)

    def SetThickness(self, num):
        self.thickness = num
        self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)

    def OnPerformClicked(self, event):
        print "Perform"
        self.RandomizeDots()
        #self.Refresh()
        dc = wx.BufferedDC(None, self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        #self.DrawLines(dc)
        self.DrawDots(dc, self.dotCenters, self.dotColors)
        self.Refresh(False)
        self.reInitBuffer = False

    def RunClicked(self, event):
        global y,s

        ts = time.time()
        y,s = M.simulate(20)
        print 'elapsed time=',time.time()-ts
        #self.reInitBuffer = True

    def DecodeClicked(self, event):
        global M,y,d
        
        ts = time.time()
        d = M.decode(y)
        print 'elapsed time=',time.time()-ts

    def OnSetSliderValue(self, val):
        global T, N_obj
        
        self.sliderVal = val
        t = min((20*val)/100,T-1)
        Frame_t = (800*t)/T + 5
        y_dots = []
        for k in xrange(0,N_obj):
            ytk = y[t][k][0,0]
            Frame_y = int(400+40*ytk)
            y_dots.append([Frame_t,Frame_y])
        s_dots = []
        for k in xrange(0,N_obj):
            stk = s[t][k]
            x_0 = stk[0,0]
            x_1 = stk[1,0]
            s_dots.append([x_1*100+600,x_0*40+400])
        self.Refresh()
        dc = wx.BufferedDC(None, self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        self.DrawDots(dc, y_dots+s_dots, N_obj*['Purple']+self.availableColors)


class ViewDotsFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, "ViewDots Frame", \
                          size=(900,1000))
        self.viewDots = ViewDotsWindow(self, -1)

if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = ViewDotsFrame(None)
    frame.Show(True)
    app.MainLoop()
