import wx
from wx.lib import buttons

from ViewDotsWindow import ViewDotsWindow

class ViewDotsFrame(wx.Frame):
    def __init__(self, parent):
        self.title = "ViewDots Frame"
        wx.Frame.__init__(self, parent, -1, self.title,
                size=(900,1000))
        self.filename = ""
        self.viewDots = ViewDotsWindow(self, -1)
        wx.EVT_MOTION(self.viewDots, self.OnViewDotsMotion)
        self.initStatusBar()
#        self.createMenuBar()
#        self.createToolBar()
        self.createPanel()

    def createPanel(self):
        controlPanel = ControlPanel(self, -1, self.viewDots)
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(controlPanel, 0, wx.EXPAND)
        box.Add(self.viewDots, 1, wx.EXPAND)
        self.SetSizer(box)

    def initStatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        self.statusbar.SetStatusWidths([-1, -2, -3])

    def OnViewDotsMotion(self, event):
        self.statusbar.SetStatusText("Pos: %s" %
                str(event.GetPositionTuple()), 0)
        self.statusbar.SetStatusText("Current Pts: %s" %
                len(self.viewDots.curLine), 1)
        self.statusbar.SetStatusText("Line Count: %s" %
                len(self.viewDots.lines), 2)
        event.Skip()

    def menuData(self):
        return [("&File", (
                    ("&New", "New ViewDots file", self.OnNew),
                    ("&Open", "Open viewDots file", self.OnOpen),
                    ("&Save", "Save viewDots file", self.OnSave),
                    ("", "", ""),
                    ("&Color", (
                        ("&Black", "", self.OnColor, wx.ITEM_RADIO),
                        ("&Red", "", self.OnColor, wx.ITEM_RADIO),
                        ("&Green", "", self.OnColor, wx.ITEM_RADIO),
                        ("&Blue", "", self.OnColor, wx.ITEM_RADIO),
                        ("&Other...", "", self.OnOtherColor, wx.ITEM_RADIO))),
                    ("", "", ""),
                    ("&Quit", "Quit", self.OnCloseWindow)))]

    def createMenuBar(self):
        menuBar = wx.MenuBar()
        for eachMenuData in self.menuData():
            menuLabel = eachMenuData[0]
            menuItems = eachMenuData[1]
            menuBar.Append(self.createMenu(menuItems), menuLabel)
        self.SetMenuBar(menuBar)

    def createMenu(self, menuData):
        menu = wx.Menu()
        for eachItem in menuData:
            if len(eachItem) == 2:
                label = eachItem[0]
                subMenu = self.createMenu(eachItem[1])
                menu.AppendMenu(wx.NewId(), label, subMenu)
            else:
                self.createMenuItem(menu, *eachItem)
        return menu

    def createMenuItem(self, menu, label, status, handler, kind=wx.ITEM_NORMAL):
        if not label:
            menu.AppendSeparator()
            return
        menuItem = menu.Append(-1, label, status, kind)
        self.Bind(wx.EVT_MENU, handler, menuItem)

    def createToolBar(self):
        toolbar = self.CreateToolBar()
        for each in self.toolbarData():
            self.createSimpleTool(toolbar, *each)
        toolbar.AddSeparator()
        for each in self.toolbarColorData():
            self.createColorTool(toolbar, each)
        toolbar.Realize()

    def createSimpleTool(self, toolbar, label, filename, help, handler):
        if not label:
            toolbar.AddSeparator()
            return
        bmp = wx.Image(filename, wx.BITMAP_TYPE_BMP).ConvertToBitmap()
        tool = toolbar.AddSimpleTool(-1, bmp, label, help)
        self.Bind(wx.EVT_MENU, handler, tool)

    def toolbarData(self):
        return (("New", "new.bmp", "Create new viewDots", self.OnNew),
                ("", "", "", ""),
                ("Open", "open.bmp", "Open existing viewDots", self.OnOpen),
                ("Save", "save.bmp", "Save existing viewDots", self.OnSave))

    def createColorTool(self, toolbar, color):
        bmp = self.MakeBitmap(color)
        tool = toolbar.AddRadioTool(-1, bmp, shortHelp=color)
        self.Bind(wx.EVT_MENU, self.OnColor, tool)

    def MakeBitmap(self, color):
        bmp = wx.EmptyBitmap(16, 15)
        dc = wx.MemoryDC()
        dc.SelectObject(bmp)
        dc.SetBackground(wx.Brush(color))
        dc.Clear()
        dc.SelectObject(wx.NullBitmap)
        return bmp

    def toolbarColorData(self):
        return ("Black", "Red", "Green", "Blue")

    def OnNew(self, event): pass

    def OnColor(self, event):
        menubar = self.GetMenuBar()
        itemId = event.GetId()
        item = menubar.FindItemById(itemId)
        if not item:
            toolbar = self.GetToolBar()
            item = toolbar.FindById(itemId)
            color = item.GetShortHelp()
        else:
            color = item.GetLabel()
        self.viewDots.SetColor(color)

    def OnCloseWindow(self, event):
        self.Destroy()

    def OnOtherColor(self, event):
        dlg = wx.ColourDialog(frame)
        dlg.GetColourData().SetChooseFull(True)
        if dlg.ShowModal() == wx.ID_OK:
            self.viewDots.SetColor(dlg.GetColourData().GetColour())
        dlg.Destroy()

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

    def __init__(self, parent, ID, viewDots):
        wx.Panel.__init__(self, parent, ID, style=wx.RAISED_BORDER)
        self.viewDots = viewDots
        buttonSize = (self.BMP_SIZE + 2 * self.BMP_BORDER,
                      self.BMP_SIZE + 2 * self.BMP_BORDER)
        colorGrid = self.createColorGrid(parent, buttonSize)
#        thicknessGrid = self.createThicknessGrid(buttonSize)
        performButton = wx.Button(parent=self, id=-1, label='Perform')
        self.Bind(wx.EVT_BUTTON, viewDots.OnPerformClicked, performButton)
        runButton = wx.Button(parent=self, id=-1, label='Run')
        self.Bind(wx.EVT_BUTTON, viewDots.RunClicked, runButton)
        viewButton = wx.Button(parent=self, id=-1, label='Decode')
        self.Bind(wx.EVT_BUTTON, viewDots.DecodeClicked, viewButton)

        fooLabel = wx.StaticText(self,-1,"foo")
        fooSlider = wx.Slider(parent=self, id=-1, style=wx.SL_VERTICAL, value=0, minValue=0, maxValue=100, size=(-1, 100))
        fooSlider.SetTickFreq(20, 1)
        self.fooSlider = fooSlider

        self.Bind(wx.EVT_SLIDER, self.sliderUpdate)

        self.layout(None, None, [performButton, runButton, viewButton, fooLabel, fooSlider])

    def sliderUpdate(self, event):
        self.viewDots.OnSetSliderValue(self.fooSlider.GetValue())

    def createColorGrid(self, parent, buttonSize):
        self.colorMap = {}
        self.colorButtons = {}
        colorGrid = wx.GridSizer(cols=self.NUM_COLS, hgap=2, vgap=2)
        for eachColor in self.colorList:
            bmp = parent.MakeBitmap(eachColor)
            b = buttons.GenBitmapToggleButton(self, -1, bmp, size=buttonSize)
            b.SetBezelWidth(1)
            b.SetUseFocusIndicator(False)
            self.Bind(wx.EVT_BUTTON, self.OnSetColour, b)
            colorGrid.Add(b, 0)
            self.colorMap[b.GetId()] = eachColor
            self.colorButtons[eachColor] = b
        self.colorButtons[self.colorList[0]].SetToggle(True)
        return colorGrid

    def createThicknessGrid(self, buttonSize):
        self.thicknessIdMap = {}
        self.thicknessButtons = {}
#        thicknessGrid = wx.GridSizer(cols=self.NUM_COLS, hgap=2, vgap=2)
        for x in range(1, self.maxThickness + 1):
            b = buttons.GenToggleButton(self, -1, str(x), size=buttonSize)
            b.SetBezelWidth(1)
            b.SetUseFocusIndicator(False)
#            self.Bind(wx.EVT_BUTTON, self.OnSetThickness, b)
#            thicknessGrid.Add(b, 0)
            self.thicknessIdMap[b.GetId()] = x
            self.thicknessButtons[x] = b
        self.thicknessButtons[1].SetToggle(True)
        return thicknessGrid

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
        if color != self.viewDots.color:
            self.colorButtons[self.viewDots.color].SetToggle(False)
        self.viewDots.SetColor(color)

    def OnSetThickness(self, event):
        thickness = self.thicknessIdMap[event.GetId()]
        if thickness != self.viewDots.thickness:
            self.thicknessButtons[self.viewDots.thickness].SetToggle(False)
        self.viewDots.SetThickness(thickness)


class ViewDotsApp(wx.App):

    def OnInit(self):
        frame = ViewDotsFrame(None)
        frame.Show(True)
        self.SetTopWindow(frame)
        return True

if __name__ == '__main__':
    app = ViewDotsApp(False)
    app.MainLoop()
