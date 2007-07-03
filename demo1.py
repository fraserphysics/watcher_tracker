"""
demo1.py 

"""
import numpy, filter1, os, math, wx

# Define the sequence of observations
r = """
#time y_0        y_1
--------------------
1     -2         2       
2     -2         2       
3     -1         1       
4     -0.1       0.1       
5     -0.1       0.1       
6     -1         1       
7     -2         2       
8     -2         2    
"""
# Read observations from string r
Ys = []
lines = r.split('\n')
for line in lines[3:-1]:
    part = line.split()
    Ys.append([float(part[1]),float(part[2])])

# Set model parameters:
O_noise = 0.5                         # Sqrt of Observation noise var
F = numpy.array([[1,1],[0,1]])        # Dynamical map
G = numpy.array([[1,0]])              # Observation map
D = numpy.array([[.25, 0],[0, .25]])  # Dynamical noise covariance
O = numpy.array([[O_noise**2]])       # Observation noise covariance
mu = numpy.array([0,0])               # Mean of initial distribution
Sigma = numpy.array([[4,0],[0,1]])    # Covariance of inital distribution
M = filter1.model1(F,G,D,O,mu,Sigma)

# Begin ugly gnuplot stuff
splot_preface = """
set hidden
set parametric
set contour base
set style data lines
set ylabel "x"
set xlabel "t"
set zlabel "P"
"""
gp_pipe = os.popen ("gnuplot",'w')
gp_pipe.write(splot_preface)
#gp_pipe.write('set cntrparam levels incremental 0, 2, 30 \n')
gp_pipe.write('set title"Filter Results"\n')

def splot_gp(mus,Sigmas):
    """ Surface plot of a sequence of Gaussian distributions."""
    File = open('/tmp/splot_data', 'w')
    Xs = numpy.arange(-4,4,.1)
    for t in xrange(len(mus)):
        mean = mus[t][0,0]
        var = Sigmas[t][0,0]
        for x in Xs:
            d = (x-mean)
            norm = 1/((2*math.pi*var)**.5)
            p = norm*math.exp(-d*d/(2*var))
            File.write('%d %f %f\n' % (t, x, p))
        File.write('\n')
    File.close()
    gp_pipe.write('splot "/tmp/splot_data"\n')
    gp_pipe.flush()
# End of ugly gnuplot stuff

# Do Kalman filtering
mus = [M.mu]
Sigmas = [M.Sigma]
def KF():
    global mus, Sigmas
    mus = [M.mu]
    Sigmas = [M.Sigma]
    for t in xrange(len(Ys)):
        yt = filter1.column(Ys[t][0])
        M.forward(yt)
        mus.append(M.mu)
        Sigmas.append(M.Sigma)
        print 'y[%2d]=%5.2f, u=(%5.2f,%5.2f), +/- %5.2f, LL=%5.2f'%(t,
          Ys[t][0],M.mu[0,0],M.mu[1,0],(M.Sigma[0,0])**.5,M.LogLike)

ID_ABOUT=101
ID_EXIT=110
ID_FILTER=111
ID_PLOT=112
class MainWindow(wx.Frame):
    def __init__(self,parent,id,title):
        wx.Frame.__init__(self,parent,wx.ID_ANY, title, size = (200,100))
        self.CreateStatusBar() # A StatusBar in the bottom of the window
        # Setting up the menu.
        filemenu= wx.Menu()
        filemenu.Append(ID_ABOUT, "&About"," Information about this program")
        filemenu.Append(ID_FILTER, "&Filter"," Kalman filter")
        filemenu.Append(ID_PLOT, "&Plot"," Plot the state distribution")
        filemenu.AppendSeparator()
        filemenu.Append(ID_EXIT,"E&xit"," Terminate the program")
        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        # attach the menu-event ID_ABOUT to the method self.OnAbout
        wx.EVT_MENU(self, ID_ABOUT, self.OnAbout)
        wx.EVT_MENU(self, ID_FILTER, self.OnFilter)
        wx.EVT_MENU(self, ID_PLOT, self.OnPlot)
        # attach the menu-event ID_EXIT to the method self.OnExit
        wx.EVT_MENU(self, ID_EXIT, self.OnExit)
        self.Show(True)
    def OnAbout(self,e):
        d= wx.MessageDialog( self, """ Derived from Sample Editor at
http://wiki.wxpython.org/Getting_Started""", "About", wx.OK)
                            # Create a message dialog box
        d.ShowModal() # Shows it
        d.Destroy() # finally destroy it when finished.
    def OnFilter(self,e):
        KF()
    def OnPlot(self,e):
        splot_gp(mus,Sigmas)
    def OnExit(self,e):
        self.Close(True)  # Close the frame.
app = wx.PySimpleApp()
frame = MainWindow(None, wx.ID_ANY, "Tracking Demo")
app.MainLoop()


#---------------
# Local Variables:
# eval: (python-mode)
# End:
