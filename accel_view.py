""" accel_view.py

A GUI for looking at the accelerations in AMF_accel.txt

Derived from GPL'ed file: resp_view.py
"""
import os, scipy, pylab, Tkinter, math
pylab.ion()

# Read the data
file_name = 'AMF_accel.txt'
file = open(file_name,'r')
v = []
e_x = []
e_v = []
for line in file.readlines():
    f = []
    # f[0]=vel_x, f[1]=vel_y, f[2]=e_x, f[3]=e_x_dot, f[4]=e_y, f[5]=e_y_dot
    for part in line.split():
        f.append(float(part))
    v.append(scipy.array([f[0],f[1]]))
    e_x.append(scipy.array([f[2],f[4]]))
    e_v.append(scipy.array([f[3],f[5]]))
T = len(v)

# calculate vectors to plot: (v_mag, e_v_parallel, e_v_perpendicular)
debug = open('debug','w') # debug
vecs = scipy.matrix(scipy.zeros((T,3)))
for t in xrange(T):
    vv = scipy.dot(v[t],v[t])
    if vv < 1e-6:
        mag = 0.0
        unit_vec = scipy.array([1.0,0.0])
    else:
        mag = math.sqrt(vv)
        unit_vec = v[t]/mag
    par = scipy.dot(unit_vec,e_v[t])
    perp = unit_vec[0]*e_v[t][1] - unit_vec[1]*e_v[t][0]
    vecs[t,0] = par
    vecs[t,1
         ] = perp
    vecs[t,2] = mag
    print >>debug, (3*'%6.2f ')%(par,perp,mag)

v_mag = scipy.matrix([[0,0,0],[0,0,100]])
a_par = scipy.matrix([[0,0,0],[100,0,0]])
a_perp = scipy.matrix([[0,0,0],[0,100,0]])

plot_data = {'data':vecs, 'A_z':v_mag,'A_x':a_par,'A_y':a_perp}
plot_color = {'data':'k.','A_z':'r-', 'A_x':'b-', 'A_y':'c-'}
#             data:black   v_mag:red   e_par:blue  e_perp:cyan

theta_matrix = scipy.matrix(scipy.diag([1.0,1,1])) # rotation matrix 
phi_matrix = scipy.matrix(scipy.diag([1.0,1,1]))   # rotation matrix 

plot_line = False
def plot_vecs():
    global theta_matrix, phi_matrix, plot_line, plot_data, plot_color

    plot_line = []
    pylab.hold(False)
    for key in plot_data.keys():
        try:
            dots = plot_data[key]*theta_matrix*phi_matrix
        except:
            print 'plot_data[%s].shape='%key,plot_data[key].shape
            raise RuntimeError
        line, = pylab.plot(dots[:,0],dots[:,1],plot_color[key])
        pylab.hold(True)
        pylab.axis([-100,100,-100,100])
        plot_line.append([plot_data[key],line])
    pylab.draw()
def update():
    # Called by new_theta() or new_phi() to redraw
    global plot_line
    if not plot_line:
        plot_vecs()
        return
    for [vec,line] in plot_line:
        dots = vec*theta_matrix*phi_matrix
        line.set_xdata(dots[:,0])
        line.set_ydata(dots[:,1])
    pylab.draw()
def new_theta(efv=None):
    global theta_matrix
    theta_angle = THETA.get()*(math.pi/180)
    theta_matrix[0,0] = math.cos(theta_angle)
    theta_matrix[0,1] = math.sin(theta_angle)
    theta_matrix[1,0] = -math.sin(theta_angle)
    theta_matrix[1,1] = math.cos(theta_angle)
    update()
def new_phi(efv=None):
    global phi_matrix
    phi_angle = PHI.get()*(math.pi/180)
    phi_matrix[1,1] = math.cos(phi_angle)
    phi_matrix[1,2] = math.sin(phi_angle)
    phi_matrix[2,1] = -math.sin(phi_angle)
    phi_matrix[2,2] = math.cos(phi_angle)
    update()

root = Tkinter.Tk()
root.title('accel_view.py')
# Row of buttons
b = Tkinter.Frame(root)
b.pack()
quit = Tkinter.Button(b,text='Quit', command=root.quit)
quit.pack(side=Tkinter.LEFT)

all = Tkinter.Button(b,text='plot', command=plot_vecs)
all.pack(side=Tkinter.LEFT)

# Row of sliders
s = Tkinter.Frame(root)
s.pack()

THETA = Tkinter.Scale(s, label='Theta',length=360,resolution=1,
                  to=(0), from_=(360), command=new_theta)
THETA.set(0)
THETA.pack(side=Tkinter.LEFT)

PHI = Tkinter.Scale(s, label='Phi',length=360,resolution=1,
                  to=(0), from_=(360), command=new_phi)
PHI.set(0)
PHI.pack(side=Tkinter.LEFT)

Tkinter.mainloop()

#Local Variables:
#mode:python
#End:
