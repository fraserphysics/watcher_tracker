"""
Mega.py Dumb script to make surface plot

"""
import numpy, os, math, time

pairs = [['tempS3','SM 3 hyp'],['tempM1','MM 1 hyp'],['tempM3','MM 3 hyp']]
# Begin ugly gnuplot stuff
splot_preface = """
set hidden
set style data lines
set xlabel "-log(Prob MD)"
set ylabel "FA rate"
set zlabel "error"
set terminal png
set output 'Mega.png'
"""
gp_pipe = os.popen ("gnuplot",'w')
gp_pipe.write(splot_preface)
gp_pipe.write('set title "Title"\n')
gp_pipe.write('splot ')
for name,title in pairs[:-1]:
    gp_pipe.write("'%s' using 1:2:($3/2.7) title '%s',"%(name, title))
name,title = pairs[-1]
gp_pipe.write("'%s' using 1:2:($3/2.7) title '%s'\n"%(name,title))
gp_pipe.flush()
raw_input('done?')
#---------------
# Local Variables:
# eval: (python-mode)
# End:
