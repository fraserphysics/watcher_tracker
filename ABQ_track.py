"""
"""
import scipy
def parse_tracks(file):
    t_2_hits = {}     
    track_2_hits = {}
    track = 0
    # Read and parse the tracks
    for line in file.readlines():
        if line[0:5] == 'START':
            continue
        if line[0:3] == 'END':
            track += 1
            continue
        t,x,y = map(lambda x:int(x),line.split())
        if not t_2_hits.has_key(t):
            t_2_hits[t] = []
        t_2_hits[t].append([x,y,track])
        if not track_2_hits.has_key(track):
            track_2_hits[track] = []
        track_2_hits[track].append([x,y,t])
    return (t_2_hits,track_2_hits)
import mvx
class ASS_ABQ(mvx.ASSOCIATION5):
    def __init__(self,*args,**kwargs):
        mvx.ASSOCIATION5.__init__(self,*args,**kwargs)
        self.type='ASS_ABQ'
        self.cause_checks = [self.check_targets, #No self.check_FAs for ABQ
                             self.check_newts]
    def New(self, *args,**kwargs):
        NA = ASS_ABQ(*args,**kwargs)
        NA.dead_targets = self.dead_targets.copy()
        return NA
class MV_ABQ(mvx.MV5):
    """ Like MV4 but each global association is composed by selecting
    one association from each cluster in a cluster flock.
    """
    def __init__(self,**kwargs):
        mvx.MV5.__init__(self,**kwargs)
        self.ASSOCIATION = ASS_ABQ
        self.PV_V = scipy.matrix([[.9,.1],[.1,.9]])
        self.Lambda_FA=0.001 # No FAs in ABQ
        self.O = scipy.matrix([[1,0,0,0],[0,0,1,0]])
    def dump(self  # MV_ABQ
             ):
        print 'MV_ABQ model dump: N_tar=%d, alpha=%5.3f A=\n'%(
            self.N_tar,self.alpha),self.A
        print 'Sigma_D=\n',self.Sigma_D
        print 'O=\n',self.O
        print 'Sigma_O=\n',self.Sigma_O
        print 'mu_init=\n',self.mu_init
        print 'Sigma_init=\n',self.Sigma_init
        print 'Sigma_FA=\n',self.Sigma_FA
        print 'PV_V=\n',self.PV_V
        print 'Lambda_new=%5.3f,  Lambda_FA=%5.3f'%(self.Lambda_new,
                                                    self.Lambda_FA)
        return
def extract_yts(track):
    yts = []
    for triple in track:
        yts.append([scipy.matrix(triple[0:2]).T])
    return yts

# Initial guesses for model parameters
O = scipy.matrix([[1,0,0,0],[0,0,1,0]])
A = scipy.matrix([[1,1,0,0],
                  [0,1,0,0],
                  [0,0,1,1],
                  [0,0,0,1]])
Sigma_D=scipy.matrix([
    [ 0,  -0.1, 0,     0     ],
    [-.1, 30,   0,     -7    ],
    [ 0,   0,   0,     -0.04 ],
    [ 0,  -7,  -0.04,  50    ]])
Sigma_O=scipy.matrix([[1.0,   0],
                      [0,   1.0]])
Sigma_init=scipy.matrix([
    [ 1e4,  0,    0,   0],
    [ 0,    5e2,  0,   0],
    [ 0,    0,    4e4, 0],
    [ 0,    0,    0,   1e3]])
mu_init=scipy.matrix([
    [250],
    [ 0],
    [250],
    [ 0]]).T
Lambda_new = 3.0
def reestimate(track_2_hits):
    global Sigma_D, Sigma_O,mu_init,Sigma_init
    Mod = mvx.MV1(N_tar=1,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,
                  mu_init=mu_init,Sigma_init=Sigma_init)
    # Now reestimate using decoded states as truth
    N=0
    N_i=0
    y0 = scipy.matrix([[],[]])
    x0 = scipy.matrix([[],[],[],[]])
    x1 = scipy.matrix([[],[],[],[]])
    x_i = scipy.matrix([[],[],[],[]])
    for track in track_2_hits.values():
        if len(track) < 10:
            continue
        Yts = extract_yts(track)
        d,tmp = Mod.decode(Yts)
        N_i += 1
        x_i = scipy.concatenate((x_i,d[0][0]),1)
        for t in xrange(0,len(track)-1):
            N += 1
            y0 = scipy.concatenate((y0,Yts[t][0]),1)
            x0 = scipy.concatenate((x0,d[0][t]),1)
            x1 = scipy.concatenate((x1,d[0][t+1]),1)
    # A,resids,rank,s = scipy.linalg.lstsq(x0.T,x1.T)
    # A = A.T
    e = x1-A*x0
    Sigma_D = e*e.T/N
    e = y0 - O*x0
    Sigma_O = e*e.T/N
    mu_init = x_i.sum(1)/N_i
    Sigma_init = x_i*x_i.T/N_i - mu_init*mu_init.T
    mu_init = mu_init.T
            
if __name__ == '__main__':  # Test code
    import sys, getopt
    opts,pargs = getopt.getopt(sys.argv[1:],'',
                               ['count',
                                'fit',
                                'track',
                                'test',
                                'In_File='
                                ])
    opt_dict ={}
    for opt in opts:
        if len(opt) is 2:
            opt_dict[opt[0]] = opt[1]
        else:
            opt_dict[opt[0]] = True  
    if opt_dict.has_key('In_File'):
        file = open(opt_dict['In_File'],'r')
    else:
        file = open('groundTruthTracks.txt','r')
    t_2_hits,track_2_hits = parse_tracks(file)
    
    if opt_dict.has_key('--track'):
        for I in xrange(2):
            print 'I=%d, mu_init=\n'%I,mu_init
            reestimate(track_2_hits)
        # Recall: t_2_hits[t].append([x,y,track])
        yts = []
        for t1 in t_2_hits.keys():
            t = t1-1 # FixMe
            yts.append([])
            for triple in t_2_hits[t1]:
                yts[t].append(scipy.matrix(triple[0:2]).T)
        Mod = MV_ABQ(N_tar=1,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,MaxD=3.2,
               mu_init=mu_init,Sigma_init=Sigma_init,Lambda_new=Lambda_new)
        d,tmp = Mod.decode(yts)
        print d
    if opt_dict.has_key('--fit'): # Fit model parameters
        T = len(t_2_hits)
        N_tracks = len(track_2_hits)
        Lambda_new = float(N_tracks)/float(T)
        Mods = [MV_ABQ(N_tar=1,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,
              mu_init=mu_init,Sigma_init=Sigma_init,Lambda_new=Lambda_new)]
        for I in xrange(2):
            print 'I=%d, mu_init=\n'%I,mu_init
            reestimate(track_2_hits)
            Mods.append(MV_ABQ(N_tar=1,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,
               mu_init=mu_init,Sigma_init=Sigma_init,Lambda_new=Lambda_new))
        for I in xrange(len(Mods)):
            print '\n\nDumping model %d'%I
            Mods[I].dump()
            
    if opt_dict.has_key('--test'):
        # For each frame, print a line for each hit
        for key,value in t_2_hits.items():
            print 't=%d'%key
            for item in value:
                print '  %3d %3d %3d'%tuple(item)
        # Print sorted list of number of hits per frame
        ts = t_2_hits.keys()
        ts.sort()
        counts = []
        for key in ts:
            counts.append(len(t_2_hits[key]))
        counts.sort()
        print counts
    if opt_dict.has_key('--count'): # Count the lengths of the tracks
        lengths = {}
        for track in track_2_hits.values():
            L = len(track)
            if lengths.has_key(L):
                lengths[L] += 1
            else:
                lengths[L] = 1
        Ls = lengths.keys()
        Ls.sort()
        for L in Ls:
            print 'There are %3d tracks of length %d'%(lengths[L],L)
  
#---------------
# Local Variables:
# eval: (python-mode)
# End:
