"""

python ABQ_track.py --test|less
   points to the three lines 2029, 4222, and 5161 in
   groundTruthTracks.txt where tracks are not sequential.  Since the
   class ASS_ABQ assumes tracks are sequential, ie,
   targets are always visible and there are no false alarms, it will
   make errors at those three points.  Oh well.

python ABQ_track.py --fit |grep -v targets
   prints the reestimated Sigma_D, Sigma_O, mu_init, and Sigma_init

time python ABQ_track.py --track > temp
   Creates AMF_tracks.txt, estimated tracks
   
"""
import scipy, mvx
def augment(A,floor=1.0): # make min eigenvalue floor
    dim = A.shape[0]
    min_e = min(scipy.linalg.eigvalsh(A))
    if min_e < floor:
        A += scipy.matrix(scipy.eye(dim))*(floor-min_e)
    return
def parse_tracks(file):
    t_2_hits = {}     # Key: t, Value: [[x,y,track#], ...] 
    track_2_hits = {} # Key: track#, Value: [[x,y,t], ...]
    track = 0
    line_no = -1
    # Read and parse the tracks
    for line in file.readlines():
        line_no += 1
        if line[0:5] == 'START':
            old_t = None
            continue
        if line[0:3] == 'END':
            track += 1
            continue
        dum,t,x,y = map(lambda x:int(x),line.split())
        if not t_2_hits.has_key(t):
            t_2_hits[t] = []
        t_2_hits[t].append([x,y,track])
        if not track_2_hits.has_key(track):
            track_2_hits[track] = []
        track_2_hits[track].append([x,y,t])
        if not (old_t == None or old_t + 1 == t):
            print 'non-sequential track at line %d'%line_no
        old_t = t
    return (t_2_hits,track_2_hits)
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
    [ 10,  -0.1, 0,     0     ],
    [-.1, 30,   0,     -7    ],
    [ 0,   0,   10,     -0.04 ],
    [ 0,  -7,  -0.04,  50    ]])
Sigma_O=scipy.matrix([[1.0,   0],
                      [0,   1.0]])
Sigma_init=scipy.matrix([
    [ 1e4,  0,    0,   0],
    [ 0,    5e2,  0,   0],
    [ 0,    0,    4e4, 0],
    [ 0,    0,    0,   1e3]],scipy.float32)
mu_init=scipy.matrix([
    [250],
    [ 0],
    [250],
    [ 0]],scipy.float32).T
Lambda_new = 3.0
def reestimate(track_2_hits,    # dict of tracks read by parse_tracks()
               track_only=False # Just report tracks from MV1.decode()
               ):
    global Sigma_D, Sigma_O,mu_init,Sigma_init
    Mod = mvx.MV1(N_tar=1,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,
                  mu_init=mu_init,Sigma_init=Sigma_init,MaxD=11)
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
        d,tmp0,tmp1,tmp2 = Mod.decode(Yts) # d[k][t] is decoded x for
                                           # target k at time t
        N_i += 1
        x_i = scipy.concatenate((x_i,d[0][0]),1)
        # Initial x; concatenate along axis #1 rather than axis #0
        for t in xrange(0,len(track)-1):
            N += 1
            y0 = scipy.concatenate((y0,Yts[t][0]),1)
            x0 = scipy.concatenate((x0,d[0][t]),1)
            x1 = scipy.concatenate((x1,d[0][t+1]),1)
    # A,resids,rank,s = scipy.linalg.lstsq(x0.T,x1.T)
    # A = A.T
    e = x1-A*x0
    if track_only:
        # x0.shape = (4,3603)
        N,T = x0.shape
        rv = scipy.zeros((T,6))
        for t in xrange(T):
            rv[t,0] = x0[1,t]
            rv[t,1] = x0[3,t]
            rv[t,2] = e[0,t]
            rv[t,3] = e[1,t]
            rv[t,4] = e[2,t]
            rv[t,5] = e[3,t]
        return rv
    Sigma_D = e*e.T/N
    e = y0 - O*x0
    Sigma_O = e*e.T/N
    mu_init = x_i.sum(1)/N_i
    Sigma_init = x_i*x_i.T/N_i - mu_init*mu_init.T
    mu_init = mu_init.T

def reestimate_IMM(track_2_hits, # dict of tracks read by parse_tracks()
               track_only=False  # Just report tracks decode()
               ):
    global Sigma_D, Sigma_O,mu_init,Sigma_init
    Stop = scipy.matrix(scipy.eye(4))
    Stop[1,1] = 0
    Stop[3,3] = 0
    small = 1e-200 # Don't create second targets
    Mod = mvx.MV5(Stop=Stop, N_tar=1,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,
                  mu_init=mu_init,Sigma_init=Sigma_init,MaxD=35,Max_NA=1,
                  Lambda_new=small, Lambda_FA=small/1e5,
                  PV_V=[[1,1e-6],[1,1e-6]] )
    IMM = Mod.IMM
    dim = len(Sigma_init)
    safe = scipy.matrix(scipy.eye(dim))*1e-8
    for it in xrange(2):
        print 'iteration=%d'%it
        IMM.mus[1] = Stop*IMM.mus[1]
        IMM.Sigmas[1] = Stop*IMM.Sigmas[1]*Stop+safe
        IMM.Sigma_D[1] = safe
        augment(IMM.Sigma_O[0],floor=0.1)  # Fudge
        augment(IMM.Sigma_D[0],floor=0.1) # Fudge
        O_y = []      # Observations at t for Sigma_O estimation
        O_x = []      # Estimated states at t for Sigma_O estimation
        x0 = [[],[]]  # Estimated states at t
        x1 = [[],[]]  # Estimated states at t+1
        x_i = [[],[]] # Estimated initial states
        trans = scipy.zeros((2,2))
        for track in track_2_hits.values():
            if len(track) < 10:
                continue
            Yts = extract_yts(track)
            d,tmp0,tmp1,tmp2 = Mod.decode(Yts) # d[k][t] is decoded x,i for
                                               # target k at time t
            if len(d) != 1:
                print 'More than one track decoded.  len(d)=%d'%len(d)
                continue
            st,mt = d[0][0] # Get state(t) and mode(t)
            x_i[mt].append(st.A.reshape(-1))
            for t in xrange(len(d[0])-1):
                st0,mt0 = d[0][t]
                st1,mt1 = d[0][t+1]
                trans[mt0,mt1] += 1
                O_y.append(Yts[t][0].A.reshape(-1))
                O_x.append(st0.A.reshape(-1))
                x0[mt0].append(st0.A.reshape(-1))
                x1[mt0].append(st1.A.reshape(-1))
        # A,resids,rank,s = scipy.linalg.lstsq(x0.T,x1.T)
        # A = A.T Don't fit A.  It is known
        for temp in (x0,x1,x_i): # Convert data to scipy arrays
            temp[0] = scipy.array(temp[0])
            temp[1] = scipy.array(temp[1])
        O_y = scipy.array(O_y)
        O_x = scipy.array(O_x)
        e = O_y - scipy.dot(O_x,O.T)
        IMM.Sigma_O[0] = IMM.Sigma_O[1] = scipy.matrix(scipy.dot(e.T,e)/len(e))
        for i in xrange(2):
            #print 'i=%d'%i, 'shapes of x0[i], x1[i]=',x0[i].shape, x1[i].shape
            e = x1[i]-scipy.dot(x0[i],IMM.A[i].T)
            IMM.Sigma_D[i] = scipy.matrix(scipy.dot(e.T,e)/len(e))
            IMM.mus[i] = scipy.matrix(x_i[i].sum(0)/len(x_i[i])).T
            N_i,temp = x_i[i].shape
            IMM.Sigmas[i] = scipy.matrix(scipy.dot(x_i[i].T,x_i[i])/N_i -\
                                             IMM.mus[i]*IMM.mus[i].T)
        D = trans.sum(1)
        IMM.Pij = (trans.T/D).T
        for i in xrange(2):
            print '\nIMM component %d:'%i
            for pair in ((IMM.Sigma_D,'Sigma_D'),(IMM.Sigma_O,'Sigma_O'),
                         (IMM.mus,'mus'),(IMM.Sigmas,'Sigmas')):
                if pair[1][0] is 'S':
                    evals = scipy.linalg.eigvalsh(pair[0][i])
                    print '  %s evals='%pair[1],evals,'\n',pair[0][i]
                else:
                    print '  %s=\n'%pair[1],pair[0][i]
        print 'Pij=\n',IMM.Pij
    pickle.dump(Mod,open('IMM_Mod','w'))
    return

if __name__ == '__main__':  # Test code
    import sys, getopt, pickle
    opts,pargs = getopt.getopt(sys.argv[1:],'',
                               ['count',
                                'fit',
                                'track',
                                'fitIMM',
                                'trackIMM',
                                'test',
                                'accel',
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
        Mod = mvx.MV_ABQ(N_tar=1,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,
                    MaxD=5.0,A_floor=6.0,Max_NA=10,mu_init=mu_init,
                    Sigma_init=Sigma_init,Lambda_new=Lambda_new,Murty_Ex=10000)
        # Recall: t_2_hits[t].append([x,y,track])
        yts = []
        for t1 in t_2_hits.keys():
            t = t1-1 # FixMe
            yts.append([])
            for triple in t_2_hits[t1]:
                yts[t].append(scipy.matrix(triple[0:2]).T)
        d,y_A,nu,close_calls = Mod.decode(yts)
        print 'nu=',nu
        file = open('AMF_tracks.txt','w')
        print >>file,'START_FILE'
        for track in d:
            print >>file,'START_TRACK'
            t=0
            for hit in track:
                t += 1
                if not hit is None:
                    print >>file,'0 %2d %3d %3d 0 0'%(t,int(hit[0][0]),
                                                      int(hit[0][2]))
                        # confidence, frame#, x, y, x_dot, y_dot
            print >>file,'END_TRACK'
        print >>file,'END_FILE'
        file.close
    if opt_dict.has_key('--fit') or opt_dict.has_key('--accel'):
        T = len(t_2_hits)
        N_tracks = len(track_2_hits)
        Lambda_new = float(N_tracks)/float(T)
        Mods = [mvx.MV_ABQ(N_tar=1,A=A,Sigma_D=Sigma_D,O=O,Sigma_O=Sigma_O,
              mu_init=mu_init,Sigma_init=Sigma_init,Lambda_new=Lambda_new)]
        if opt_dict.has_key('--accel'): # Print estimated states/accelerations
            file = open('AMF_accel.txt','w')
            v_a = reestimate(track_2_hits,track_only=True)
            T,N = v_a.shape
            for t in xrange(T):
                print >>file, (6*'%6.2f ')%(v_a[t,0],v_a[t,1],v_a[t,2],
                           v_a[t,3],v_a[t,4],v_a[t,5])
        if opt_dict.has_key('--fit'):   # Fit model parameters
            for I in xrange(2):
                print 'I=%d, mu_init=\n'%I,mu_init
                reestimate(track_2_hits)
                Mods.append(mvx.MV_ABQ(N_tar=1,A=A,Sigma_D=Sigma_D,O=O,
                    Sigma_O=Sigma_O,mu_init=mu_init,Sigma_init=Sigma_init,
                    Lambda_new=Lambda_new))
            for I in xrange(len(Mods)):
                print '\n\nDumping model %d'%I
                Mods[I].dump()
    if opt_dict.has_key('--fitIMM'):   # Fit IMM model parameters
        v_a = reestimate_IMM(track_2_hits,track_only=True)
    
    if opt_dict.has_key('--trackIMM'):   # Track with IMM
        # Read the fit model.  Fix several parameters
        Mod = pickle.load(open('IMM_Mod','r'))
        Mod.Lambda_new = 3.0
        Mod.log_min_pd = -25.0
        Mod.Max_NA = 5
        Mod.Murty_Ex = 1000
        Mod.A_floor = 6.0
        IMM = Mod.IMM
        dim = len(Sigma_init)
        safe = scipy.matrix(scipy.eye(dim))*1e-8
        Stop = scipy.matrix(scipy.eye(4))
        Stop[1,1] = 0
        Stop[3,3] = 0
        IMM.mus[1] = Stop*IMM.mus[1]
        IMM.Sigmas[1] = Stop*IMM.Sigmas[1]*Stop+safe
        IMM.Sigma_D[1] = safe
        augment(IMM.Sigma_O[0],floor=0.1) # Fudge
        augment(IMM.Sigma_D[0],floor=0.1) # Fudge
        yts = []
        for t1 in t_2_hits.keys():
            t = t1-1 # FixMe
            yts.append([])
            for triple in t_2_hits[t1]:
                yts[t].append(scipy.matrix(triple[0:2]).T)
        d,y_A,nu,close_calls = Mod.decode(yts)
        print 'nu=',nu
        file = open('AMF_IMM_tracks.txt','w')
        print >>file,'START_FILE'
        for track in d:
            print >>file,'START_TRACK'
            t=0
            for hit in track:
                t += 1
                if not hit is None:
                    print >>file,'0 %2d %3d %3d 0 0'%(t,int(hit[0][0]),
                                                      int(hit[0][2]))
                        # confidence, frame#, x, y, x_dot, y_dot
            print >>file,'END_TRACK'
        print >>file,'END_FILE'
        file.close
            
    if opt_dict.has_key('--test'):
        # Check for missed hits in each track
        for track in track_2_hits.values():
            t = track[0][2]
            for hit in track[1:]:
                if hit[2] != t+1:
                    print 'gap in track at t=%d,'%(t+1,),(3*' %d')%tuple(old_hit)
                t += 1
                old_hit = hit
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
