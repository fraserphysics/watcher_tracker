import sys
hits = {}
#file = open(sys.argv[1],'r')
file = open('data/ABQ_Intersection/groundTruthTracks.txt','r')
for line in file.readlines():
    if line[0:5] == 'START' or  line[0:3] == 'END':
        continue
    t,x,y = map(lambda x:int(x),line.split())
    if not hits.has_key(t):
        hits[t] = []
    hits[t].append([x,y])
ts = hits.keys()
ts.sort()
counts = []
for key in ts:
    counts.append(len(hits[key]))
counts.sort()
print counts
