import mv1, time,random
random.seed(3)
ts = time.time()
M = mv1.MV1(N_obj=4)
y,s = M.simulate(5)
d = M.decode(y)
print 'len(y)=',len(y), 'len(s)=',len(s),'len(d)=',len(d)
for t in xrange(len(y)):
    print 't=%d    y         s           d'%t
    for k in xrange(len(y[t])):
        print ' k=%d  %4.2f  '%(k,y[t][k][0,0]),
        for f in (s[t][k],d[t][k]):
            print '(%4.2f, %4.2f)  '%(f[0,0],f[1,0]),
        print ' '
print 'elapsed time=',time.time()-ts
