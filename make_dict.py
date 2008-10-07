"""

Creates or updates "survey/dict" a file that contains a pickled
dictionary of the tracking scores.  There is a score for each "T*"
file in "survey/"

Be on a 64bit machine: vooka, wooka or mitch

scp -r andy@watcher:projects/ps/trunk/survey .

export PATH=/projects/watcher/linux64/bin:$PATH
python2.4 make_dict

"""

import os,pickle

try:
    Dict = pickle.load(open('survey/dict','r'))
except:
    Dict = {}
for name in os.listdir('survey'):
    if name[0] != 'T' or Dict.has_key(name):
        continue
    P = os.popen(
        'evaluate track -t /projects/watcher/tracking/abqGroundTruth.trx -i ' +
        'survey/' + name + ' -fs 1 -fe 100 -v 1')
    Dict[name] = float(P.readlines()[0].split()[1])
    print '%-26s %6.2f'%(name, Dict[name])
pickle.dump(Dict,open('survey/dict','w'))

#---------------
# Local Variables:
# eval: (python-mode)
# End:
