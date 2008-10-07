"""
Plots selected slices of "survey/dict" a file that contains a pickled
dictionary of the tracking scores.  There is a score for each "T*"
file in "survey/"
"""

import os,pickle

flat_dict = pickle.load(open('survey/dict','r'))
md_fa_mod_na = {}
MD_dict = {}
FA_dict = {}
Mod_dict = {}
NA_dict = {}
for key in flat_dict.keys():
    MD = float(key[5:6])
    start = 10
    stop = key.find('.Mod')
    FA = float(key[start:stop])
    start = stop + 5
    stop = start+1
    Mod = key[start:stop]
    start = stop+4
    stop = start + 1
    NA = int(key[start:stop])

    MD_dict[MD] = True
    FA_dict[FA] = True
    Mod_dict[Mod] = True
    NA_dict[NA] = True
    if not md_fa_mod_na.has_key(MD):
        md_fa_mod_na[MD] = {}
    if not md_fa_mod_na[MD].has_key(FA):
        md_fa_mod_na[MD][FA] = {}
    if not md_fa_mod_na[MD][FA].has_key(Mod):
        md_fa_mod_na[MD][FA][Mod] = {}
    md_fa_mod_na[MD][FA][Mod][NA] = flat_dict[key]
Mods = ['S','M']
MDs = MD_dict.keys()
FAs = FA_dict.keys()
NAs = NA_dict.keys()
for L in [MDs,FAs,NAs]:
    L.sort()
for Mod in Mods:
    for NA in NAs:
        file = open('temp'+Mod+str(NA),'w')
        for FA in FAs:
            for MD in MDs:
                print >>file, '%3.1f %3.1f %7.3f'%(
                    MD,FA,md_fa_mod_na[MD][FA][Mod][NA])
            print >>file, ''

#---------------
# Local Variables:
# eval: (python-mode)
# End:
