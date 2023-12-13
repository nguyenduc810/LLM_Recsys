'''
good keyword: 
- Food name 
- Food gredient 
'''

import json
import numpy as np 

def compute_idf(count, num_doc):
    return np.log(num_doc/count)


def print_dict(idict, value, num=100):
    c = 0
    tmp = {} 
    for k, v in idict.items():
        if v == value:
            tmp[k] = v
            c += 1
            if c >= num:
                print(tmp)
                break


def keep_np(idict, np_to_test=None, min_value=None, max_value=None):
    removed = []
    satisfied = []
    if np_to_test is None: 
        np_to_test = idict.keys()
    for k in np_to_test:
        v = idict[k]
        sat = False
        if max_value is None or v <= max_value:
            if min_value is None or v >= min_value:
                satisfied.append(k)
                sat = True
        if not sat: 
            removed.append(k)
    return removed, satisfied
        


dt = json.load(open('data/ke_spacy/phoenix.json'))
dt.keys()

np2count = dt['np2count']
np2rev = dt['np2review_count']
np2res = dt['np2res_count']

c = sorted(np2count.items(), key=lambda x:x[1], reverse=True)
rv = sorted(np2rev.items(), key=lambda x:x[1], reverse=True)
rs = sorted(np2res.items(), key=lambda x:x[1], reverse=True)

c[:100]
rv[:100]
rs[:100]

# compute idf 
num = 900 
idf = {} 

for k, v in np2res.items():
    idf[k] = compute_idf(v, num)

idf_sorted = sorted(idf.items(), key=lambda x:x[1], reverse=True)
idf_sorted[:100]
idf_sorted[-100:]

rv[-100:]

print_dict(np2rev, 10)

# --------------------------------------------------------------------------------
# FILTERING 
# 1. Filter out keywords appearing in less than N reviews (N = 10)
removed_1, satisfied_1 = keep_np(np2rev, min_value=10)
len(removed_1)
len(satisfied_1)
# 2. Filter out keywords appearing in M% of the restaurants (M=10% --> 810 rest)
removed_2, sats_2 = keep_np(np2res, np_to_test=satisfied_1, max_value=630)
removed_2