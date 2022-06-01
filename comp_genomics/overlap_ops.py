import peakbucket as pb

import ezHalLiftover as ez
from collections import deque

import pandas as pd

def unique(seqs, distance=100, verbose=True):
    '''given a list of intervals,
    deduplicates them by condensing at a given distance'''
    c = 0
    new = deque()
    seqs = deque(seqs)
    
    i = seqs.popleft()
    
    while seqs:
        if pb.overlap2(i, seqs[0], distance=distance):
            i = pb.condense(i, seqs[0], assert_pkname=False)
            seqs.popleft()
            c+=1
        else:
            new.append(i)
            i = seqs.popleft()

    new.append(i)

    if verbose: print(c)
    return new

def classify(A, B, distance=100):
    '''given two lists of peaks A and B,
    partitions each list into unique and overlapping
    (consider using my_pb.all_peaks() as input)
    returns: olaps_A, uniqs_A, olaps_B, uniqs_B
    '''
    A = deque(unique(sorted(A)))
    B = deque(unique(sorted(B)))
    l_A = len(A)
    l_B = len(B)
    for a in A:
        a.source = 'A'
        a.olap = False
    for b in B:
        b.source = 'B'
        b.olap = False

    S = A + B
    seqs = deque(sorted(S))
    
    olaps = deque()
    uniqs = deque()

    i = seqs.popleft()
    c = i # keep a separate copy for aggregation comparisons
    
    while seqs:
        if pb.overlap2(c, seqs[0], distance=distance):
            seqs[0].olap = True # dirty the "next" peak -- otherwise last peaks of a run will get lost
            olaps.append(i)
            i = seqs.popleft()
            c = pb.condense(c, i, assert_pkname=False)
        else:
            uniqs.append(i)
            i = seqs.popleft()
            c = i
    
    # end behavior
    if c == i: # if we haven't hit an overlap
        uniqs.append(i)
    else:
        olaps.append(i)

    # filter out last peaks
    cln = deque()
    while uniqs:
        p = uniqs.popleft()
        if p.olap:
            olaps.append(p)
        else:
            cln.append(p)
    uniqs = cln
    olaps = sorted(olaps)
    
    # disentangle them
    uniqs_A = [i for i in uniqs if i.source == 'A']
    uniqs_B = [i for i in uniqs if i.source == 'B']
    olaps_A = [i for i in olaps if i.source == 'A']
    olaps_B = [i for i in olaps if i.source == 'B']
    
    print(len(olaps_A), len(uniqs_A), len(olaps_B), len(uniqs_B))
    
    assert len(uniqs_A) + len(olaps_A) == l_A, l_A
    assert len(uniqs_B) + len(olaps_B) == l_B, l_B
    
    return olaps_A, uniqs_A, olaps_B, uniqs_B

def name_intersect(A, B):
    '''use the name fields on pbs A and B to intersect'''

    return set([str(s.name) for s in A.all_peaks()]) & set([str(s.name) for s in B.all_peaks()])

def name_difference(A, B):
    '''use the name fields on pbs A and B to return the (symmetric) difference (A, B, not both)'''

    return set([str(s.name) for s in A.all_peaks()]) ^ set([str(s.name) for s in B.all_peaks()])

def name_subtract(A, B):
    '''use the name fields on pbs A and B to return the difference (A not in B)'''

    return set([str(s.name) for s in A.all_peaks()]) - set([str(s.name) for s in B.all_peaks()])
