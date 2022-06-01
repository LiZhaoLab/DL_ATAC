from collections import defaultdict, deque
from overlap_ops import unique
import multiprocessing
import peakbucket as pb

# functions to generate the relationship graph:
# a defaultdict of a set of peak names B-N,
# keyed on a peak name A,
# such that peaks B-N overlap peak A
# (e.g. "pk_1" : {"pk_1", "pk_2","pk_3"})
# note that the algorithm in `adj_search` does not halt

CLUSTER_AGG_DISTANCE = 100
# the distance in bp at which peaks get grouped together
# as the same cluster
# used in both the call to `unique`

CLUSTER_COMPARE_FUNCTION = 'fast'
# controls which functions gets used when
# building the relationship graph
# choose from 'fast' or 'known'

from copy import copy
def fast_classify(A, B, distance=100):
    # this is much more efficient (~3X) than silent_classify
    # but the algo is different enough that it may not be correct?
    # better used when debugging but maybe we should cross-check it
    # when making the Final Set of Results
    '''given a single Peak A and a list of peaks B,
    finds overlaps
    returns: olaps_A, olaps_B, uniqs_B
    '''
    # we make an unsafe assumption that A and B are already unique-ified and sorted
    A = deque([A])
    B = deque(B)
    l_A = len(A)
    l_B = len(B)
    for a in A:
        a.source = 'A'
        a.olap = False
    for b in B:
        b.source = 'B'
        b.olap = False

    seqs = B # if we don't need A, then we don't need to resort and re-dequeify
    
    olaps = deque()
    uniqs = deque()

    for c in A:
        while seqs:
            if pb.overlap2(c, seqs[0], distance=distance):
                seqs[0].olap = True # dirty the "next" peak -- otherwise last peaks of a run will get lost
                i = seqs.popleft()
                olaps.append(i) # moved to account for logic adjustment
            elif c < seqs[0]:
                # fast halt
                comparison_peak = copy(c)
                comparison_peak.start = max(0, c.start - distance)
                comparison_peak.end = c.end + distance
                
                if comparison_peak < seqs[0]:
                    break 
            else:
                i = seqs.popleft()
                uniqs.append(i) # ditto

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
    olaps_A = [i for i in olaps if i.source == 'A']
    olaps_B = [i for i in olaps if i.source == 'B']

    return olaps_A, olaps_B

def silent_classify(A, B, distance=100):
    # this differs from overlap_ops.classify
    # twice: it doesn't yield any debug info
    # and A gets turned into a list first
    # (since a single Peak will get decomposed into a set of letters
    # otherwise)
    '''given a single Peak A and a list of peaks B,
    partitions each list into unique and overlapping
    (consider using my_pb.all_peaks() as input)
    returns: olaps_A, uniqs_A, olaps_B, uniqs_B
    '''
    A = deque(unique(sorted([A]), verbose=False))
    B = deque(unique(sorted(B), verbose=False))
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

    assert len(uniqs_A) + len(olaps_A) == l_A, l_A
    assert len(uniqs_B) + len(olaps_B) == l_B, l_B
    
    return olaps_A, uniqs_A, olaps_B, uniqs_B

def _lambda_(x, which_func=CLUSTER_COMPARE_FUNCTION):
    # this looks crazy, but the "uniqueness"
    # check at the beginning is expensive (120ms)
    # and doesn't need to be done by each worker,
    # so it makes more sense to do it once
    # and cache it...
    '''which_func is either the fast or the known-good one'''
    which_func = CLUSTER_COMPARE_FUNCTION
    assert which_func in ['fast', 'known']

    y = _cached_input_pgb
    # note that this comes out of the `global` call from
    # `network_pks_mt`

    if which_func == 'fast':
        return fast_classify(x, y, distance=CLUSTER_AGG_DISTANCE)
    elif which_func == 'known':
        out = silent_classify(x, y, distance=CLUSTER_AGG_DISTANCE)
        # needed since silent classify returns extraneous results
        # that don't match fast_classify
        return out[0], out[2]

def network_pks_mt(pgb1, pgb2, context):
    '''compute overlap network from pgb1 to pgb2 in context
    this differs from pangen.pgb_compare since this is much more expensive
    but breaks out what peak overlaps what peak.
    This is heavily multithreaded for speed.
    
    set CLUSTER_AGG_DISTANCE to adjust distance.
    set CLUSTER_COMPARE_FUNCTION to change which overlap function gets called'''

    global _cached_input_pgb
    # we need this so it can get to _lambda_
    # we could pass it through, but then the mp call gets super ugly
    # since starmap (which is the only option preserving strict ordering)
    # unpacks args annoyingly

    _cached_input_pgb = unique(sorted(pgb2[context].all_peaks()),
                               distance=CLUSTER_AGG_DISTANCE,
                              verbose=False)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-2) as pool:
        res = pool.map(_lambda_, pgb1[context].all_peaks())

    res = build_rel_graph(res, pgb1[context].all_peaks())

    return res

def build_rel_graph(results, peaks):
    '''ingests the list of results coming off of mp.pool.map
    (field 0 -- variable)
    (field 1 -- list of Peaks that overlap)'''

    rels = defaultdict(set)

    assert len(results) == len(peaks)

    for result, query in zip(results, peaks):
        hits = result[1]
        rels[query.name] = set([query.name])
        # I think my graph-tracing algo may need this; 
        # anyways, it seems cheap enough to explicitly
        # self-connect
        
        if hits: # aka there are any results
            for hit in hits:
                rels[query.name] |= set([hit.name])
    return rels

def network(pgb1, pgb2):
    '''using pgb1 as a reference, calculates what peaks in pgb2 overlap what peaks in pgb1
    across all contexts available in pgb1.
    We assume an overlap in any context is sufficient.
    
    set CLUSTER_AGG_DISTANCE to adjust distance.
    set CLUSTER_COMPARE_FUNCTION to change which overlap function gets called.'''
    results = {}
    contexts = pgb1.species()
    for context in contexts:
        results[context] = network_pks_mt(pgb1, pgb2, context)

    # combine the results back together
    combined = {}
    for k in pgb1.list_peaks():
        hits = [results[c][k] for c in contexts]
        combined[k] = set().union(*hits) # we take any overlap in any context
    return combined

def combine_rel_graphs(gr1, gr2):
    '''given 2 graphs (dict of sets), combines them as a union of keys and values'''

    combined = defaultdict(set)
    for k in (set(gr1.keys()) | set(gr2.keys())):
        # since this is general, not guaranteed all keys will be in all graphs
        if k in gr1.keys():
            combined[k] |= gr1[k]
        if k in gr2.keys():
            combined[k] |= gr2[k]

    return combined

def generate_full_network(pgbs):
    '''given a list of pgbs, does all pairwise comparisons
    to return a relationship graph of what overlaps with what.

    We assume an overlap in any context is sufficient.

    set CLUSTER_AGG_DISTANCE to adjust distance.
    set CLUSTER_COMPARE_FUNCTION to change which overlap function gets called.'''
    from itertools import combinations

    pairs = combinations(pgbs, 2)
    combined = {}

    for pair in pairs:
        n = network(pair[0], pair[1])
        combined = combine_rel_graphs(combined, n)
        n = network(pair[1], pair[0])
        combined = combine_rel_graphs(combined, n)

    return combined

# TODO: consider making this object-oriented instead of assuming we pass around the same structure?
# anyways, the basic relationship graph structure is
# a defaultdict of a set of peak names B-N,
# keyed on a peak name A,
# such that peaks B-N overlap peak A
# (e.g. "pk_1" : {"pk_1", "pk_2","pk_3"})
# note that the algorithm in `adj_search` does not halt
# if the graph is not fully bidirectional

def adj_search(n, root, rels, state_dict, sets_dict, debug=False):
    '''search through relationship graph recursively;
    Called by cluster_peaks.
    n is the current node for this iteration;
    root is the `root` node (at the top of the recursion);
    state_ and sets_dict hold the working tables of
    what cluster each peak is assigned to (state)
    and what peaks are in each cluster (sets)
    
    pass `debug=True` for verbose diagnostic output'''

    if debug: print(f'at {n}, from root {root}')
    visited = state_dict[n]
    if not visited:
        # check the rels
        state_dict[n] = root # assign this peak to the root
        sets_dict[root].update({n})

        if debug: print(f'{n} not assigned; marking as {root}')

        connected = rels[n]
        for c in connected:
            if debug: print('recursing...')
            adj_search(c, root, rels, state_dict, sets_dict, debug=debug)

    elif visited != root:
        # we are revisiting from another root node
        # so the root node is NOT actually a new root
        if debug:
            print(f"{visited} is not {root}")
            print(f"rewriting {state_dict[root]} as {visited}")

        state_dict[root] = state_dict[n]
        sets_dict[state_dict[n]].update({root})
        # make sure the current root (which may not be the real root)
        # points the same place as the already-found root

        connected = rels[n]
        for c in connected:
            if debug: print('recursing ---')
            #adj_search(c, state_dict[n], rels, state_dict, sets_dict, debug=debug)
            adj_search(c, state_dict[n], rels, state_dict, sets_dict, debug=debug)
    elif visited == root:
        if debug: print(f"{visited} is {root}; no-op")
    else:
        raise ValueError("should not happen")

    if debug: print('returning...')
    return

def cluster_peaks(rels, debug=False):
    '''given a graph of what peaks overlap,
    return a dict of peak name: parent peak (arbitrary, deterministic)
    and a dict of parent peak: set(all member peaks)
    
    The graph takes the following form:
    a defaultdict of a set of peak names B-N,
    keyed on a peak name A,
    such that peaks B-N overlap peak A
    (e.g. "pk_1" : {"pk_1", "pk_2","pk_3"})'''
    names = sorted(set(rels.keys()))

    # verbally warn if rels isn't bidirectional
    validate_graph(rels, verbose=False, harmonize=False)

    state_dict = {n:False for n in names}
    sets_dict = defaultdict(set)
    for n in names:
        if debug: print('\n\ninitiating', n)
        adj_search(n, n, rels, state_dict, sets_dict, debug=debug)
    keys_to_drop = set(sets_dict.keys()) - set(state_dict.values())
    for k in keys_to_drop: del sets_dict[k]

    # every peak should end up in sets_dict exactly once, so length should be conserved
    assert sum([len(l) for l in sets_dict.values()]) == len(names)

    return state_dict, sets_dict

def validate_graph(gr, verbose=True, harmonize=False):
    '''evaluate if the graph is fully bidirectional;
    the adj_search algo will fail if it is not.
    Pass harmonize=True to correct issues IN-PLACE'''
    count = 0
    for k in gr.keys():
        for v in gr[k]:
            if k not in gr[v]:
                if verbose: print(f"asymmetry: {k} -> {v}")
                count += 1
                if harmonize: gr[v] |= set([k])
    if count != 0: print(f"{count} asymmetries detected")
    if harmonize: print('corrected in place')
    return gr

# some internal debugging functions
def generate_random_interval(n, width=20, maxcoord=500):
    demo_A = sorted([random.randint(0,maxcoord - width) for i in range(n)])
    demo_A = [(r, r+width) for r in demo_A]
    return demo_A

def debug_build_rel_graph(cases):
    '''test_cases = [generate_random_interval(50, 200, 50000) for a in range(3)]
    then convert test_cases into Peaks, then call this to build a simple rel_graph'''
    rels = defaultdict(set)
    allpeaks = cases[0] + cases[1] + cases[2]

    for i, p1 in enumerate(allpeaks):
        for j, p2 in enumerate(allpeaks):
            if p1.overlap2(p2,100): rels[p1.name] |= {p2.name}
    return rels

import matplotlib.pyplot as plt
pyplot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from itertools import cycle

def assign_colors(sets):
    '''assign colors to sets with more than 1 member (non-singletons)'''
    singletons = [k for k in sets.keys() if len(sets[k]) == 1]
    clusters = [k for k in sets.keys() if k not in singletons]
    
    color_dict = {k: c for k, c in zip(clusters, cycle(pyplot_colors))}
    for k in singletons:
        color_dict[k] = 'black'
    return color_dict