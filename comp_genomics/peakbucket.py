# PeakBucket and Peak classes for comparison / condensation

from collections import deque, defaultdict
#import itertools
from copy import copy
import pandas as pd
from functools import total_ordering

@total_ordering
class Peak():
    def __init__(self, contig, start, end, name):
        self.contig = contig
        self.start = min(int(start), int(end)) # enforces starts always <= end
        self.end =   max(int(start), int(end))
        self.name = name
        
    def __repr__(self):
        return f"{self.contig}: {self.start}-{self.end} ({self.name})"
    
    def __eq__(self, other):
        return self.__repr__() == other.__repr__()
    
    def __lt__(self, other):
        if self.contig < other.contig:
            return True
        elif self.contig == other.contig and self.start < other.start:
            return True
        elif self.contig == other.contig and self.start == other.start and self.end < other.end:
            return True
        else:
            return False

    def overlap(self, other):
        assert isinstance(other, self.__class__)
        
        if self.contig != other.contig:
            return False
        elif self.start <= other.start <= self.end:
            return True
        elif self.start <= other.end   <= self.end:
            return True
        elif other.start <=self.start  <= other.end:
            return True
        elif other.start <=self.end    <= other.end:
            return True
        else:
            return False

class PeakBucket():

    def __init__(self, data=None):
        #self.peaks = defaultdict(list)
        self.peaks = defaultdict(deque)

        if data:
            try:
                for p in iter(data):
                    self.add_peak(p)
            except:
                raise ValueError(f"failed to construct a PeakBucket from {data}")
    
    def add_peak(self, newpeak):
        
        key = newpeak.name # takes advantage of new peak's sequence field
        
        if newpeak not in self.peaks[key]:
            self.peaks[key].append(newpeak) # adds newpeak to list of peaks under that sequence 
    
    def nfrags(self, name):
        '''
        counts number of peak fragments for a given peakname
        '''
        return len(self.peaks[name])
    
    #def merge(self, newpeakbucket, debug=False):
    #    '''
    #    merges the newpeakbucket into the existing one
    #    '''
    #    if debug:
    #        old = len(self.peaks)
    #        old_total = len(self.all_peaks())
    #    
    #    for newpeak in newpeakbucket.all_peaks():
    #        self.add_peak(newpeak)
    #    
    #    if debug:
    #        new = len(self.peaks)
    #        new_total = len(self.all_peaks())
    #        print (f'Was {old} peaks ({old_total} total); now {new} peaks ({new_total} total)')
    
    def all_peaks(self):
        '''
        returns all the member peak objects as a list
        '''
        return [peak for peaklists in self.peaks.values() for peak in peaklists]
    
    def list_unique_peaknames(self):
        '''
        returns only the unique peak objects as a list
        '''
        return sorted([peaks for peaks in self.peaks.keys()])
    
    def count_fragments_by_peakname(self):
        '''
        returns list of tuples of number of locations by unique peak sequence
        '''
        return [(seq, self.nfrags(seq)) for seq in self.list_unique_peaknames()]
    
    def to_pickle(self, filepath):
        try:
            f = open(Path(filepath), 'wb')
            pickle.dump(self, f, protocol=4)
        
            return True
        except:
            return False
        
    #def prune(self, distance=0):
    #    collapses = 0
    #    for peakname in self.list_unique_peaknames():
    #        for peak1, peak2 in itertools.combinations(self.peaks[peakname], 2):
    #            # necessary for configurable distances
    #            comparison_peak = copy(peak1)
    #            comparison_peak.start = max(0, peak1.start - distance)
    #            comparison_peak.end = peak1.end + distance
    #            
    #            if comparison_peak.overlap(peak2):
    #                self.condense(peak1, peak2)
    #                collapses += 1
    #        
    #    return collapses
    
    def prune(self, distance=0):
        collapses = 0
        for peakname in self.list_unique_peaknames():
            
            # algorithm assumes that aggregation for a given distance is transitive
            # i.e. a first greedy pass will get maximum aggregation, so only need to do 
            #      each pairwise comparison exactly once, aggregating as we go
            
            # this is a double loop design, where we pop the first value from D1,
            #    aggregate everything in D1 that can be aggregated,
            #    and put those that can't in D2.
            # Then, we put that first value in D3, move D2 back to D1, pop a new value from D1, and repeat
            # We halt based off exhausting D1 
            # Finally, we set self.peaks[peakname] (the list of peaks) to D3 (the processed peaks)
            D1 = self.peaks[peakname].copy()   # deque to pop from
            D2 = deque()                       # deque to put used values
            D3 = deque()                       # deque to put fully aggregated values
            
            while len(D1) != 0:
                peak1 = D1.popleft()
                
                for n in range(0, len(D1)):
                    peak2 = D1.popleft()
                    
                    # necessary for configurable distances
                    comparison_peak = copy(peak1)
                    comparison_peak.start = max(0, peak1.start - distance)
                    comparison_peak.end = peak1.end + distance
                    
                    if comparison_peak.overlap(peak2):
                        peak1 = self.condense(peak1, peak2)
                        collapses += 1
                    else:
                        D2.append(peak2)
                
                D3.append(peak1)
                D1 = D2
                D2 = deque()
                
            self.peaks[peakname] = D3
            
        return collapses
    
    def condense(self, peak1, peak2):
        '''condense the given two peaks together, returning the new peak object'''
        
        assert peak1.name == peak2.name
        assert peak1.contig == peak2.contig
        
        # create new peak object
        newstart = min(peak1.start, peak2.start)
        newend   = max(peak1.end, peak2.end)
        newpeak = Peak(peak1.contig, newstart, newend, peak1.name)
    
        ## delete old objects
        #if peak1 in self[peak1.name]: self[peak1.name].remove(peak1)
        #if peak2 in self[peak2.name]: self[peak2.name].remove(peak2)
        #
        ## add new peak object
        #self.add_peak(newpeak)
        
        #return
        return newpeak
        
    def __getitem__(self, index):
        return self.peaks[index]
        
    def __iter__(self):
        #return iter(list(self.peaks))
        return iter(self.peaks.items()) # we want to iterate over the recipes
    
    def __repr__(self):
        return f"peakbucket: {len(self.peaks)} unique peaks; {len(self.all_peaks())} total peaks"
    
    def add_from_pandas(self, df, contig='contig', start='start', end='end', name='name'):
        '''from a pandas df with columns for contig, start, end, name, named as specified'''
        
        for i, row in df.iterrows():
            self.add_peak(Peak(row[contig], row[start], row[end], row[name]))
        
        return
    
    def to_pandas(self):
        '''to a pandas df'''
        return pd.DataFrame([[p.contig, p.start, p.end, p.name] for p in self.all_peaks()],
                           columns = ['contig', 'start', 'end', 'name'])
    
def overlap(peak1, peak2):
    '''convenience function for checking 2 Peaks for a strict overlap
    for configurable distance, use overlap2.'''
    assert isinstance(peak1, Peak)
    assert isinstance(peak2, Peak)
    
    return peak1.overlap(peak2)

def overlap2(peak1, peak2, distance):
    '''allows configurable distance'''
    assert isinstance(peak1, Peak)
    assert isinstance(peak2, Peak)
    
    comparison_peak = copy(peak1)
    comparison_peak.start = max(0, peak1.start - distance)
    comparison_peak.end = peak1.end + distance

    return comparison_peak.overlap(peak2)

def condense(peak1, peak2, assert_pkname=True):
    '''condense the given two peaks together, returning the new peak object'''
        
    if assert_pkname: assert peak1.name == peak2.name
    assert peak1.contig == peak2.contig

    # create new peak object
    newstart = min(peak1.start, peak2.start)
    newend   = max(peak1.end, peak2.end)
    newpeak = Peak(peak1.contig, newstart, newend, peak1.name)

    return newpeak
