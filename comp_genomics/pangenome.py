from collections import defaultdict
from peakbucket import PeakBucket, Peak
from itertools import groupby
from overlap_ops import classify, name_intersect
from pathlib import Path

class PanGenomeBucket():
    '''a pangenome generalization of PeakBucket; holds multiple PeakBuckets'''
    def __init__(self, data, native):
        '''constructor: a dict-like of spname: PeakBuckets
        must specify which is the native species (TBD change?)'''
        assert isinstance(data, dict)
        self.buckets = defaultdict(PeakBucket, data)
        self.native = native
        if self.native not in self.buckets.keys():
            print(f"Warning: native species '{self.native}' not in data\nBehavior may be unexpected!")

    def __repr__(self):
        out = f"{self.native} native peaks:\n"
        for s in self.species():
            out+= f"{s}: {self.buckets[s].__repr__()}\n"
        return out

    def __getitem__(self, index):
        return self.buckets[index]
    
    def __iter__(self):
        return iter(self.buckets.items())
    
    def mapped(self, relative_to='native'):
        '''returns peakbucket of mapped peaks relative_to (default native)
        #~~for now, does NOT assume only single-mapped ones count~~#
        '''
        if relative_to == 'native':
            relative_to = self.native

        species = set(self.species()) - set([relative_to])
        new_bucket = self.buckets[relative_to]
        for sp in species:
            new_bucket = pb_name_intersect(new_bucket,
                           self.buckets[sp])[0]
            # we take the first value since pb_name_intersect returns
            # buckets in order of input
        return new_bucket
    
    def prune(self):
        '''prunes all member peakbuckets to be conformant (same peaks throughout)'''
        for sp in self.species():
            self.buckets[sp] = self.mapped(relative_to=sp)
        return self
    
    def prune_multimapped(self):
        '''prunes all member peakbuckets to drop multimapped peaks'''
        for sp in self.species():
            self.buckets[sp] = drop_multimapped(self.buckets[sp])
        self.prune()
        return self
    
    def species(self):
        '''lists contained species. Guaranteed to return native first'''
        non_native = set(self.buckets.keys()) - set([self.native])
        return [self.native] + sorted(non_native)
    
    def conformant(self):
        '''check if the pgb is conformant
        that is, if the set of peaks across each genome is identical'''
        return all_equal((set(self[y].list_unique_peaknames()) for y in self.species()))
    
    def list_peaks(self):
        '''return a Set of the peaks contained in the pgb.
        Only defined (and will complain if not true) if same peaks across all genome contexts'''
        if not self.conformant:
            raise ValueError("Not conformant -- listing peaks not well defined. call self.prune()?")
        else:
            return set(self[self.native].list_unique_peaknames())
    
    def subtract_peaks(self, fpeaks):
        '''subtracts the listed peaks from all member peakbuckets'''
        fpeaks = set(fpeaks)
        currpeaks = self.list_peaks()
        
        keep = currpeaks - fpeaks
        
        for sp in self.species():
            triaged = PeakBucket()
            A = self.buckets[sp]
            for p in [p for p in A.all_peaks() if p.name in keep]: triaged.add_peak(p)
            self.buckets[sp] = triaged
        
        assert self.conformant()
        return self
    
    def intersect_peaks(self, fpeaks):
        '''keeps only the listed peaks from all member peakbuckets'''
        fpeaks = set(fpeaks)
        currpeaks = self.list_peaks()
        
        keep = currpeaks & fpeaks
        
        for sp in self.species():
            triaged = PeakBucket()
            A = self.buckets[sp]
            for p in [p for p in A.all_peaks() if p.name in keep]: triaged.add_peak(p)
            self.buckets[sp] = triaged
        
        assert self.conformant()
        return self

    def filter_peaks_to_chr(self, chrom):
        '''keeps only peaks on the given chrom from all member peakbuckets
        chrom can be either a string or an iterable of strings'''
        # handle the single arg or iterable arg
        if isinstance(chrom, str): chrom = [chrom]

        for sp in self.species():
            triaged = PeakBucket()
            A = self.buckets[sp]
            for p in [p for p in A.all_peaks() if p.contig in chrom]: triaged.add_peak(p)
            self.buckets[sp] = triaged

        print(f'debug:{self.conformant()}')
        return self

    def symm_diff_peaks(self, fpeaks):
        '''NOT IMPLEMENTED'''
        raise ValueError("NOT IMPLEMENTED")
        
    def copy(self):
        '''DEEP copy of self'''
        from copy import deepcopy
        return deepcopy(self)

    def to_narrowPeak(self, directory, stem):
        '''save to directory w leading stem
        e.g. full format: path/to/dir/stem.mel_to_mel.narrowPeak'''
        for sp in self.species():
            self.buckets[sp].to_pandas().to_csv(Path(directory)/str(stem+f'{self.native}_to_{sp}.narrowPeak'),
                                                header=False, index=False, sep='\t')
        return

class PanGenomeOrthologyResult():
    '''helper class to ease cat-herding'''
    def __init__(self, uniq=None, cons=None, source=None, context=None):
        self.unique = uniq
        self.conserved = cons
        self.source = source
        self.context = context
    
    def __repr__(self):
        return f"u: {self.unique.__repr__()}, c: {self.conserved.__repr__()}"
    
    def drop_multimapped(self):
        try:
            self.unique.prune_multimapped()
            self.conserved.prune_multimapped()  
            return self
        except:
            print('invalid operation')
            return

class PanGenomeOrthology():
    '''helper class to store the full PanGenome orthology relationships'''
    pass

def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def _get_names(peak_list):
    assert isinstance(peak_list[0], Peak)
    
    return set((p.name for p in peak_list))

def _pgb_pairwise(pgb1, pgb2, context, distance):
    res = classify(pgb1[context].all_peaks(), pgb2[context].all_peaks(), distance)
        
    pgb1_n_overlaps = _get_names(res[0])
    pgb1_n_unique   = _get_names(res[1])
    pgb1_n_res = PanGenomeOrthologyResult(uniq=pgb1_n_unique,
                                          cons=pgb1_n_overlaps,
                                          source=pgb1.native,
                                          context=context)

    pgb2_n_overlaps = _get_names(res[2])
    pgb2_n_unique   = _get_names(res[3])
    pgb2_n_res = PanGenomeOrthologyResult(uniq=pgb2_n_unique,
                                          cons=pgb2_n_overlaps,
                                          source=pgb2.native,
                                          context=context)
    return pgb1_n_res, pgb2_n_res

def _pgb_zip_results(pgb, results):
    '''given a pgb and an iterable of PanGenomeOrthologyResults, return a final PGOR'''
    final_pgb = PanGenomeOrthologyResult(uniq=pgb.copy(),
                                          cons=pgb.copy())
    
    for res in results: final_pgb.unique.subtract_peaks(res.conserved)
    for res in results: final_pgb.unique.intersect_peaks(res.unique)
    
    for res in results: final_pgb.conserved.intersect_peaks(res.conserved)
    
    return final_pgb

def pgb_compare(pgb1, pgb2, distance=100):
    
    pgb1_n_res = []
    pgb2_n_res = []
    for sp in set(pgb1.species()) & set(pgb2.species()):
        r1, r2 = _pgb_pairwise(pgb1, pgb2, sp, distance=distance)
        pgb1_n_res.append(r1)
        pgb2_n_res.append(r2)
    
    final_pgb1 = _pgb_zip_results(pgb1, pgb1_n_res)
    final_pgb2 = _pgb_zip_results(pgb2, pgb2_n_res)
    
    return final_pgb1, final_pgb2

def pb_name_intersect(A, B):
    '''intersect 2 peakbuckets based off shared names;
    returns the pb intersects from A and from B, separately
    (that preserves the coord info)'''
    
    keep = name_intersect(A, B)
    triaged_A = PeakBucket()
    triaged_B = PeakBucket()
    for p in [p for p in A.all_peaks() if p.name in keep]: triaged_A.add_peak(p)
    for p in [p for p in B.all_peaks() if p.name in keep]: triaged_B.add_peak(p)
    return triaged_A, triaged_B

def pb_name_subtract(A, B):
    '''subtract members of B from A based off shared names;
    returns A'''
    
    keep = name_subtract(A, B)
    triaged_A = PeakBucket()
    for p in [p for p in A.all_peaks() if p.name in keep]: triaged_A.add_peak(p)
    return triaged_A

def drop_multimapped(peakbucket):
    '''return a peakbucket without multiply-mapped peaks'''
    
    keep = [t[0] for t in peakbucket.count_fragments_by_peakname() if t[1] == 1]

    triaged = PeakBucket()
    for p in [p for p in peakbucket.all_peaks() if p.name in keep]: triaged.add_peak(p)
    return triaged
