# utility functions for wrapping halLiftover
#  ----- EBZ


import subprocess
from peakbucket import Peak, PeakBucket, overlap, overlap2, condense
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
import builtins

_TEMP_DIR = Path('/tmp/eztemp/halliftover')
if not _TEMP_DIR.exists(): _TEMP_DIR.mkdir(parents=True)

file_to_species = {
    "A01_A02_A03": "Drosophila_sechellia",
    "A04_A05_A06": "Drosophila_suzukii",
    "A07_A08_A09": "Drosophila_biarmipes",
    "A10_A11_A12": "Drosophila_yakuba",
    "A13_A14_A15": "Drosophila_melanogaster",
    "A16_A17_A18": "Drosophila_simulans",
    "A19_A20_A21": "Drosophila_santomea",
}

def dispatch(species_in,
             species_out = 'Drosophila_melanogaster',
             tempinfile = '/tmp/eztemp/halliftover/tmppeaks.bed4',
             tempoutfile='/tmp/eztemp/halliftover/tmppeaks.out.bed4',
             halFile = '/ru-auth/local/home/ezheng/scratch/Dro7genomes/Dro7genomes.hal'):
    '''runs halLiftover as a Popen subprocess'''

    # halLiftover run
    proc = subprocess.Popen(args=[str('/ru-auth/local/home/ezheng/src/cactus-bin-v1.0.0/bin/halLiftover'),
                                    halFile,
                                    species_in,
                                    tempinfile,
                                    species_out,
                                    tempoutfile],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding = 'utf-8')
    
    while True:
        output = proc.stdout.readline()
        
        #if output == '' and proc.poll() is not None:
        if proc.poll() is not None:
            break
        if output:
            print(output.strip())
            
    return proc.returncode

def aggregate_peak_fragments(df, distance=100):
    '''takes a pandas df of halliftover peaks and saves an aggregated version 
        where peaks at given distance or closer are aggregated'''
    
    assert isinstance(df, pd.DataFrame)
    pks = PeakBucket()
    pks.add_from_pandas(df, start = 'start.hal', end='end.hal', name='peak_name')
    print(pks)
    print(pks.prune(), "peaks aggregated at distance 0.")
    print(pks)
    print(pks.prune(distance=30), "peaks aggregated at distance 30.")
    print(pks)
    print(pks.prune(distance=distance), f"peaks aggregated at distance {distance}.")
    print(pks)
    print('======================')
    pksdf = pks.to_pandas()
    
    #pksdf.to_csv(f'./{fileacc}.bed4', header=False, index=False, sep='\t')
    
    return pksdf

def ezHalLiftover(halFile, species1, species2, beddf, aggregation_distance = 100):
    '''wrapper for a call to halLiftover
    takes a df of a pre-cleaned BED file
    returns a df of the aggregated results'''
    tmp_in_file  = _TEMP_DIR / 'tmppeaks.bed4'
    tmp_out_file = _TEMP_DIR / 'tmppeaks.out.bed4'
    
    # make the bed4 and output diagnostics
    print(species1, 'to', species2, ":")
    print(f"read {beddf.shape[0]} lines ", end='')

    # reset columns back to numbers
    beddf = beddf.rename(columns = dict(map(reversed, enumerate(beddf.columns))))
    
    peaks_bed4 = beddf.iloc[:, [0, 1, 2, 3]].copy()
    
    n_unique_peaks = len(peaks_bed4[3].drop_duplicates())
    print(f"comprising {n_unique_peaks} peaks")

    peaks_bed4.to_csv(tmp_in_file, header=False, index=False, sep='\t')
    
    # run halLiftover
    retcode = dispatch(species_in = species1,
                       species_out = species2,
                       tempinfile = tmp_in_file,
                       tempoutfile= tmp_out_file,
                       halFile = halFile)
    
    if retcode != 0:
        raise ValueError(f'ERROR: got return code {retcode}')
    
    # read back results and post-process
    results = pd.read_csv(tmp_out_file,
                          sep='\t', names=['contig', 'start.hal', 'end.hal', 'peak_name'])

    print('results:')
    print(f'read {results.shape[0]} lines.')
    
    matches = peaks_bed4[~peaks_bed4[3].isin(results['peak_name'])].drop_duplicates(subset=3)
    print(f"{matches.shape[0]} of {n_unique_peaks} peaks unmatched.")
    print('======================')
    
    # stick duplicate peaks back together
    newresults = aggregate_peak_fragments(results, distance = aggregation_distance) # BED4-equivalent
    
    newresults_full = pd.merge(newresults, beddf, 
             how='left', left_on = ['name'], 
             right_on = [3]).drop([0, 1, 2, 3], axis=1)
    
    #newresults.to_csv(f'./{fileacc}.bed4', header=False, index=False, sep='\t')
    
    return newresults_full

def posthoc_reaggregate(acc, distance=100, retvalue='npks'):
    df = pd.read_csv(f'./{acc}.bed4', sep='\t', names=['contig', 'start.hal', 'end.hal', 'peak_name'])
    
    assert isinstance(df, pd.DataFrame)
    assert retvalue in ['npks', 'df']
    
    pks = PeakBucket()
    pks.add_from_pandas(df, start = 'start.hal', end='end.hal', name='peak_name')
    print(pks)
    npruned = pks.prune(distance=distance)
    print(npruned, f"peaks aggregated at distance {distance}.")
    print(pks)
    print('======================')
    
    if retvalue == 'npks':
        return npruned
    elif retvalue == 'df':
        pksdf = pks.to_pandas()
        return pksdf
    else:
        raise ValueError
        
def deduplicate_peak_names(bed_df, name_col = 3, stem = '_hal'):
    '''deduplicates any duplicated peak names. returns new DF.'''
    
    # surprisingly clever trick from https://stackoverflow.com/questions/57804697/

    name_groups = bed_df.groupby(name_col)[name_col]
    suffix = name_groups.cumcount()+1
    repeats = name_groups.transform('size')

    new = np.where(repeats > 1, bed_df[name_col] + stem + suffix.map(str), bed_df[name_col])
    
    newdf = bed_df.copy()
    newdf[name_col] = new
    
    return newdf

def contained_in_opb(d, refpb=None, distance = 100, output='bool'):
    # get reference peaks
    g = d.name
    g = d.name.split('_hal')[0]
    refs = refpb[g]
    assert len(refs) > 0
    
    # generate a PeakBucket of the back-mapped fragments
    _ = PeakBucket()
    _.add_from_pandas(d)
    _.prune(distance) # sometimes the chunks are trivially combinable
    
    outlist = []
    for q in _.all_peaks():
        for ref in refs:
            if overlap2(q, ref, distance):
                outlist.append(q)
    
    if output == 'bool':
        return (len(outlist) > 0)
    elif output == 'list':
        return outlist
    else:
        raise ValueError
        
def exact_refpb(s, refpb):

    return (Peak(s.contig, s.start, s.end, s['name']) in refpb.peaks[s['name']])

def evaluate_symmetry(original, mapped, deduped_mapped, backmapped, distance, log=None):
    '''Evaluates symmetry of back-mapping.
    Given dfs of the original BED file,
                 the mapped BED file,
                 the deduplicated mapped BED file,
                 and the backmapped BED file,
    returns 1) a BED df of only those mapped peaks that singly back-map within the given distance (default 0)
        and 2) as above, but of only those that multiply back-map as above.
    
    Set log to a valid file path to redirect detailed mismatch diagnostics to there.'''
    # mapped peaks
    mapped = mapped.rename(columns={0:'contig',1:'start',2:'end',3:'name'})

    # original peaks (no roundtripped) peaks
    original = original.rename(columns={0:'contig',1:'start',2:'end',3:'name'})

    # generate a PeakBucket of the original in-species (no mapping) peaks
    refpb = PeakBucket()
    refpb.add_from_pandas(original)

    print = builtins.print
    print('reference', refpb)

    # round-tripped peaks
    backmapped['origin'] = backmapped.name.str.split('_hal').str[0]
    deduped_mapped['origin'] = deduped_mapped.name.str.split('_hal').str[0]
    
    # characterize *exact* matches
    backmapped['exact'] = backmapped.apply(exact_refpb, args=(refpb,), axis=1)
    
    no_exact_matches = backmapped.groupby('name').apply(lambda _: ~_['exact'].any())
    
    print("Exact matches:", len(backmapped[backmapped['exact']].origin.drop_duplicates()))
    print("Nonexact matches:", len(no_exact_matches.index.drop_duplicates()))
    
    # generate a list of the mapped peak names (deduplicated) that backmap within parameters
    # below yields a pd.Series of lists of Peaks
    overlaplist = backmapped.groupby('name').apply(contained_in_opb, refpb, distance=distance, output='list')
    # and an actual flat list of names
    overlap_mapped_names = [peak.name for i, j in overlaplist.iteritems() for peak in j]
    
    # N of back-mapping overlaps per mapped peak (deduplicated)
    # thus, this is actually one-mapped-to-many-backmapped,
    # which is indep. of orig-to-mapped frequency
    overlap_frequency = [len(j) for i, j in overlaplist.iteritems()]
    print("Distribution how many mapped peaks multi-map in the original species:")
    print(pd.Series(overlap_frequency).value_counts())
    
    # details on overlap matches
    supported_mapped = deduped_mapped[deduped_mapped['name'].isin(overlap_mapped_names)]
    
    # N of supported mapped peaks per ORIGINAL peak
    print("Distribution how many mapped peaks multi-map in the original species:")
    print(supported_mapped.origin.value_counts().value_counts().sort_index())
    
    overlap_multimapped_names = (supported_mapped.groupby('origin').apply(len) > 1)
    overlap_multimapped_names = overlap_multimapped_names[overlap_multimapped_names].index.values
    overlap_multimapped = supported_mapped[supported_mapped.origin.isin(overlap_multimapped_names)]
    overlap_singlemapped = supported_mapped[~supported_mapped.origin.isin(overlap_multimapped_names)]
    
    all_overlap = (overlap_multimapped.shape[0] + overlap_singlemapped.shape[0] == deduped_mapped.shape[0])
    print("Does every mapped peak back-map at least once to the original peak?", all_overlap)
    
    contig_integrity = True
    contig_mismatch = 0
    
    # set up external logging if path to log is passed
    if log:
        log = Path(log).expanduser()
        assert log.parent.exists()
        if log.exists():
            print(f"Warning: log file {log} exists!")
        
        f = open(Path(log), 'a')
        print = partial(print, file=f)
        
    for i, row in overlap_multimapped.iterrows():

        peak_name = row['name'] # name of peaks (with _hal)
        orig_name = peak_name.split('_hal')[0] # name w/o hal
        print(peak_name)

        #halnames = nr[nr.name == pn].orig_name.values
        mapped_contigs = mapped[mapped['name']==orig_name].contig.drop_duplicates()
        if len(mapped_contigs) != 1:
            print("All fragments are NOT on same mapped contig!")
            print(mapped_contigs)
            contig_integrity = False
            contig_mismatch += 1
        
        print(Peak(row['contig'], row['start'], row['end'], row['name']))
        print('original unmapped peak:')
        print(refpb.peaks[orig_name][0])
        print('====\n\n')
    
    # put things back
    if log:
        print = builtins.print
        f.close()
    
    if not contig_integrity:
        print(f'{contig_mismatch} peaks are on different contigs and may have translocated.')
    
    singleton_backmap_pb = PeakBucket()
    multiple_backmap_pb = PeakBucket()
    
    # note that PeakBucket.add_from_pandas modifies the object IN-PLACE
    singleton_backmap_pb.add_from_pandas(overlap_singlemapped)
    multiple_backmap_pb.add_from_pandas(overlap_multimapped)
    
    print("singleton", singleton_backmap_pb)
    print("multiple", multiple_backmap_pb)
    
    singletons = pd.merge(singleton_backmap_pb.to_pandas(), original.iloc[:,3:],
                         how='left', left_on = ['name'],
                         right_on = ['name'])
    multiples = multiple_backmap_pb.to_pandas()
    multiples['name'] = multiples['name'].str.split('_hal').str[0]
    multiples =  pd.merge(multiples, original.iloc[:,3:],
                         how='left', left_on = ['name'],
                         right_on = ['name'])
    
    return singletons, multiples

def ezMapBed(beddf, halFile, species1, species2, filename,
             output_dir = './', intermediate_files = True,
             agg_distance = 100, back_distance = 100):
    '''
    given a pre-cleaned beddf and a halFile,
    use halLiftover to remap the intervals in the beddf
    from species 1 to species 2
    
    writes to output_dir by default with filename stem given by filename.
    writes intermediate progress files by default.
    
    agg_distance controls the distance with which ezHalLiftover aggregates peaks
        (default 100 bp)
    back_distance controls the tolerance for back-mapping peaks
        (default 100)
    '''
    output_dir = Path(output_dir).expanduser()
    assert output_dir.exists() and output_dir.is_dir()
    
    original = beddf
    mapped = ezHalLiftover(halFile = halFile,
                             species1 = species1,
                             species2 = species2,
                             beddf = original,
                             aggregation_distance = agg_distance)
    if intermediate_files:
        mapped.to_csv(output_dir / f'{filename}.fwd.narrowPeak', header=False, index=False, sep='\t')
        mapped.iloc[:,0:4].to_csv(output_dir / f'{filename}.fwd.bed4', header=False, index=False, sep='\t')

    dedupe_mapped = deduplicate_peak_names(mapped, name_col='name')

    backmapped = ezHalLiftover(halFile = halFile,
                             species1 = species2,
                             species2 = species1,
                             beddf = dedupe_mapped,
                             aggregation_distance = agg_distance)
    
    if intermediate_files:
        backmapped.to_csv(output_dir / f'{filename}.bck.narrowPeak', header=False, index=False, sep='\t')
        backmapped.iloc[:,0:4].to_csv(output_dir / f'{filename}.bck.bed4', header=False, index=False, sep='\t')
    
    if intermediate_files:
        log = output_dir / f'{filename}.bck.log'
    else:
        log = None
    
    single, multiple = evaluate_symmetry(original, mapped, dedupe_mapped, backmapped, log=log,
                                         distance=back_distance)

    single.to_csv(output_dir / f'{filename}.single.narrowPeak', header=False, index=False, sep='\t')
    multiple.to_csv(output_dir / f'{filename}.multiple.narrowPeak', header=False, index=False, sep='\t')
    
    return
