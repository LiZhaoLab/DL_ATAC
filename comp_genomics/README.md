## Neural Network scripts

### [pangenome.py](pangenome.py)

Provides the PanGenomeBucket class,
which manages sets of associated intervals
across multiple contexts
(for example, the same set of intervals
in different genomes).

### [ezHalLiftover.py](ezHalLiftover.py)

Provides both a command-line and importable module interface
for using `halLiftover` to extract orthology
from a Cactus HAL graph.

### [peakbucket.py](peakbucket.py)

Provides the PeakBucket class,
which manages intervals within a given context
(for example, a set of intervals in one genome).

### [overlap_ops.py](overlap_ops.py)

Implements operations for intersecting / subtracting different PeakBuckets.

### [overlap_network.py](overlap_network.py)

Implements methods for orthology comparisons
between two sets (as PanGenomeBuckets)
in all contexts (aka across all species).
