/** The registry of `aifn-methods/unsupervised/clustering`: the clustering procedures as traceable algorithms. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as centroid from './centroid'
import * as density from './density'
import * as hierarchical from './hierarchical'
import * as mixture from './mixture'
import * as spectral from './spectral'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'unsupervised/clustering')
const fn = definer<FunctionInfo>('function', 'unsupervised/clustering')

algorithm(
  {
    key: 'kmeansSteps',
    name: "Lloyd's k-means",
    summary: 'Assign each point to its nearest centroid, then move each centroid to its cluster mean.',
    problem: 'objective',
    state: { iterate: 'centroids', objective: 'inertia', flags: ['converged'] },
    random: true,
    notes: ['k-means'],
    cite: ['lloyd1982', 'arthur2007'],
  },
  centroid.kmeansSteps,
)
algorithm(
  {
    key: 'miniBatchKMeansSteps',
    name: 'Mini-batch k-means',
    problem: 'objective',
    state: { iterate: 'centroids', objective: 'inertia', flags: [] },
    random: true,
    notes: ['k-means'],
  },
  centroid.miniBatchKMeansSteps,
)
algorithm(
  {
    key: 'kMedoidsSteps',
    name: 'k-medoids (PAM swaps)',
    problem: 'objective',
    state: { iterate: 'medoids', objective: 'cost', flags: ['converged'] },
    notes: ['k-means'],
  },
  centroid.kMedoidsSteps,
)
algorithm(
  {
    key: 'gaussianMixtureSteps',
    name: 'EM for a Gaussian mixture',
    summary: 'E-step responsibilities, M-step weights, means and covariances; the log-likelihood never decreases.',
    problem: 'objective',
    state: { iterate: 'means', objective: 'logLikelihood', flags: ['converged', 'diverged'] },
    random: true,
    notes: ['gaussian-mixture-model', 'expectation-maximisation'],
    cite: ['dempster1977'],
  },
  mixture.gaussianMixtureSteps,
)
algorithm(
  {
    key: 'meanShiftSteps',
    name: 'Mean shift',
    problem: 'objective',
    state: { iterate: 'seeds', objective: 'shift', flags: ['converged'] },
    notes: ['density-based-spatial-clustering', 'kernel-density-estimation'],
  },
  density.meanShiftSteps,
)
algorithm(
  {
    key: 'agglomerativeSteps',
    name: 'Agglomerative clustering',
    summary: 'Merge the two closest clusters per step under a linkage rule.',
    problem: 'graph',
    state: { iterate: 'labels', flags: [] },
    notes: ['hierarchical-clustering'],
    cite: ['lance1967', 'ward1963', 'mullner2011'],
  },
  hierarchical.agglomerativeSteps,
)
fn(
  { key: 'linkage', name: 'Linkage matrix', role: 'fit', notes: ['hierarchical-clustering'], cite: ['mullner2011'] },
  hierarchical.linkage,
)
fn(
  {
    key: 'dendrogram',
    name: 'Dendrogram layout',
    role: 'transform',
    returns: 'tree',
    notes: ['hierarchical-clustering'],
  },
  hierarchical.dendrogram,
)
fn(
  {
    key: 'cutTree',
    name: 'Cut a dendrogram',
    role: 'transform',
    notes: ['hierarchical-clustering', 'choosing-the-number-of-clusters'],
  },
  hierarchical.cutTree,
)
fn(
  { key: 'mergeTree', name: 'Merge tree', role: 'transform', returns: 'tree', notes: ['hierarchical-clustering'] },
  hierarchical.mergeTree,
)
fn(
  {
    key: 'affinityMatrix',
    name: 'Affinity matrix',
    role: 'construction',
    notes: ['spectral-clustering'],
    cite: ['ng2002', 'vonluxburg2007'],
  },
  spectral.affinityMatrix,
)

/** The algorithms of the module, keyed by factory name. */
export const clusteringAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  centroid,
  mixture,
  density,
  hierarchical,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const clusteringFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  centroid,
  hierarchical,
  spectral,
) as Table<FunctionInfo>
