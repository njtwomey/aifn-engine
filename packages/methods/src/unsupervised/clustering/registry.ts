/**
 * The registry of `aifn-methods/unsupervised/clustering`: the clustering procedures as traceable algorithms, and the
 * functions of hierarchical and spectral clustering. The estimators register themselves beside their code, with
 * `defineModel`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as centroid from './centroid'
import * as constrained from './constrained'
import * as density from './density'
import * as hierarchical from './hierarchical'
import * as mixture from './mixture'
import * as semiMixture from './semi-supervised-mixture'
import * as spectral from './spectral'

/** A registry table: entries keyed by name, each a function with its metadata of kind `I`. */
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

// Constrained clustering with pairwise side information (constrained.ts).
const CONSTRAINED = ['semi-supervised-clustering']

algorithm(
  {
    key: 'constrainedKMeansSteps',
    name: 'Constrained k-means (COP-k-means, PCK-means)',
    summary:
      'Lloyd’s algorithm whose assignment breaks no must-link or cannot-link (COP) or pays per broken one (PCK).',
    problem: 'objective',
    state: { iterate: 'centroids', objective: 'objective', flags: ['converged'] },
    notes: CONSTRAINED,
    cite: ['wagstaff2001constrained', 'basu2004active'],
  },
  constrained.constrainedKMeansSteps,
)
fn(
  {
    key: 'constraintClosure',
    name: 'Closure of pairwise constraints',
    summary: 'Must-link groups by transitive closure, cannot-links lifted to groups, and any contradiction.',
    role: 'construction',
    notes: CONSTRAINED,
    cite: ['wagstaff2001constrained'],
  },
  constrained.constraintClosure,
)
fn(
  {
    key: 'seededCentroids',
    name: 'Seeded k-means centroids',
    role: 'construction',
    notes: CONSTRAINED,
    cite: ['basu2002seeding'],
  },
  constrained.seededCentroids,
)
fn(
  {
    key: 'relevantComponentMetric',
    name: 'Relevant component analysis',
    tex:
      '\\hat C = \\tfrac1p \\textstyle\\sum_j \\sum_{i \\in G_j} (x_i - \\hat\\mu_j)(x_i - \\hat\\mu_j)^\\top,' +
      '\\; x \\mapsto \\hat C^{-1/2} x',
    summary: 'A Mahalanobis metric from must-link chunklets: whiten by the within-chunklet covariance.',
    role: 'fit',
    notes: CONSTRAINED,
    cite: ['barhillel2005rca'],
  },
  constrained.relevantComponentMetric,
)

// Semi-supervised EM for Gaussian classes (semi-supervised-mixture.ts).

const SSL_EM = [
  'generative-semi-supervised-learning',
  'when-semi-supervised-learning-fails',
  'open-set-semi-supervised-learning',
  'expectation-maximisation',
]
algorithm(
  {
    key: 'semiSupervisedMixtureSteps',
    name: 'Semi-supervised EM for Gaussian classes',
    tex:
      '\\sum_{i \\in L} \\log \\pi_{y_i} \\mathcal{N}(x_i \\mid \\mu_{y_i}, \\Sigma_{y_i}) + ' +
      '\\lambda \\sum_{i \\in U} \\log \\sum_k \\pi_k \\mathcal{N}(x_i \\mid \\mu_k, \\Sigma_k)',
    summary:
      'One Gaussian per class; labelled responsibilities clamped, unlabelled ones re-estimated and weighted by λ.',
    problem: 'objective',
    state: { iterate: 'means', objective: 'objective', flags: ['converged', 'diverged'] },
    notes: SSL_EM,
    cite: ['nigam2000em', 'cozman2003mixture', 'zhu2009intro'],
  },
  semiMixture.semiSupervisedMixtureSteps,
)
fn(
  {
    key: 'labelledOnlyFit',
    name: 'Labelled-only Gaussian class fit',
    summary: 'Class frequencies, means and a pooled (or per-class) covariance from the labelled points alone.',
    role: 'fit',
    notes: SSL_EM,
  },
  semiMixture.labelledOnlyFit,
)
fn(
  {
    key: 'classPosterior',
    name: 'Posterior class probabilities of Gaussian classes',
    tex: 'p(y = k \\mid x) \\propto \\pi_k \\mathcal{N}(x \\mid \\mu_k, \\Sigma_k)',
    role: 'inference',
    notes: SSL_EM,
  },
  semiMixture.classPosterior,
)
fn(
  { key: 'classLogJoint', name: 'Log joint densities of Gaussian classes', role: 'inference', notes: SSL_EM },
  semiMixture.classLogJoint,
)

/** The algorithms of the module, keyed by factory name. */
export const clusteringAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  centroid,
  mixture,
  density,
  hierarchical,
  constrained,
  semiMixture,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const clusteringFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  centroid,
  hierarchical,
  spectral,
  constrained,
  semiMixture,
) as Table<FunctionInfo>
