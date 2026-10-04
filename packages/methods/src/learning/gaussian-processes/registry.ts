/**
 * The registry of `aifn-methods/learning/gaussian-processes`: GP regression and classification functions, the GPLVM
 * fit as a traceable algorithm. (The sparse GP and RVM algorithms register themselves in `sparse.ts` and `rvm.ts`.)
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as classification from './classification'
import * as classificationEp from './classification-ep'
import * as gplvm from './gplvm'
import * as ordinal from './ordinal'
import * as regression from './regression'
import * as rvm from './rvm'
import * as sparse from './sparse'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/gaussian-processes')
const fn = definer<FunctionInfo>('function', 'learning/gaussian-processes')
const GP = ['gaussian-process']
const HYPER = ['gaussian-process-hyperparameter-learning']
const GPC = ['gaussian-process-classification']
const LVM = ['gaussian-process-latent-variable-model']

algorithm(
  {
    key: 'gplvmFitSteps',
    name: 'GPLVM fit',
    summary: 'L-BFGS on the latent positions and kernel hyperparameters of a GPLVM, maximising the log posterior.',
    problem: 'objective',
    state: {
      iterate: 'latent',
      objective: 'logPosterior',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged'],
    },
    notes: LVM,
    cite: ['lawrence2005'],
  },
  gplvm.gplvmFitSteps,
)
fn({ key: 'fitGplvm', name: 'Fit a GPLVM', role: 'fit', notes: LVM, cite: ['lawrence2005'] }, gplvm.fitGplvm)
fn({ key: 'gplvmProblem', name: 'GPLVM problem', role: 'construction', notes: LVM }, gplvm.gplvmProblem)
fn({ key: 'gplvmModel', name: 'GPLVM model', role: 'construction', notes: LVM }, gplvm.gplvmModel)

fn(
  {
    key: 'gpPrior',
    name: 'GP prior',
    role: 'construction',
    notes: [...GP, 'covariance-functions'],
    cite: ['rasmussen2006'],
  },
  regression.gpPrior,
)
fn(
  {
    key: 'samplePrior',
    name: 'Sample a GP prior',
    role: 'simulation',
    random: true,
    notes: [...GP, 'covariance-functions'],
  },
  regression.samplePrior,
)
fn(
  {
    key: 'gpPosterior',
    name: 'GP posterior',
    summary: 'The posterior mean and covariance by a Cholesky factor of K + σ²I.',
    role: 'inference',
    notes: [...GP, 'gaussian-processes-and-bayesian-linear-regression'],
    cite: ['rasmussen2006'],
  },
  regression.gpPosterior,
)
fn(
  {
    key: 'logMarginalLikelihood',
    name: 'GP log marginal likelihood',
    role: 'estimator',
    notes: [...HYPER, ...GP],
    cite: ['rasmussen2006'],
  },
  regression.logMarginalLikelihood,
)
fn(
  {
    key: 'logMarginalLikelihoodGradient',
    name: 'Gradient of the GP log marginal likelihood',
    role: 'estimator',
    notes: HYPER,
    cite: ['rasmussen2006'],
  },
  regression.logMarginalLikelihoodGradient,
)
fn(
  { key: 'kernelLogVector', name: 'Kernel hyperparameters as a log vector', role: 'transform', notes: HYPER },
  regression.kernelLogVector,
)
fn(
  {
    key: 'fitGp',
    name: 'Fit GP hyperparameters',
    role: 'fit',
    notes: [...HYPER, 'automatic-relevance-determination'],
    cite: ['rasmussen2006'],
  },
  regression.fitGp,
)
fn(
  {
    key: 'laplaceMode',
    name: 'Laplace mode (Newton)',
    role: 'inference',
    notes: [...GPC, 'laplace-approximation'],
    cite: ['williams1998', 'rasmussen2006'],
  },
  classification.laplaceMode,
)
fn(
  {
    key: 'laplaceLogMarginal',
    name: 'Laplace log marginal likelihood',
    role: 'estimator',
    notes: [...GPC, 'laplace-approximation'],
  },
  classification.laplaceLogMarginal,
)
fn({ key: 'laplaceEvidence', name: 'Laplace evidence', role: 'estimator', notes: GPC }, classification.laplaceEvidence)
fn(
  { key: 'gpClassifierEvidenceGradient', name: 'GP classifier evidence gradient', role: 'estimator', notes: GPC },
  classification.gpClassifierEvidenceGradient,
)
fn(
  {
    key: 'gpEpLogMarginal',
    name: 'EP log marginal likelihood (GP classification)',
    role: 'estimator',
    notes: [...GPC, 'expectation-propagation-gaussian-process-classification'],
  },
  classification.gpEpLogMarginal,
)
fn({ key: 'fitGpClassifier', name: 'Fit a GP classifier', role: 'fit', notes: GPC }, classification.fitGpClassifier)
fn(
  {
    key: 'gpEp',
    name: 'GP classification by EP',
    role: 'inference',
    notes: ['expectation-propagation-gaussian-process-classification', ...GPC],
    cite: ['minka2001', 'kuss2005'],
  },
  classificationEp.gpEp,
)
fn(
  {
    key: 'gpEpEvidence',
    name: 'GP classification EP evidence',
    role: 'estimator',
    notes: ['expectation-propagation-gaussian-process-classification'],
  },
  classificationEp.gpEpEvidence,
)
fn(
  {
    key: 'ordinalLaplaceTerms',
    name: 'Ordinal GP Laplace terms',
    role: 'estimator',
    notes: ['gaussian-process-ordinal-regression'],
  },
  ordinal.ordinalLaplaceTerms,
)
fn(
  {
    key: 'sparseGp',
    name: 'Sparse GP (inducing points)',
    role: 'inference',
    notes: ['sparse-gaussian-processes'],
    cite: ['titsias2009'],
  },
  sparse.sparseGp,
)
fn(
  { key: 'sparseGpAt', name: 'Sparse GP prediction', role: 'inference', notes: ['sparse-gaussian-processes'] },
  sparse.sparseGpAt,
)
fn(
  {
    key: 'sparseLogMarginal',
    name: 'Sparse GP bound',
    role: 'estimator',
    notes: ['sparse-gaussian-processes'],
    cite: ['titsias2009'],
  },
  sparse.sparseLogMarginal,
)
fn(
  { key: 'fitSparseGp', name: 'Fit a sparse GP', role: 'fit', notes: ['sparse-gaussian-processes'] },
  sparse.fitSparseGp,
)
fn(
  {
    key: 'rvmPosterior',
    name: 'RVM posterior',
    role: 'inference',
    notes: ['relevance-vector-machine'],
    cite: ['tipping2001'],
  },
  rvm.rvmPosterior,
)
fn(
  { key: 'rvmProblem', name: 'RVM problem', role: 'construction', notes: ['relevance-vector-machine'] },
  rvm.rvmProblem,
)
fn({ key: 'rvmModel', name: 'RVM model', role: 'construction', notes: ['relevance-vector-machine'] }, rvm.rvmModel)

/** The algorithms registered here, keyed by factory name. */
export const gaussianProcessAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  gplvm,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const gaussianProcessFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  regression,
  classification,
  classificationEp,
  ordinal,
  sparse,
  rvm,
  gplvm,
) as Table<FunctionInfo>
