/**
 * The registry of `aifn-methods/generative/gan`: GAN training as a traceable algorithm, and the networks, the
 * diagnostics against a known density and the streamed run as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as diagnostics from './diagnostics'
import * as networks from './gan'
import * as run from './run'
import * as semi from './semi-supervised'

/** A table of the module's registry entries of one kind, keyed by name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'generative/gan')
const fn = definer<FunctionInfo>('function', 'generative/gan')
const GAN = ['generative-adversarial-network']

algorithm(
  {
    key: 'ganTraining',
    name: 'GAN training',
    summary:
      'Alternating discriminator and generator updates under the minimax, non-saturating, WGAN-GP or hinge game.',
    problem: 'network',
    state: { iterate: 'generator', objective: 'generatorLoss', flags: ['diverged'] },
    random: true,
    notes: GAN,
    cite: ['goodfellow2014', 'arjovsky2017', 'gulrajani2017'],
  },
  networks.ganTraining,
)

fn(
  {
    key: 'gan',
    name: 'GAN networks',
    summary: 'An MLP generator from latent noise to points and an MLP discriminator from points to a score.',
    role: 'construction',
    notes: GAN,
    cite: ['goodfellow2014'],
  },
  networks.gan,
)
fn(
  {
    key: 'modeCoverage',
    name: 'Mode coverage',
    summary: 'Modes of a known mixture holding high-quality generated points, and the share of high-quality points.',
    role: 'estimator',
    notes: GAN,
  },
  diagnostics.modeCoverage,
)
fn(
  {
    key: 'optimalDiscriminator',
    name: 'Optimal discriminator',
    summary: 'D*(x) = p_data/(p_data + p_g), with p_g a Gaussian KDE of generated points.',
    role: 'estimator',
    notes: GAN,
    cite: ['goodfellow2014'],
  },
  diagnostics.optimalDiscriminator,
)
fn(
  {
    key: 'ganRun',
    name: 'Streamed GAN training run',
    summary: 'Train a GAN and yield losses and checkpoints (samples, discriminator field, D*, coverage) for a worker.',
    role: 'simulation',
    random: true,
    notes: GAN,
  },
  run.ganRun,
)

// ── Semi-supervised GANs (Salimans et al. 2016; Dai et al. 2017) ─────────────────────────────────────────────────────

const SSGAN = ['semi-supervised-generative-adversarial-networks']

algorithm(
  {
    key: 'semiSupervisedGanTraining',
    name: 'Semi-supervised GAN training',
    summary:
      'A K-class discriminator whose implicit class K + 1 is “generated”, trained on labels plus the GAN game, against a feature-matching, non-saturating or fixed generator.',
    problem: 'network',
    state: { iterate: 'critic', objective: 'criticLoss', flags: ['diverged'] },
    random: true,
    notes: SSGAN,
    cite: ['salimans2016', 'dai2017bad'],
  },
  semi.semiSupervisedGanTraining,
)
fn(
  {
    key: 'semiSupervisedGan',
    name: 'Semi-supervised GAN networks',
    summary: 'An MLP generator, and a discriminator of a leaky-ReLU trunk and a head of K real-class logits.',
    role: 'construction',
    notes: SSGAN,
    cite: ['salimans2016'],
  },
  semi.semiSupervisedGan,
)
fn(
  {
    key: 'classAndFakeProbabilities',
    name: 'Real-class and generated probabilities',
    tex: 'p(y = k \\mid x) = \\frac{e^{l_k}}{Z(x) + 1},\\quad p(\\text{generated} \\mid x) = \\frac{1}{Z(x) + 1}',
    role: 'transform',
    notes: SSGAN,
    cite: ['salimans2016'],
  },
  semi.classAndFakeProbabilities,
)
fn(
  {
    key: 'complementSampler',
    name: 'Complement generator',
    summary: 'Uniform points in a box, kept only when farther than r from every data point: a “bad” generator.',
    role: 'simulation',
    random: true,
    notes: SSGAN,
    cite: ['dai2017bad'],
  },
  semi.complementSampler,
)
fn(
  { key: 'initSemiSupervisedGan', name: 'Semi-supervised GAN start', role: 'construction', random: true, notes: SSGAN },
  semi.initSemiSupervisedGan,
)
fn(
  { key: 'generatedPoints', name: 'Generated points as an array', role: 'transform', notes: SSGAN },
  semi.generatedPoints,
)
fn(
  {
    key: 'semiSupervisedGanRun',
    name: 'Streamed semi-supervised GAN run',
    summary: 'Train a semi-supervised GAN from plain data, yielding losses, generated points and probabilities.',
    role: 'simulation',
    random: true,
    notes: SSGAN,
    cite: ['salimans2016', 'dai2017bad'],
  },
  semi.semiSupervisedGanRun,
)

/** The algorithms of the module, keyed by factory name. */
export const ganAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  networks,
  semi,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const ganFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  networks,
  diagnostics,
  run,
  semi,
) as Table<FunctionInfo>
