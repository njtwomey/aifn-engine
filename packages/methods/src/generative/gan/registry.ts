/**
 * The registry of `aifn-methods/generative/gan`: GAN training as a traceable algorithm, and the networks, the
 * diagnostics against a known density and the streamed run as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as diagnostics from './diagnostics'
import * as networks from './gan'
import * as run from './run'

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

/** The algorithms of the module, keyed by factory name. */
export const ganAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', networks) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const ganFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  networks,
  diagnostics,
  run,
) as Table<FunctionInfo>
