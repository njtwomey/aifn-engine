/** The registry of `aifn-methods/generative/autoencoders`. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as m2 from './m2'
import * as models from './models'
import * as run from './run'

const fn = definer<FunctionInfo>('function', 'generative/autoencoders')
const AE = ['autoencoder', 'variational-autoencoder']

fn(
  {
    key: 'autoencoder',
    name: 'Autoencoder family',
    summary: 'MLP encoder and decoder for an autoencoder, a (β-)VAE, a conditional VAE, a VQ-VAE or an RQ-VAE.',
    role: 'construction',
    notes: [
      ...AE,
      'beta-variational-autoencoder',
      'conditional-variational-autoencoder',
      'vector-quantised-variational-autoencoder',
    ],
    cite: ['kingma2014', 'higgins2017', 'sohn2015', 'oord2017', 'lee2022rqvae'],
  },
  models.autoencoder,
)
fn(
  {
    key: 'autoencoderLoss',
    name: 'Autoencoder losses',
    summary:
      'Reconstruction plus β·KL to N(0, I) by the reparameterisation trick (VAE), or codebook and commitment terms with a straight-through code (VQ-VAE; RQ-VAE stage by stage).',
    role: 'property',
    notes: [...AE, 'vector-quantised-variational-autoencoder'],
    cite: ['kingma2014', 'oord2017', 'lee2022rqvae'],
  },
  models.autoencoderLoss,
)
const RQ = { notes: ['vector-quantised-variational-autoencoder', 'additive-and-residual-quantisation'] }
fn(
  {
    key: 'autoencoderQuantise',
    name: 'Discrete codes of an autoencoder',
    summary: 'The encoder’s output quantised stage by stage: one codeword index per stage, coarse to fine.',
    role: 'transform',
    ...RQ,
    cite: ['oord2017', 'lee2022rqvae'],
  },
  models.autoencoderQuantise,
)
fn(
  {
    key: 'latentOfCodes',
    name: 'Latent of a code',
    summary: 'The sum of a code’s codewords over its first stages: the latent its coarse-to-fine prefixes decode.',
    role: 'transform',
    ...RQ,
    cite: ['lee2022rqvae'],
  },
  models.latentOfCodes,
)
fn(
  {
    key: 'initQuantiserCodebook',
    name: 'Codebooks from the data',
    summary: 'Start the codebooks from k-means, or residual k-means, of the encoder’s outputs.',
    role: 'construction',
    random: true,
    ...RQ,
    cite: ['lee2022rqvae', 'chen2010rvq'],
  },
  models.initQuantiserCodebook,
)
fn({ key: 'autoencoderEncode', name: 'Encode', role: 'transform', notes: AE }, models.autoencoderEncode)
fn({ key: 'autoencoderDecode', name: 'Decode', role: 'transform', notes: AE }, models.autoencoderDecode)
fn(
  { key: 'initAutoencoder', name: 'Initial autoencoder parameters', role: 'construction', random: true, notes: AE },
  models.initAutoencoder,
)
fn(
  {
    key: 'autoencoderRun',
    name: 'Streamed autoencoder training',
    summary:
      'Adam on minibatches; loss parts, codes, reconstructions, prior samples and a decoded code grid over training.',
    role: 'simulation',
    random: true,
    notes: AE,
  },
  run.autoencoderRun,
)

// ── Semi-supervised VAE, M2 (Kingma et al. 2014) ─────────────────────────────────────────────────────────────────────

const M2 = ['semi-supervised-variational-autoencoders']
const algorithm = definer<AlgorithmInfo>('algorithm', 'generative/autoencoders')

algorithm(
  {
    key: 'semiSupervisedVaeTraining',
    name: 'Semi-supervised VAE (M2) training',
    summary:
      'A label-plus-code generative model: labelled points by their bound, unlabelled ones by the bound summed over a classifier’s guess, plus a classification loss.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', flags: ['diverged'] },
    random: true,
    notes: M2,
    cite: ['kingma2014semi'],
  },
  m2.semiSupervisedVaeTraining,
)
fn(
  {
    key: 'm2Model',
    name: 'M2 networks',
    summary: 'A conditional VAE for p(x | y, z) and q(z | x, y), and an MLP classifier for q(y | x).',
    role: 'construction',
    notes: M2,
    cite: ['kingma2014semi'],
  },
  m2.m2Model,
)
fn({ key: 'initM2', name: 'Initial M2 parameters', role: 'construction', random: true, notes: M2 }, m2.initM2)
fn(
  {
    key: 'labelledBound',
    name: 'M2 labelled bound',
    tex: '\\mathcal{L}(x, y) = -\\mathbb{E}_{q(z \\mid x, y)}[\\log p(x \\mid y, z)] + \\mathrm{KL}(q(z \\mid x, y) \\| p(z)) - \\log p(y)',
    role: 'property',
    notes: M2,
    cite: ['kingma2014semi'],
  },
  m2.labelledBound,
)
fn(
  {
    key: 'unlabelledBound',
    name: 'M2 unlabelled bound',
    tex: '\\mathcal{U}(x) = \\textstyle\\sum_y q(y \\mid x)\\, \\mathcal{L}(x, y) - \\mathcal{H}[q(y \\mid x)]',
    role: 'property',
    notes: M2,
    cite: ['kingma2014semi'],
  },
  m2.unlabelledBound,
)
fn(
  { key: 'm2ClassProbabilities', name: 'M2 class probabilities', role: 'transform', notes: M2 },
  m2.m2ClassProbabilities,
)
fn({ key: 'm2Decode', name: 'M2 decoder means', role: 'transform', notes: M2 }, m2.m2Decode)
fn(
  {
    key: 'semiSupervisedVaeRun',
    name: 'Streamed M2 training',
    summary: 'Train M2 or its classifier alone from plain data, yielding bounds, probabilities and decoded curves.',
    role: 'simulation',
    random: true,
    notes: M2,
    cite: ['kingma2014semi'],
  },
  m2.semiSupervisedVaeRun,
)

/** The functions of the module, keyed by name. */
export const autoencoderFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', models, run, m2) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
/** The algorithms of the module, keyed by factory name. */
export const autoencoderAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', m2) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
