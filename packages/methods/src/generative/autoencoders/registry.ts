/** The registry of `aifn-methods/generative/autoencoders`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
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

/** The functions of the module, keyed by name. */
export const autoencoderFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', models, run) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
