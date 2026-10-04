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
    summary: 'MLP encoder and decoder for an autoencoder, a (β-)VAE, a conditional VAE or a VQ-VAE.',
    role: 'construction',
    notes: [
      ...AE,
      'beta-variational-autoencoder',
      'conditional-variational-autoencoder',
      'vector-quantised-variational-autoencoder',
    ],
    cite: ['kingma2014', 'higgins2017', 'sohn2015', 'oord2017'],
  },
  models.autoencoder,
)
fn(
  {
    key: 'autoencoderLoss',
    name: 'Autoencoder losses',
    summary:
      'Reconstruction plus β·KL to N(0, I) by the reparameterisation trick (VAE), or codebook and commitment terms with a straight-through code (VQ-VAE).',
    role: 'property',
    notes: [...AE, 'vector-quantised-variational-autoencoder'],
    cite: ['kingma2014', 'oord2017'],
  },
  models.autoencoderLoss,
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
