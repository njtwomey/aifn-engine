/**
 * Autoencoders on small data (2-d point clouds, 5 × 7 digits), with multilayer perceptrons from `aifn-compute/nn` and
 * gradients by autodiff:
 *
 * - **autoencoder**: an encoder to a low-dimensional code and a decoder back, trained on squared reconstruction error;
 * - **variational autoencoder** (VAE; Kingma and Welling, 2014): the encoder outputs a Gaussian q(z|x) = N(μ, diag σ²),
 *   a code is drawn by the reparameterisation z = μ + σ ⊙ ε, and the loss is the negative evidence lower bound,
 *   reconstruction error plus β · KL(q(z|x) ‖ N(0, I)), with β = 1 the ELBO and β > 1 the β-VAE (Higgins et al., 2017);
 * - **conditional VAE** (Sohn, Lee and Yan, 2015): encoder and decoder also see the one-hot label, so the decoder draws
 *   from p(x | z, y) and the code is free to hold what the label does not;
 * - **VQ-VAE** (van den Oord, Vinyals and Kavukcuoglu, 2017): the encoder's output z_e snaps to its nearest codebook
 *   vector e_k, the decoder sees z_q = z_e + stopgrad(e_k − z_e) (the straight-through estimator), and the loss adds
 *   ‖stopgrad(z_e) − e_k‖² (moving the codebook) and β‖z_e − stopgrad(e_k)‖² (the commitment cost).
 *
 * Reconstruction is a Gaussian likelihood with standard deviation σ (squared error / 2σ²) for real data, or Bernoulli
 * (binary cross-entropy on logits) for binary pixels.
 */

import { stopGradient } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, standardNormals, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  exp,
  fromData,
  matmul,
  mean,
  mul,
  slice,
  square,
  sub,
  sum,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { oneHot } from 'aifn-compute/learning/losses'
import { assignNearest } from 'aifn-compute/numerics/neighbours'
import { logSigmoid, sigmoid } from 'aifn-compute/numerics/special'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'

/** The autoencoders of this module. */
export type AutoencoderKind = 'autoencoder' | 'vae' | 'cvae' | 'vqvae'

/** The structure of an autoencoder, plain data. */
export interface AutoencoderSpec {
  readonly kind: AutoencoderKind
  readonly inputs: number
  readonly latent: number
  readonly hidden: readonly number[]
  /** Label classes (the conditional VAE). */
  readonly classes: number
  /** Codebook size (the VQ-VAE). */
  readonly codes: number
  readonly likelihood: 'gaussian' | 'bernoulli'
  /** β: the KL weight of a VAE, or the commitment weight of a VQ-VAE. */
  readonly beta: number
  /** The Gaussian likelihood's standard deviation σ: reconstruction is ‖x − x̂‖²/(2σ²) (default 0.1). */
  readonly observationSd: number
}

/** An autoencoder: its spec and its networks. */
export interface Autoencoder {
  readonly spec: AutoencoderSpec
  readonly encoder: Layer<Params[]>
  readonly decoder: Layer<Params[]>
}

/** An autoencoder's parameters, a pytree. */
export type AutoencoderParams = { encoder: Params[]; decoder: Params[]; codebook: Tensor }

/** Options of `autoencoder`. */
export type AutoencoderOptions = Partial<Omit<AutoencoderSpec, 'inputs'>> & { inputs: number }

/** Build an autoencoder (defaults: a VAE with a 2-d code, hidden [64, 64], Gaussian likelihood, β = 1, 16 codes). */
export function autoencoder(options: AutoencoderOptions): Autoencoder {
  const spec: AutoencoderSpec = {
    kind: options.kind ?? 'vae',
    inputs: options.inputs,
    latent: options.latent ?? 2,
    hidden: options.hidden ?? [64, 64],
    classes: options.classes ?? 0,
    codes: options.codes ?? 16,
    likelihood: options.likelihood ?? 'gaussian',
    beta: options.beta ?? (options.kind === 'vqvae' ? 0.25 : 1),
    observationSd: options.observationSd ?? 0.1,
  }
  const conditional = spec.kind === 'cvae' ? spec.classes : 0
  const out = spec.kind === 'vae' || spec.kind === 'cvae' ? 2 * spec.latent : spec.latent
  return {
    spec,
    encoder: Mlp([spec.inputs + conditional, ...spec.hidden, out], { activation: 'relu' }),
    decoder: Mlp([spec.latent + conditional, ...[...spec.hidden].reverse(), spec.inputs], { activation: 'relu' }),
  }
}

/** Initial parameters: both networks and a codebook drawn N(0, 1) (used by the VQ-VAE only). */
export function initAutoencoder(model: Autoencoder, s: Stream): AutoencoderParams {
  const { codes, latent } = model.spec
  return {
    encoder: model.encoder.init(child(s, 'encoder')),
    decoder: model.decoder.init(child(s, 'decoder')),
    codebook: fromData(standardNormals(child(s, 'codebook'), codes * latent), [codes, latent]),
  }
}

const rowsOf = (v: Value) => (unwrap(v) as Tensor).shape[0]

/** What the encoder says about inputs x [n, D] (and labels for the conditional VAE). */
export function autoencoderEncode(
  model: Autoencoder,
  params: AutoencoderParams,
  x: Value,
  labels?: Tensor,
): { mean: Value; logVariance: Value | null } {
  const { kind, latent } = model.spec
  const input = kind === 'cvae' && labels ? concat([x, labels], 1) : x
  const h = model.encoder.apply(params.encoder, input)
  if (kind === 'vae' || kind === 'cvae')
    return { mean: slice(h, null, [0, latent]), logVariance: slice(h, null, [latent, 2 * latent]) }
  return { mean: h, logVariance: null }
}

/** Decoder outputs at codes z [n, L]: means (Gaussian) or logits (Bernoulli). */
export function decodeRaw(model: Autoencoder, params: AutoencoderParams, z: Value, labels?: Tensor): Value {
  const input = model.spec.kind === 'cvae' && labels ? concat([z, labels], 1) : z
  return model.decoder.apply(params.decoder, input)
}

/** Decoded points or pixel probabilities at codes z [n, L], as a plain tensor. */
export function autoencoderDecode(model: Autoencoder, params: AutoencoderParams, z: Tensor, labels?: Tensor): Tensor {
  const out = decodeRaw(model, params, z, labels)
  return unwrap(model.spec.likelihood === 'bernoulli' ? sigmoid(out) : out) as Tensor
}

/** The loss of a batch and its parts: reconstruction, and the KL (VAE) or codebook + commitment (VQ-VAE) term. */
export function autoencoderLoss(
  model: Autoencoder,
  params: AutoencoderParams,
  x: Tensor,
  s: Stream,
  labels?: Tensor,
): { loss: Value; reconstruction: Value; regulariser: Value } {
  const { kind, likelihood, beta, latent, codes, observationSd } = model.spec
  const n = rowsOf(x)
  const enc = autoencoderEncode(model, params, x, labels)
  let z: Value
  let regulariser: Value = 0
  if (enc.logVariance) {
    const eps = fromData(standardNormals(child(s, 'eps'), n * latent), [n, latent])
    z = add(enc.mean, mul(exp(mul(0.5, enc.logVariance)), eps))
    // KL(N(μ, σ²) ‖ N(0, 1)) = ½ Σ (μ² + σ² − 1 − log σ²), averaged over the batch.
    regulariser = mul(0.5 / n, sum(sub(add(square(enc.mean), exp(enc.logVariance)), add(1, enc.logVariance))))
  } else if (kind === 'vqvae') {
    const ze = enc.mean
    const k = assignNearest(unwrap(stopGradient(ze)) as Tensor, unwrap(stopGradient(params.codebook)) as Tensor).labels
    const eq = matmul(oneHot(k, codes), params.codebook)
    z = add(ze, stopGradient(sub(eq, ze)))
    const codebookTerm = mean(sum(square(sub(stopGradient(ze), eq)), 1))
    const commitment = mean(sum(square(sub(ze, stopGradient(eq))), 1))
    regulariser = add(codebookTerm, mul(beta, commitment))
  } else z = enc.mean
  const out = decodeRaw(model, params, z, labels)
  const reconstruction =
    likelihood === 'bernoulli'
      ? mul(-1 / n, sum(add(mul(x, logSigmoid(out)), mul(sub(1, x), logSigmoid(mul(-1, out))))))
      : mul(0.5 / (n * observationSd * observationSd), sum(square(sub(out, x))))
  const weight = enc.logVariance ? beta : 1
  return { loss: add(reconstruction, mul(weight, regulariser)), reconstruction, regulariser }
}
