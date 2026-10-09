/**
 * Autoencoders on small data (2-d point clouds, $5 \times 7$ digits), with multilayer perceptrons from `aifn-compute/nn`
 * and gradients by autodiff. Inputs are rows $\xvec \in \reals^p$; codes are $\zvec \in \reals^L$.
 *
 * - **autoencoder**: an encoder to a low-dimensional code and a decoder back, trained on squared reconstruction error;
 * - **variational autoencoder** (VAE; Kingma and Welling, 2014): the encoder outputs a Gaussian
 *   $q(\zvec \mid \xvec) = \Gauss(\muvec, \operatorname{diag} \sigmavec^2)$, a code is drawn by the reparameterisation
 *   $\zvec = \muvec + \sigmavec \odot \epsilonvec$, and the loss is the negative evidence lower bound, reconstruction
 *   error plus $\beta \, \mathrm{KL}(q(\zvec \mid \xvec) \,\|\, \Gauss(\mathbf{0}, \Imat))$, with $\beta = 1$ the ELBO and
 *   $\beta > 1$ the $\beta$-VAE (Higgins et al., 2017);
 * - **conditional VAE** (Sohn, Lee and Yan, 2015): encoder and decoder also see the one-hot label, so the decoder draws
 *   from $p(\xvec \mid \zvec, y)$ and the code is free to hold what the label does not;
 * - **VQ-VAE** (van den Oord, Vinyals and Kavukcuoglu, 2017): the encoder's output $\zvec_e$ snaps to its nearest
 *   codeword $\evec_k$, the decoder sees $\zvec_q = \zvec_e + \mathrm{sg}(\evec_k - \zvec_e)$ (the straight-through
 *   estimator, $\mathrm{sg}$ stopping the gradient), and the loss adds $\lVert \mathrm{sg}(\zvec_e) - \evec_k \rVert^2$
 *   (moving the codebook) and $\beta \lVert \zvec_e - \mathrm{sg}(\evec_k) \rVert^2$ (the commitment cost).
 * - **RQ-VAE** (Lee, Kim, Kim, Cho and Han, 2022): the encoder's output is quantised by $D$ residual stages instead of
 *   one (`aifn-compute/numerics/neighbours`' residual quantisation): $\rvec_0 = \zvec_e$, $\evec_d$ is the codeword
 *   nearest to $\rvec_{d-1}$ and $\rvec_d = \rvec_{d-1} - \evec_d$, so the code is $D$ indices and its
 *   quantisation $\hat\zvec = \sum_d \evec_d$ is refined coarse to fine. Each stage has its own codebook, as
 *   SoundStream's residual quantiser and TIGER's semantic IDs do (the default), or all share one, as Lee et al. do
 *   with thousands of codewords (`sharedCodebook: true`). The decoder sees $\zvec_e + \mathrm{sg}(\hat\zvec - \zvec_e)$; the loss adds the codebook term
 *   $\sum_d \lVert \mathrm{sg}(\rvec_{d-1}) - \evec_d \rVert^2$ and Lee et al.'s commitment to every partial sum,
 *   $\beta \sum_d \lVert \zvec_e - \mathrm{sg}(\hat\zvec^{(d)}) \rVert^2$, with
 *   $\hat\zvec^{(d)} = \sum_{i \le d} \evec_i$.
 *
 * Reconstruction is a Gaussian likelihood with standard deviation $\sigma$ (squared error over $2\sigma^2$) for real
 * data, or Bernoulli (binary cross-entropy on logits) for binary pixels.
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
  reshape,
  slice,
  square,
  sub,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { oneHot } from 'aifn-compute/learning/losses'
import { assignNearest, residualQuantiser, trainCodebook } from 'aifn-compute/numerics/neighbours'
import { logSigmoid, sigmoid } from 'aifn-compute/numerics/special'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'

/** The autoencoders of this module. */
export type AutoencoderKind = 'autoencoder' | 'vae' | 'cvae' | 'vqvae' | 'rqvae'

/** The structure of an autoencoder, plain data. */
export interface AutoencoderSpec {
  /** Which autoencoder. */
  readonly kind: AutoencoderKind
  /** The input width $p$. */
  readonly inputs: number
  /** The code width $L$. */
  readonly latent: number
  /** The hidden layers' widths of the encoder (the decoder mirrors them). */
  readonly hidden: readonly number[]
  /** Label classes (the conditional VAE). */
  readonly classes: number
  /** Codebook size: codewords per codebook (the VQ-VAE and the RQ-VAE). */
  readonly codes: number
  /** Quantisation stages $D$ (the RQ-VAE; 1 for every other kind). */
  readonly depth: number
  /**
   * The RQ-VAE: one codebook shared by every stage (true, as Lee et al.) or one per stage (false, the default). A shared
   * codebook must serve residuals of every scale, which needs many codewords; small codebooks work far better one per
   * stage.
   */
  readonly sharedCodebook: boolean
  /** The reconstruction's likelihood: Gaussian for real data, Bernoulli for binary pixels. */
  readonly likelihood: 'gaussian' | 'bernoulli'
  /** $\beta$: the KL weight of a VAE, or the commitment weight of a VQ-VAE or an RQ-VAE. */
  readonly beta: number
  /**
   * The Gaussian likelihood's standard deviation $\sigma$: reconstruction is
   * $\lVert \xvec - \hat\xvec \rVert^2 / (2\sigma^2)$ (default 0.1).
   */
  readonly observationSd: number
}

/** An autoencoder: its spec and its networks. */
export interface Autoencoder {
  /** Its structure. */
  readonly spec: AutoencoderSpec
  /** The encoder, inputs to codes (to means and log-variances for a VAE). */
  readonly encoder: Layer<Params[]>
  /** The decoder, codes to outputs (means, or logits for Bernoulli pixels). */
  readonly decoder: Layer<Params[]>
}

/**
 * An autoencoder's parameters, a pytree. The codebook is $[B K, L]$ for $B$ codebooks of $K$ codewords of the latent
 * width $L$: codebook $b$ is rows $bK$ to $(b + 1)K - 1$ ($B = 1$ but for an RQ-VAE with one codebook per stage).
 */
export type AutoencoderParams = { encoder: Params[]; decoder: Params[]; codebook: Tensor }

/** Options of `autoencoder`. */
export type AutoencoderOptions = Partial<Omit<AutoencoderSpec, 'inputs'>> & { inputs: number }

/**
 * Build an autoencoder of any kind: an encoder and a mirrored decoder, multilayer perceptrons with ReLU activations.
 * The defaults are a VAE with a 2-d code, hidden widths $[64, 64]$, a Gaussian likelihood and $\beta = 1$; a VQ-VAE or
 * RQ-VAE has 16 codewords per codebook and $\beta = 0.25$, and an RQ-VAE 4 stages with a codebook each.
 *
 * @param options The input width $p$ (required) and any field of `AutoencoderSpec` to change from the defaults.
 * @returns The model: its spec and its two networks, without parameters (`initAutoencoder` draws them).
 *
 * @example An RQ-VAE with three stages of eight codewords each
 * const model = autoencoder({ kind: 'rqvae', inputs: 2, latent: 2, codes: 8, depth: 3 })
 * print(model.spec)
 */
export function autoencoder(options: AutoencoderOptions): Autoencoder {
  const kind = options.kind ?? 'vae'
  const quantised = kind === 'vqvae' || kind === 'rqvae'
  const spec: AutoencoderSpec = {
    kind,
    inputs: options.inputs,
    latent: options.latent ?? 2,
    hidden: options.hidden ?? [64, 64],
    classes: options.classes ?? 0,
    codes: options.codes ?? 16,
    depth: kind === 'rqvae' ? (options.depth ?? 4) : 1,
    sharedCodebook: kind === 'rqvae' ? (options.sharedCodebook ?? false) : true,
    likelihood: options.likelihood ?? 'gaussian',
    beta: options.beta ?? (quantised ? 0.25 : 1),
    observationSd: options.observationSd ?? 0.1,
  }
  if (!(Number.isInteger(spec.depth) && spec.depth >= 1))
    throw new DomainError('autoencoder', `autoencoder: depth must be a positive integer, got ${spec.depth}`)
  const conditional = spec.kind === 'cvae' ? spec.classes : 0
  const out = spec.kind === 'vae' || spec.kind === 'cvae' ? 2 * spec.latent : spec.latent
  return {
    spec,
    encoder: Mlp([spec.inputs + conditional, ...spec.hidden, out], { activation: 'relu' }),
    decoder: Mlp([spec.latent + conditional, ...[...spec.hidden].reverse(), spec.inputs], { activation: 'relu' }),
  }
}

/**
 * The number of codebooks $B$: the stages of an RQ-VAE with one codebook per stage, else 1.
 *
 * @param spec The autoencoder's structure.
 * @returns $B$.
 */
const codebooksOf = (spec: AutoencoderSpec) => (spec.kind === 'rqvae' && !spec.sharedCodebook ? spec.depth : 1)

/**
 * Initial parameters: both networks, and codebooks with entries drawn from $\Gauss(0, 1)$ (used by the VQ-VAE and the
 * RQ-VAE only; `initQuantiserCodebook` starts them from the data instead).
 *
 * @param model The autoencoder.
 * @param s The stream: the encoder, decoder and codebook draw from its children `'encoder'`, `'decoder'`, `'codebook'`.
 * @returns The parameters, a pytree.
 *
 * @example The shapes of an RQ-VAE's parameters
 * const model = autoencoder({ kind: 'rqvae', inputs: 2, latent: 2, codes: 8, depth: 3 })
 * const params = initAutoencoder(model, stream(1))
 * print('codebook shape:', params.codebook.shape, ' encoder layers:', params.encoder.length)
 */
export function initAutoencoder(model: Autoencoder, s: Stream): AutoencoderParams {
  const { codes, latent } = model.spec
  const rows = codebooksOf(model.spec) * codes
  return {
    encoder: model.encoder.init(child(s, 'encoder')),
    decoder: model.decoder.init(child(s, 'decoder')),
    codebook: fromData(standardNormals(child(s, 'codebook'), rows * latent), [rows, latent]),
  }
}

/**
 * Start the codebooks of a VQ-VAE or an RQ-VAE from the data: from $k$-means of the encoder's outputs on `x` (a VQ-VAE,
 * or an RQ-VAE with a shared codebook), or from a residual quantiser trained on them, one codebook per stage
 * (`residualQuantiser`). A data-dependent start keeps codewords from going unused, as $k$-means starts do in
 * SoundStream and the RQ-VAE. Other kinds are returned unchanged.
 *
 * @param model The autoencoder.
 * @param params Its parameters, whose encoder is used as it is.
 * @param x The training inputs $[n, p]$, at least `codes` rows.
 * @param s The stream of the $k$-means seeding.
 * @returns The parameters with the codebook replaced.
 *
 * @example The first stage's codebook starts on the data's codes
 * const s = stream(1)
 * const x = normal(s, 0, 1, { shape: [200, 2] })
 * const model = autoencoder({ kind: 'rqvae', inputs: 2, latent: 2, codes: 4, depth: 2 })
 * const params = initQuantiserCodebook(model, initAutoencoder(model, s), x, s)
 * print('codebooks, stage by stage:', params.codebook)
 */
export function initQuantiserCodebook(
  model: Autoencoder,
  params: AutoencoderParams,
  x: Tensor,
  s: Stream,
): AutoencoderParams {
  const { kind, codes, depth } = model.spec
  if (kind !== 'vqvae' && kind !== 'rqvae') return params
  const ze = unwrap(autoencoderEncode(model, params, x).mean) as Tensor
  const codebook =
    codebooksOf(model.spec) > 1
      ? (() => {
          const rq = residualQuantiser(ze, { levels: depth, codewords: codes, stream: child(s, 'codebook') })
          return reshape(rq.codebooks, [depth * codes, rq.d])
        })()
      : trainCodebook(ze, codes, { stream: child(s, 'codebook') }).centroids
  return { ...params, codebook }
}

/**
 * The number of rows of a batch.
 *
 * @param v The batch, $[n, \cdot]$.
 * @returns $n$.
 */
const rowsOf = (v: Value) => (unwrap(v) as Tensor).shape[0]

/**
 * What the encoder says about inputs: the code (an autoencoder), the encoder's output $\zvec_e$ before quantisation
 * (a VQ-VAE or RQ-VAE), or the mean and log-variance of $q(\zvec \mid \xvec)$ (a VAE).
 *
 * @param model The autoencoder.
 * @param params Its parameters.
 * @param x The inputs $[n, p]$ (traced inside a gradient).
 * @param labels One-hot labels $[n, C]$, which a conditional VAE's encoder also reads; ignored by the other kinds.
 * @returns `mean`, the codes $[n, L]$, and `logVariance` $[n, L]$ for a VAE (null otherwise).
 *
 * @example The codes of three points under an untrained autoencoder
 * const model = autoencoder({ kind: 'autoencoder', inputs: 2, latent: 2 })
 * const params = initAutoencoder(model, stream(1))
 * print(autoencoderEncode(model, params, tensor([[0, 0], [1, 0], [0, 1]])).mean)
 */
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

/**
 * The decoder's raw outputs at codes: means (Gaussian) or logits (Bernoulli).
 *
 * @param model The autoencoder.
 * @param params Its parameters.
 * @param z The codes $[n, L]$ (traced inside a gradient).
 * @param labels One-hot labels $[n, C]$ for a conditional VAE; ignored by the other kinds.
 * @returns The outputs $[n, p]$.
 */
export function decodeRaw(model: Autoencoder, params: AutoencoderParams, z: Value, labels?: Tensor): Value {
  const input = model.spec.kind === 'cvae' && labels ? concat([z, labels], 1) : z
  return model.decoder.apply(params.decoder, input)
}

/**
 * Decoded points or pixel probabilities at codes, as a plain tensor: the decoder's means, or the sigmoid of its logits
 * for Bernoulli pixels.
 *
 * @param model The autoencoder.
 * @param params Its parameters.
 * @param z The codes $[n, L]$; for a VQ-VAE or RQ-VAE, quantised latents such as `latentOfCodes` gives.
 * @param labels One-hot labels $[n, C]$ for a conditional VAE; ignored by the other kinds.
 * @returns The outputs $[n, p]$.
 *
 * @example Decode a few codes of an untrained model
 * const model = autoencoder({ kind: 'autoencoder', inputs: 3, latent: 2 })
 * print(autoencoderDecode(model, initAutoencoder(model, stream(1)), tensor([[0, 0], [1, -1]])))
 */
export function autoencoderDecode(model: Autoencoder, params: AutoencoderParams, z: Tensor, labels?: Tensor): Tensor {
  const out = decodeRaw(model, params, z, labels)
  return unwrap(model.spec.likelihood === 'bernoulli' ? sigmoid(out) : out) as Tensor
}

/**
 * The codebook of stage $d$: the shared codebook, or the $d$th of an RQ-VAE's per-stage codebooks.
 *
 * @param spec The autoencoder's structure.
 * @param codebook All the codebooks, $[B K, L]$.
 * @param d The stage, from 0.
 * @returns The stage's codebook, $[K, L]$.
 */
function stageCodebook(spec: AutoencoderSpec, codebook: Value, d: number): Value {
  if (codebooksOf(spec) === 1) return codebook
  return slice(codebook, [d * spec.codes, (d + 1) * spec.codes])
}

/** What residual quantisation of a batch of codes gives, with gradients flowing to the codebooks. */
type Quantisation = {
  /** The codes $[n, D]$ (int32): the codeword chosen at each stage. */
  codes: Tensor
  /** The quantisation $\hat\zvec = \sum_d \evec_d$ (differentiable in the codebooks). */
  quantised: Value
  /** The partial sums $\hat\zvec^{(d)}$ of the first $d$ stages, $d = 1, \dots, D$. */
  partials: Value[]
  /** The codewords $\evec_d$ chosen at each stage, and the residuals $\rvec_{d-1}$ they quantised (constants). */
  stages: { codeword: Value; residual: Tensor }[]
}

/**
 * Residual quantisation of codes $\zvec$ $[n, L]$ through the model's stages: the nearest codeword of each stage to what
 * the stages before left. The assignments are made on constants; the codewords keep their gradients.
 *
 * @param spec The autoencoder's structure.
 * @param codebook All the codebooks, $[B K, L]$.
 * @param z The codes to quantise, $[n, L]$ (read as constants).
 * @returns The codes, the quantisation, its partial sums and each stage's codeword and residual.
 */
function residualQuantise(spec: AutoencoderSpec, codebook: Value, z: Value): Quantisation {
  const { codes: K, depth } = spec
  let residual = unwrap(stopGradient(z)) as Tensor
  const n = residual.shape[0]
  const L = residual.shape[1]
  const indices = new Int32Array(n * depth)
  let quantised: Value = 0
  const partials: Value[] = []
  const stages: { codeword: Value; residual: Tensor }[] = []
  for (let d = 0; d < depth; d++) {
    const book = stageCodebook(spec, codebook, d)
    const k = assignNearest(residual, unwrap(stopGradient(book)) as Tensor).labels
    const kk = toFlat(k)
    for (let i = 0; i < n; i++) indices[i * depth + d] = kk[i]
    const codeword = matmul(oneHot(k, K), book)
    stages.push({ codeword, residual })
    quantised = add(quantised, codeword)
    partials.push(quantised)
    const used = toFlat(unwrap(stopGradient(codeword)) as Tensor)
    const r = toFlat(residual)
    residual = fromData(
      Float64Array.from(r, (v, q) => v - used[q]),
      [n, L],
    )
  }
  return { codes: fromData(indices, [n, depth]), quantised, partials, stages }
}

/**
 * The discrete codes of inputs under a VQ-VAE ($D = 1$) or an RQ-VAE: the encoder's output quantised stage by stage,
 * with the quantised latent and its partial sums. An RQ-VAE's code is its inputs' semantic ID: rows that share the
 * first $d$ indices share the coarse latent $\hat\zvec^{(d)}$, so the codes form a tree of $K$-way refinements.
 *
 * @param model A VQ-VAE or an RQ-VAE.
 * @param params Its parameters.
 * @param x The inputs $[n, D]$.
 * @returns The codes $[n, D]$ (int32), the quantised latent $\hat\zvec$ $[n, L]$ and the partial sums
 *   $\hat\zvec^{(1)}, \dots, \hat\zvec^{(D)}$ (each $[n, L]$).
 *
 * @example Semantic IDs of a few points
 * const s = stream(2)
 * const x = normal(s, 0, 1, { shape: [100, 2] })
 * const model = autoencoder({ kind: 'rqvae', inputs: 2, latent: 2, codes: 4, depth: 3 })
 * const params = initQuantiserCodebook(model, initAutoencoder(model, s), x, s)
 * print('codes of the first three rows:', slice(autoencoderQuantise(model, params, x).codes, [0, 3]))
 */
export function autoencoderQuantise(
  model: Autoencoder,
  params: AutoencoderParams,
  x: Tensor,
): { codes: Tensor; quantised: Tensor; partials: Tensor[] } {
  const { kind } = model.spec
  if (kind !== 'vqvae' && kind !== 'rqvae')
    throw new DomainError('autoencoderQuantise', `autoencoderQuantise: a ${kind} has no codebook`)
  const q = residualQuantise(model.spec, unwrap(params.codebook), autoencoderEncode(model, params, x).mean)
  return {
    codes: q.codes,
    quantised: unwrap(q.quantised) as Tensor,
    partials: q.partials.map((p) => unwrap(p) as Tensor),
  }
}

/**
 * The quantised latent of codes: the sum of each row's codewords over its first `levels` stages, the coarse-to-fine
 * latent that `autoencoderDecode` turns into an output.
 *
 * @param model A VQ-VAE or an RQ-VAE.
 * @param params Its parameters.
 * @param codes The codes $[n, D]$, as `autoencoderQuantise` gives them.
 * @param levels How many leading stages to sum, from 0 to $D$ (default $D$).
 * @returns The latent $[n, L]$.
 *
 * @example A point's latent, refined stage by stage towards the encoder's output
 * const s = stream(3)
 * const x = normal(s, 0, 1, { shape: [200, 2] })
 * const model = autoencoder({ kind: 'rqvae', inputs: 2, latent: 2, codes: 4, depth: 3 })
 * const params = initQuantiserCodebook(model, initAutoencoder(model, s), x, s)
 * const point = slice(x, [0, 1])
 * const { codes } = autoencoderQuantise(model, params, point)
 * print('encoder output:', autoencoderEncode(model, params, point).mean, ' code:', codes)
 * for (const levels of [1, 2, 3]) print(`${levels} stage(s):`, latentOfCodes(model, params, codes, levels))
 */
export function latentOfCodes(
  model: Autoencoder,
  params: AutoencoderParams,
  codes: Tensor | readonly (readonly number[])[],
  levels?: number,
): Tensor {
  const { codes: K, depth, latent: L } = model.spec
  const D = levels ?? depth
  const rows = Array.isArray(codes) ? (codes as readonly (readonly number[])[]) : null
  const flat = rows ? rows.flat() : Array.from(toFlat(codes as Tensor))
  const n = rows ? rows.length : (codes as Tensor).shape[0]
  if (flat.length !== n * depth) throw new DomainError('latentOfCodes', `latentOfCodes: codes must be [n, ${depth}]`)
  if (!(Number.isInteger(D) && D >= 0 && D <= depth))
    throw new DomainError('latentOfCodes', `latentOfCodes: levels must be an integer from 0 to ${depth}`)
  const C = toFlat(unwrap(params.codebook) as Tensor)
  const shared = codebooksOf(model.spec) === 1
  const out = new Float64Array(n * L)
  for (let i = 0; i < n; i++)
    for (let d = 0; d < D; d++) {
      const k = flat[i * depth + d]
      if (!(Number.isInteger(k) && k >= 0 && k < K))
        throw new DomainError('latentOfCodes', `latentOfCodes: code ${k} is not a codeword 0 … ${K - 1}`)
      const row = (shared ? 0 : d * K) + k
      for (let j = 0; j < L; j++) out[i * L + j] += C[row * L + j]
    }
  return fromData(out, [n, L])
}

/**
 * The training loss of a batch and its parts: the reconstruction term, and the regulariser, which is the KL divergence
 * to the prior (a VAE, weighted by $\beta$), or the codebook and commitment terms (a VQ-VAE; an RQ-VAE stage by stage).
 * Differentiable in the parameters through `grad`.
 *
 * @param model The autoencoder.
 * @param params Its parameters.
 * @param x The batch $[n, p]$.
 * @param s The stream of a VAE's reparameterisation noise (child `'eps'`); unused by the other kinds.
 * @param labels One-hot labels $[n, C]$ for a conditional VAE; ignored by the other kinds.
 * @returns `loss`, the total, with `reconstruction` and `regulariser`.
 *
 * @example The RQ-VAE's loss and its parts at the start
 * const s = stream(4)
 * const x = normal(s, 0, 1, { shape: [64, 2] })
 * const model = autoencoder({ kind: 'rqvae', inputs: 2, latent: 2, codes: 8, depth: 3, observationSd: 1 })
 * const params = initQuantiserCodebook(model, initAutoencoder(model, s), x, s)
 * const parts = autoencoderLoss(model, params, x, s)
 * print('loss', parts.loss, ' reconstruction', parts.reconstruction, ' codebook + commitment', parts.regulariser)
 */
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
  } else if (kind === 'rqvae') {
    const ze = enc.mean
    const q = residualQuantise(model.spec, params.codebook, ze)
    z = add(ze, stopGradient(sub(q.quantised, ze)))
    let codebookTerm: Value = 0
    let commitment: Value = 0
    q.stages.forEach((stage, d) => {
      codebookTerm = add(codebookTerm, mean(sum(square(sub(stage.residual, stage.codeword)), 1)))
      commitment = add(commitment, mean(sum(square(sub(ze, stopGradient(q.partials[d]))), 1)))
    })
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
