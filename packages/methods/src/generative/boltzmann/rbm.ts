/**
 * The Bernoulli restricted Boltzmann machine (Smolensky, 1986; Hinton, 2002): binary visible units v ∈ {0, 1}ᴰ and
 * hidden units h ∈ {0, 1}ᴴ with energy E(v, h) = −aᵀv − bᵀh − vᵀWh, so the conditionals factorise,
 * p(hⱼ = 1 | v) = σ(bⱼ + vᵀW_{·j}) and p(vᵢ = 1 | h) = σ(aᵢ + W_{i·}h), and the free energy is
 * F(v) = −aᵀv − Σⱼ log(1 + exp(bⱼ + vᵀW_{·j})).
 *
 * The log-likelihood gradient is ⟨vhᵀ⟩_data − ⟨vhᵀ⟩_model. Contrastive divergence (CD-k; Hinton, 2002) estimates the
 * model term by k steps of block Gibbs sampling started at the data; persistent CD (PCD; Tieleman, 2008) keeps the
 * chains between updates. For a small hidden layer the partition function Z = Σ_h exp(bᵀh) Πᵢ (1 + exp(aᵢ + W_{i·}h))
 * is summed exactly over the 2ᴴ hidden states, which gives the exact log-likelihood of the data to track training.
 */

import { child, stream, units, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
const softplus = (x: number) => (x > 30 ? x : Math.log1p(Math.exp(x)))

/** An RBM's parameters: visible biases a [D], hidden biases b [H], weights W [D × H] row-major. */
export interface Rbm {
  readonly visible: number
  readonly hidden: number
  readonly a: Float64Array
  readonly b: Float64Array
  readonly W: Float64Array
}

/** A new RBM with random weights N(0, 0.1²) and zero biases (large enough to leave the symmetric start quickly). */
export function rbm(s: Stream, visible: number, hidden: number): Rbm {
  const u = units(child(s, 'weights'), 2 * visible * hidden)
  const W = new Float64Array(visible * hidden)
  for (let k = 0; k < W.length; k++)
    W[k] = 0.1 * Math.sqrt(-2 * Math.log(1 - u[2 * k])) * Math.cos(2 * Math.PI * u[2 * k + 1])
  return { visible, hidden, a: new Float64Array(visible), b: new Float64Array(hidden), W }
}

/** p(h = 1 | v) for one visible vector. */
export function hiddenProbabilities(m: Rbm, v: ArrayLike<number>): Float64Array {
  const out = Float64Array.from(m.b)
  for (let i = 0; i < m.visible; i++)
    if (v[i]) for (let j = 0; j < m.hidden; j++) out[j] += v[i] * m.W[i * m.hidden + j]
  return out.map(sigmoid)
}

/** p(v = 1 | h) for one hidden vector. */
export function visibleProbabilities(m: Rbm, h: ArrayLike<number>): Float64Array {
  const out = Float64Array.from(m.a)
  for (let i = 0; i < m.visible; i++) {
    let s = 0
    for (let j = 0; j < m.hidden; j++) s += m.W[i * m.hidden + j] * h[j]
    out[i] += s
  }
  return out.map(sigmoid)
}

/** The free energy F(v) = −aᵀv − Σⱼ softplus(bⱼ + vᵀW_{·j}). */
export function freeEnergy(m: Rbm, v: ArrayLike<number>): number {
  let f = 0
  for (let i = 0; i < m.visible; i++) f -= m.a[i] * v[i]
  for (let j = 0; j < m.hidden; j++) {
    let x = m.b[j]
    for (let i = 0; i < m.visible; i++) x += v[i] * m.W[i * m.hidden + j]
    f -= softplus(x)
  }
  return f
}

/** log Z by summing over the 2ᴴ hidden states (H ≤ 20). */
export function logPartition(m: Rbm): number {
  if (m.hidden > 20) throw new DomainError('logPartition', 'logPartition: exact only for at most 20 hidden units')
  const terms: number[] = []
  const h = new Float64Array(m.hidden)
  for (let mask = 0; mask < 1 << m.hidden; mask++) {
    let t = 0
    for (let j = 0; j < m.hidden; j++) {
      h[j] = (mask >> j) & 1
      t += m.b[j] * h[j]
    }
    for (let i = 0; i < m.visible; i++) {
      let x = m.a[i]
      for (let j = 0; j < m.hidden; j++) x += m.W[i * m.hidden + j] * h[j]
      t += softplus(x)
    }
    terms.push(t)
  }
  // A loop, not Math.max(...terms): 2¹⁶ arguments overflow the stack in a worker.
  const top = terms.reduce((a, t) => (t > a ? t : a), -Infinity)
  return top + Math.log(terms.reduce((s, t) => s + Math.exp(t - top), 0))
}

/** The mean exact log-likelihood of rows of v [n, D]: −F(v) − log Z. */
export function rbmLogLikelihood(m: Rbm, v: MatrixLike): number {
  const V = dense.toMatrixF64(v, 'rbmLogLikelihood')
  const logZ = logPartition(m)
  let s = 0
  for (let r = 0; r < V.m; r++) s += -freeEnergy(m, V.data.subarray(r * V.n, (r + 1) * V.n)) - logZ
  return s / V.m
}

/** Sample 0/1 units from their probabilities with uniforms. */
const bernoulliRow = (p: ArrayLike<number>, u: ArrayLike<number>) => Float64Array.from(p, (q, i) => (u[i] < q ? 1 : 0))

/** k steps of block Gibbs sampling v → h → v from `v`, from the stream; returns the final v and p(h | v). */
export function gibbsChain(m: Rbm, v: ArrayLike<number>, k: number, s: Stream): { v: Float64Array; h: Float64Array } {
  let vis = Float64Array.from(v)
  for (let t = 0; t < k; t++) {
    const h = bernoulliRow(hiddenProbabilities(m, vis), units(child(s, 'h', t), m.hidden))
    vis = bernoulliRow(visibleProbabilities(m, h), units(child(s, 'v', t), m.visible))
  }
  return { v: vis, h: hiddenProbabilities(m, vis) }
}

/** Options of `contrastiveDivergenceStep`. */
export interface CdOptions {
  /** Gibbs steps k (default 1). */
  k?: number
  /** Step size (default 0.05). */
  learningRate?: number
  /** Persistent chains (PCD): pass the chains' visible states; they are advanced and returned. */
  chains?: Float64Array[]
}

/**
 * One CD-k (or PCD-k) update on a minibatch of rows v [n, D]: W += η (⟨v p(h|v)ᵀ⟩_data − ⟨v′ p(h|v′)ᵀ⟩_chains), and
 * likewise the biases. Returns the new RBM, the advanced chains and the reconstruction error of the data's one-step
 * reconstructions.
 */
export function contrastiveDivergenceStep(
  m: Rbm,
  v: MatrixLike,
  s: Stream,
  options: CdOptions = {},
): { rbm: Rbm; chains: Float64Array[]; reconstructionError: number } {
  const { k = 1, learningRate = 0.05 } = options
  const V = dense.toMatrixF64(v, 'contrastiveDivergenceStep')
  const n = V.m
  const { visible: D, hidden: H } = m
  const dW = new Float64Array(D * H)
  const da = new Float64Array(D)
  const db = new Float64Array(H)
  const starts = options.chains ?? Array.from({ length: n }, (_, r) => V.data.slice(r * D, (r + 1) * D))
  const chains: Float64Array[] = []
  let err = 0
  for (let r = 0; r < n; r++) {
    const x = V.data.subarray(r * D, (r + 1) * D)
    const ph = hiddenProbabilities(m, x)
    for (let i = 0; i < D; i++) {
      da[i] += x[i] / n
      for (let j = 0; j < H; j++) dW[i * H + j] += (x[i] * ph[j]) / n
    }
    for (let j = 0; j < H; j++) db[j] += ph[j] / n
    const recon = visibleProbabilities(m, ph)
    for (let i = 0; i < D; i++) err += (recon[i] - x[i]) ** 2 / n
  }
  for (let c = 0; c < starts.length; c++) {
    const { v: vk, h: hk } = gibbsChain(m, starts[c], k, child(s, 'chain', c))
    chains.push(vk)
    const w = 1 / starts.length
    for (let i = 0; i < D; i++) {
      da[i] -= vk[i] * w
      for (let j = 0; j < H; j++) dW[i * H + j] -= vk[i] * hk[j] * w
    }
    for (let j = 0; j < H; j++) db[j] -= hk[j] * w
  }
  return {
    rbm: {
      ...m,
      a: Float64Array.from(m.a, (x, i) => x + learningRate * da[i]),
      b: Float64Array.from(m.b, (x, j) => x + learningRate * db[j]),
      W: Float64Array.from(m.W, (x, q) => x + learningRate * dW[q]),
    },
    chains,
    reconstructionError: err,
  }
}

/** Options of `rbmRun`. */
export interface RbmRunOptions {
  hidden?: number
  k?: number
  persistent?: boolean
  learningRate?: number
  epochs?: number
  batchSize?: number
  /** Gibbs steps between the samples shown at checkpoints (default 200), and how many samples (default 16). */
  sampleSteps?: number
  samples?: number
  seed?: number | string
}

/** One checkpoint of an RBM run. */
export interface RbmCheckpoint {
  epoch: number
  /** The weights as D × H, and the samples (each a D-vector) from long Gibbs chains. */
  weights: Float64Array
  samples: Float64Array
}

/** An RBM run: the exact log-likelihood (when H ≤ 16) and reconstruction error per epoch, and checkpoints. */
export interface RbmRun {
  visible: number
  hidden: number
  logLikelihood: number[]
  reconstructionError: number[]
  /** log of the number of distinct training patterns: the log-likelihood of their uniform distribution. */
  bestLogLikelihood: number
  checkpoints: RbmCheckpoint[]
  done: boolean
}

/** Train an RBM by CD-k or PCD-k on binary rows x [n, D], yielding after every epoch. Deterministic in `seed`. */
export function* rbmRun(data: { x: MatrixLike } | MatrixLike, options: RbmRunOptions = {}): Generator<RbmRun, RbmRun> {
  const x = 'x' in (data as object) ? (data as { x: MatrixLike }).x : (data as MatrixLike)
  const { hidden = 8, k = 1, persistent = false, learningRate = 0.1, epochs = 500, batchSize = 4 } = options
  const { sampleSteps = 200, samples = 16, seed = 0 } = options
  const V = dense.toMatrixF64(x, 'rbmRun')
  const { m: n, n: D } = V
  const s0 = (name: string) => child(stream(seed), name)
  let model = rbm(s0('init'), D, hidden)
  let chains: Float64Array[] | undefined = persistent
    ? Array.from({ length: Math.min(batchSize, n) }, (_, c) => V.data.slice((c % n) * D, ((c % n) + 1) * D))
    : undefined
  const distinct = new Set<string>()
  for (let r = 0; r < n; r++) distinct.add(Array.from(V.data.subarray(r * D, (r + 1) * D)).join(''))
  const run: RbmRun = {
    visible: D,
    hidden,
    logLikelihood: [],
    reconstructionError: [],
    bestLogLikelihood: -Math.log(distinct.size),
    checkpoints: [],
    done: false,
  }
  const exact = hidden <= 16
  const snapshotSamples = (e: number) => {
    const out = new Float64Array(samples * D)
    for (let c = 0; c < samples; c++) {
      const u = units(child(s0('samples'), e, c), D)
      const start = Float64Array.from(u, (q) => (q < 0.5 ? 1 : 0))
      out.set(gibbsChain(model, start, sampleSteps, child(s0('gibbs'), e, c)).v, c * D)
    }
    return out
  }
  const checkpoint = (e: number) =>
    run.checkpoints.push({ epoch: e, weights: Float64Array.from(model.W), samples: snapshotSamples(e) })
  run.logLikelihood.push(exact ? rbmLogLikelihood(model, x) : NaN)
  run.reconstructionError.push(NaN)
  checkpoint(0)
  yield { ...run, checkpoints: run.checkpoints.slice() }
  const every = Math.max(1, Math.round(epochs / 25))
  for (let e = 1; e <= epochs; e++) {
    const order = Array.from({ length: n }, (_, r) => r)
    const u = units(child(s0('order'), e), n)
    order.sort((p, q) => u[p] - u[q])
    let err = 0
    for (let b = 0; b < n; b += batchSize) {
      const ids = order.slice(b, b + batchSize)
      const batch = new Float64Array(ids.length * D)
      ids.forEach((r, q) => batch.set(V.data.subarray(r * D, (r + 1) * D), q * D))
      const step = contrastiveDivergenceStep(model, fromData(batch, [ids.length, D]), child(s0('cd'), e, b), {
        k,
        learningRate,
        chains,
      })
      model = step.rbm
      if (persistent) chains = step.chains
      err += (step.reconstructionError * ids.length) / n
    }
    run.logLikelihood.push(exact ? rbmLogLikelihood(model, x) : NaN)
    run.reconstructionError.push(err)
    if (e % every === 0 || e === epochs) checkpoint(e)
    run.done = e === epochs
    yield {
      ...run,
      logLikelihood: [...run.logLikelihood],
      reconstructionError: [...run.reconstructionError],
      checkpoints: run.checkpoints.slice(),
    }
  }
  return run
}
