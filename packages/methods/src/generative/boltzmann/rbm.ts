/**
 * The Bernoulli restricted Boltzmann machine (Smolensky, 1986; Hinton, 2002): binary visible units
 * $\vvec \in \{0, 1\}^D$ and hidden units $\hvec \in \{0, 1\}^H$ with energy
 * $E(\vvec, \hvec) = -\avec^\top\vvec - \bvec^\top\hvec - \vvec^\top\Wmat\hvec$, so the conditionals factorise,
 * $p(h_j = 1 \mid \vvec) = \sigma(b_j + \vvec^\top\Wmat_{:j})$ and
 * $p(v_i = 1 \mid \hvec) = \sigma(a_i + \Wmat_{i:}\hvec)$ ($\Wmat_{:j}$ the $j$th column of $\Wmat$, $\Wmat_{i:}$
 * its $i$th row), and the free energy is
 * $F(\vvec) = -\avec^\top\vvec - \sum_j \log(1 + \exp(b_j + \vvec^\top\Wmat_{:j}))$.
 *
 * The log-likelihood gradient is
 * $\langle \vvec\hvec^\top \rangle_{\text{data}} - \langle \vvec\hvec^\top \rangle_{\text{model}}$. Contrastive
 * divergence (CD-$k$; Hinton, 2002) estimates the model term by $k$ steps of block Gibbs sampling started at the data;
 * persistent CD (PCD; Tieleman, 2008) keeps the chains between updates. For a small hidden layer the partition function
 * $Z = \sum_{\hvec} \exp(\bvec^\top\hvec) \prod_i (1 + \exp(a_i + \Wmat_{i:}\hvec))$ is summed exactly over the
 * $2^H$ hidden states, which gives the exact log-likelihood of the data to track training.
 *
 * An RBM is plain data (`Rbm`, weights row-major), and every function returns new arrays and leaves its arguments
 * unchanged. Units are read as numbers, so probabilities can stand in for binary states (as the mean-field
 * reconstructions and the layers of a deep belief network use them).
 */

import { child, stream, units, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The logistic function $\sigma(x) = 1 / (1 + e^{-x})$.
 *
 * @param x The argument.
 * @returns $\sigma(x)$, in $(0, 1)$.
 */
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
/**
 * The softplus $\log(1 + e^x)$, taken as $x$ itself above 30, where the two agree to double precision.
 *
 * @param x The argument.
 * @returns $\log(1 + e^x)$.
 */
const softplus = (x: number) => (x > 30 ? x : Math.log1p(Math.exp(x)))

/** An RBM's parameters: visible biases $\avec$ ($D$), hidden biases $\bvec$ ($H$), weights $\Wmat$ ($D \times H$). */
export interface Rbm {
  /** The number of visible units $D$. */
  readonly visible: number
  /** The number of hidden units $H$. */
  readonly hidden: number
  /** The visible biases $\avec$, $D$ values. */
  readonly a: Float64Array
  /** The hidden biases $\bvec$, $H$ values. */
  readonly b: Float64Array
  /** The weights $\Wmat$, row-major $D \times H$: $W_{ij}$ is entry `i * H + j`. */
  readonly W: Float64Array
}

/**
 * A new RBM with weights drawn from $\Gauss(0, 0.1^2)$ (by Box–Muller) and zero biases: large enough to leave the
 * symmetric start quickly.
 *
 * @param s The stream; the weights are drawn from its child `'weights'`.
 * @param visible The number of visible units $D$.
 * @param hidden The number of hidden units $H$.
 * @returns The RBM.
 *
 * @example A small RBM's parameters
 * const m = rbm(stream(1), 3, 2)
 * print('W (3 × 2, row-major) =', m.W)
 * print('a =', m.a, ' b =', m.b)
 */
export function rbm(s: Stream, visible: number, hidden: number): Rbm {
  const u = units(child(s, 'weights'), 2 * visible * hidden)
  const W = new Float64Array(visible * hidden)
  for (let k = 0; k < W.length; k++)
    W[k] = 0.1 * Math.sqrt(-2 * Math.log(1 - u[2 * k])) * Math.cos(2 * Math.PI * u[2 * k + 1])
  return { visible, hidden, a: new Float64Array(visible), b: new Float64Array(hidden), W }
}

/**
 * The hidden units' probabilities $p(h_j = 1 \mid \vvec) = \sigma(b_j + \vvec^\top\Wmat_{:j})$ for one visible vector.
 *
 * @param m The RBM.
 * @param v The visible vector, $D$ values: binary states, or probabilities for a mean-field pass.
 * @returns The $H$ probabilities.
 *
 * @example The hidden probabilities of one visible vector
 * const m = rbm(stream(1), 4, 3)
 * print('p(h = 1 | v) =', hiddenProbabilities(m, [1, 0, 1, 0]))
 */
export function hiddenProbabilities(m: Rbm, v: ArrayLike<number>): Float64Array {
  const out = Float64Array.from(m.b)
  for (let i = 0; i < m.visible; i++)
    if (v[i]) for (let j = 0; j < m.hidden; j++) out[j] += v[i] * m.W[i * m.hidden + j]
  return out.map(sigmoid)
}

/**
 * The visible units' probabilities $p(v_i = 1 \mid \hvec) = \sigma(a_i + \Wmat_{i:}\hvec)$ for one hidden vector.
 *
 * @param m The RBM.
 * @param h The hidden vector, $H$ values: binary states, or probabilities for a mean-field pass.
 * @returns The $D$ probabilities.
 *
 * @example Down from the hidden layer: a mean-field reconstruction of a visible vector
 * const m = rbm(stream(1), 4, 3)
 * const h = hiddenProbabilities(m, [1, 0, 1, 0])
 * print('p(v = 1 | h) =', visibleProbabilities(m, h))
 */
export function visibleProbabilities(m: Rbm, h: ArrayLike<number>): Float64Array {
  const out = Float64Array.from(m.a)
  for (let i = 0; i < m.visible; i++) {
    let s = 0
    for (let j = 0; j < m.hidden; j++) s += m.W[i * m.hidden + j] * h[j]
    out[i] += s
  }
  return out.map(sigmoid)
}

/**
 * The free energy $F(\vvec) = -\avec^\top\vvec - \sum_j \operatorname{softplus}(b_j + \vvec^\top\Wmat_{:j})$, with
 * $p(\vvec) = e^{-F(\vvec)} / Z$: the energy with the hidden units summed out.
 *
 * @param m The RBM.
 * @param v The visible vector, $D$ values.
 * @returns $F(\vvec)$.
 *
 * @example The free energy of a visible vector
 * const m = rbm(stream(1), 4, 3)
 * print('F([1, 0, 1, 0]) =', freeEnergy(m, [1, 0, 1, 0]))
 *
 * @example With zero weights and biases every hidden unit adds $-\log 2$
 * const m = { visible: 2, hidden: 3, a: new Float64Array(2), b: new Float64Array(3), W: new Float64Array(6) }
 * print('F =', freeEnergy(m, [1, 1]), ' -3 log 2 =', -3 * Math.log(2))
 */
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

/**
 * The log partition function $\log Z$, summed exactly over the $2^H$ hidden states (with the visible units summed out
 * in closed form). Throws `DomainError` for more than 20 hidden units.
 *
 * @param m The RBM, with $H \le 20$.
 * @returns $\log Z$.
 *
 * @example The probabilities $e^{-F(\vvec)} / Z$ of all $2^D$ visible vectors sum to 1
 * const m = rbm(stream(1), 3, 2)
 * const logZ = logPartition(m)
 * let total = 0
 * for (let k = 0; k < 8; k++) total += Math.exp(-freeEnergy(m, [k & 1, (k >> 1) & 1, (k >> 2) & 1]) - logZ)
 * print('log Z =', logZ)
 * print('sum of p(v) =', total)
 */
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

/**
 * The exact log-likelihood $-F(\vvec) - \log Z$, averaged over rows. Throws `DomainError` for more than 20 hidden
 * units (through `logPartition`).
 *
 * @param m The RBM, with $H \le 20$.
 * @param v The visible vectors, $n \times D$, one per row.
 * @returns The mean log-likelihood per row.
 *
 * @example Untrained, every pattern is near $-D \log 2$
 * const m = rbm(stream(1), 4, 2)
 * print('mean log p(v) =', rbmLogLikelihood(m, tensor([[1, 1, 0, 0], [0, 0, 1, 1]])))
 * print('-4 log 2 =', -4 * Math.log(2))
 */
export function rbmLogLikelihood(m: Rbm, v: MatrixLike): number {
  const V = dense.toMatrixF64(v, 'rbmLogLikelihood')
  const logZ = logPartition(m)
  let s = 0
  for (let r = 0; r < V.m; r++) s += -freeEnergy(m, V.data.subarray(r * V.n, (r + 1) * V.n)) - logZ
  return s / V.m
}

/**
 * Sample 0/1 units from their probabilities with uniforms: unit $i$ is 1 when $u_i < p_i$.
 *
 * @param p The units' probabilities of being 1.
 * @param u Uniforms on $[0, 1)$, at least as many as `p`.
 * @returns The 0/1 states, one per entry of `p`.
 */
const bernoulliRow = (p: ArrayLike<number>, u: ArrayLike<number>) => Float64Array.from(p, (q, i) => (u[i] < q ? 1 : 0))

/**
 * $k$ steps of block Gibbs sampling $\vvec \to \hvec \to \vvec$ from `v`, each step sampling the hidden units given
 * the visible ones and then the visible given the hidden.
 *
 * @param m The RBM.
 * @param v The starting visible vector, $D$ values; not modified.
 * @param k The number of Gibbs steps (0 returns a copy of `v`).
 * @param s The stream; step $t$ draws its hidden units from `child(s, 'h', t)` and visible ones from
 *   `child(s, 'v', t)`.
 * @returns `v`, the visible states after $k$ steps (0/1), and `h`, the hidden probabilities $p(\hvec = 1 \mid \vvec)$
 *   at those states (probabilities, not samples).
 *
 * @example One Gibbs step from a visible vector
 * const m = rbm(stream(1), 4, 3)
 * const { v, h } = gibbsChain(m, [1, 0, 1, 0], 1, stream(2))
 * print('v after one step =', v)
 * print('p(h = 1 | v) =', h)
 */
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
  /** Gibbs steps $k$ (default 1). */
  k?: number
  /** The step size $\eta$ (default 0.05). */
  learningRate?: number
  /**
   * Persistent chains (PCD): the chains' visible states, any number of them, each $D$ values. They are advanced
   * without being modified, and the advanced states are returned. Left out, one chain starts at each data row (CD).
   */
  chains?: Float64Array[]
}

/**
 * One CD-$k$ (or PCD-$k$) update on a minibatch of rows $\vvec$, with $\vvec'$ the chains' states after $k$ Gibbs
 * steps: $\Wmat \mathrel{+}= \eta (\langle \vvec\, p(\hvec \mid \vvec)^\top \rangle_{\text{data}} -
 * \langle \vvec'\, p(\hvec \mid \vvec')^\top \rangle_{\text{chains}})$, and likewise $\avec$ with $\vvec$ and
 * $\bvec$ with $p(\hvec \mid \vvec)$. The RBM passed in is not modified.
 *
 * @param m The RBM to update.
 * @param v The minibatch, $n \times D$, one visible vector per row.
 * @param s The stream; chain $c$ runs on `child(s, 'chain', c)`.
 * @param options The Gibbs steps $k$, the step size $\eta$ and, for PCD, the persistent chains.
 * @returns `rbm`, the updated RBM; `chains`, the chains' visible states after $k$ steps (pass them back for PCD); and
 *   `reconstructionError`, the squared error $\sum_i (p(v_i = 1 \mid p(\hvec \mid \vvec)) - v_i)^2$ of the mean-field
 *   reconstruction, averaged over the rows, under the RBM before the update.
 *
 * @example One contrastive-divergence update
 * const m = rbm(stream(1), 2, 2)
 * const step = contrastiveDivergenceStep(m, tensor([[1, 1], [0, 0]]), stream(2), { learningRate: 0.5 })
 * print('W before =', m.W)
 * print('W after =', step.rbm.W)
 * print('reconstruction error =', step.reconstructionError)
 *
 * @example A hundred CD-1 updates raise the log-likelihood towards its best, $-\log 2$ for two patterns
 * let m = rbm(stream(1), 4, 2)
 * const data = tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
 * print('log-likelihood before =', rbmLogLikelihood(m, data))
 * for (let t = 0; t < 100; t++) m = contrastiveDivergenceStep(m, data, stream(t), { learningRate: 1 }).rbm
 * print('log-likelihood after =', rbmLogLikelihood(m, data), ' best =', -Math.log(2))
 *
 * @example Persistent CD: the chains are passed back in at every update
 * let m = rbm(stream(1), 4, 2)
 * const data = tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
 * let chains = [new Float64Array(4), new Float64Array(4)]
 * for (let t = 0; t < 100; t++) {
 *   const step = contrastiveDivergenceStep(m, data, stream(t), { learningRate: 0.5, chains })
 *   m = step.rbm
 *   chains = step.chains
 * }
 * print('log-likelihood =', rbmLogLikelihood(m, data))
 * print('the chains end on =', chains)
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
  /** The number of hidden units $H$ (default 8). */
  hidden?: number
  /** Gibbs steps $k$ per update (default 1). */
  k?: number
  /**
   * Persistent CD (default false): keep one chain per minibatch row, started at the first rows of the data, across
   * updates.
   */
  persistent?: boolean
  /** The step size $\eta$ (default 0.1). */
  learningRate?: number
  /** Passes over the data (default 500). */
  epochs?: number
  /** Rows per minibatch (default 4). */
  batchSize?: number
  /** Gibbs steps of each chain whose end is a sample shown at a checkpoint (default 200). */
  sampleSteps?: number
  /** Samples shown at each checkpoint (default 16), from chains started at uniform random 0/1 states. */
  samples?: number
  /** The seed of the run's stream (default 0). */
  seed?: number | string
}

/** One checkpoint of an RBM run. */
export interface RbmCheckpoint {
  /** The epoch after which it was taken (0 before training). */
  epoch: number
  /** The weights $\Wmat$, row-major $D \times H$. */
  weights: Float64Array
  /** The samples from long Gibbs chains, each $D$ values, one after another. */
  samples: Float64Array
}

/** An RBM run: the exact log-likelihood (when $H \le 16$) and reconstruction error per epoch, and checkpoints. */
export interface RbmRun {
  /** The number of visible units $D$. */
  visible: number
  /** The number of hidden units $H$. */
  hidden: number
  /** The exact mean log-likelihood of the data after each epoch, from epoch 0; NaN for $H > 16$. */
  logLikelihood: number[]
  /** The mean reconstruction error over the epoch's minibatches, from epoch 0 (NaN). */
  reconstructionError: number[]
  /**
   * $-\log$ of the number of distinct training patterns: the mean log-likelihood of the uniform distribution on them,
   * the best possible when each appears equally often.
   */
  bestLogLikelihood: number
  /** The checkpoints so far: epoch 0, then about every twenty-fifth of the run, and the last epoch. */
  checkpoints: RbmCheckpoint[]
  /** True after the last epoch. */
  done: boolean
}

/**
 * Train an RBM by CD-$k$ or PCD-$k$ on binary rows, as a generator that yields a snapshot before training and after
 * every epoch. Each epoch visits the rows in a fresh random order, in minibatches. Deterministic in `seed`.
 *
 * @param data The training rows, $n \times D$, with 0/1 entries: a matrix, or `{ x }` holding one.
 * @param options The hidden layer, the CD variant and its step size, the epochs and minibatches, the samples shown and
 *   the seed.
 * @returns A generator of `RbmRun` snapshots; the last has `done` set.
 *
 * @example An RBM learns two patterns: the log-likelihood rises towards its best
 * const x = tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
 * let run
 * const options = { hidden: 2, epochs: 100, batchSize: 2, learningRate: 0.5, samples: 4, sampleSteps: 20 }
 * for (const r of rbmRun(x, options)) run = r
 * print('log-likelihood at epochs 0, 25, 50, 100:', [0, 25, 50, 100].map((e) => run.logLikelihood[e]))
 * print('best possible:', run.bestLogLikelihood)
 * print('samples at the end:', run.checkpoints.at(-1).samples)
 */
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
