/**
 * Running several chains of one sampler on child streams, and stacking their draws for the diagnostics (Gelman &
 * Rubin, 1992; Vehtari et al., 2021, run several chains so that $\hat R$ and the multi-chain ESS can be computed), and
 * the Rao–Blackwellised estimate over such draws.
 */

import { replicate, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { trace, type Algorithm, type Recorder, type Trace } from 'aifn-compute/foundation/trace'
import { data } from './util'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** Options for `sampleChains`. */
export type SampleChainsOptions<S> = {
  /** Number of chains $m$. Default 4. */
  chains?: number
  /** Steps per chain, counting the warmup. */
  steps: number
  /** The root stream: chain k runs on `child(stream, k)`. */
  stream: Stream
  /** Steps discarded from the start of each chain (burn-in): the states of steps 1 to `warmup`. Default 0. */
  warmup?: number
  /** Keep every `every`-th state (thinning). Default 1. */
  every?: number
  /** Extra recorders for each chain's trace (`x` is always recorded). */
  record?: Record<string, Recorder<S>>
}

/** The chains' traces and their draws stacked for the diagnostics. */
export type ChainsResult<S> = {
  /** Each chain's trace, warmup included, with `x` recorded at every kept step. */
  traces: Trace<S>[]
  /**
   * The kept draws after warmup, $m \times n \times d$: chain, draw, coordinate (the initial state is never a draw).
   * $n$ is the shortest chain's count, so a chain that stopped early truncates every chain.
   */
  draws: Tensor
  /** The step number of each draw (length $n$), from the first chain. */
  steps: number[]
}

/**
 * Run $m$ chains of `alg`, chain $k$ from `start(k)` on `child(stream, k)` (so chain $k$'s draws depend only on the
 * stream and $k$: adding chains keeps the first ones), and stack the draws $\xvec$ after `warmup` into an
 * $m \times n \times d$ tensor for `effectiveSampleSize`, `splitRhat` and `monteCarloStandardError`.
 *
 * @param alg The sampler, any algorithm whose state has a point `x`.
 * @param start The start of every chain, or a function from the chain index $k$ to its start (for overdispersed
 *   starts).
 * @param options The number of `chains`, the `steps` and root `stream`, and the `warmup`, thinning (`every`) and
 *   extra recorders (`record`).
 * @returns The traces, the stacked draws and the step number of each draw.
 *
 * @example Four chains of a random walk on a standard normal
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * const { draws, steps } = sampleChains(randomWalkMetropolis(target, { scale: 2.4 }), { x0: [0] }, {
 *   chains: 4, steps: 250, warmup: 50, stream: stream(1),
 * })
 * print('draws shape =', draws.shape)
 * print('first and last step kept =', steps[0], steps[steps.length - 1])
 * print('mean =', mean(draws))
 * print('split R-hat =', splitRhat(draws))
 *
 * @example Overdispersed starts, one per chain
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * const { draws } = sampleChains(randomWalkMetropolis(target, { scale: 2.4 }), (k) => ({ x0: [10 * (k - 1.5)] }), {
 *   chains: 4, steps: 5, stream: stream(2),
 * })
 * print('first draw of each chain =', toFlat(draws).filter((_, i) => i % 5 === 0))
 * print('split R-hat after 5 steps =', splitRhat(draws))
 */
export function sampleChains<Start, S extends Status & { x: Vector }>(
  alg: Algorithm<Start, S>,
  start: Start | ((k: number) => Start),
  options: SampleChainsOptions<S>,
): ChainsResult<S> {
  const { chains = 4, steps, stream, warmup = 0, every = 1 } = options
  const startFor = (k: number) => (typeof start === 'function' ? (start as (k: number) => Start)(k) : start)
  const traces = replicate(
    chains,
    stream,
    (s, k) => trace(alg, startFor(k), steps, { stream: s, every, record: { x: (st: S) => st.x, ...options.record } }),
    { cache: false },
  )
  const kept = (index: Int32Array) => Array.from(index, (i, j) => [i, j] as const).filter(([i]) => i > warmup && i > 0)
  const keep = kept(traces[0].index)
  const n = Math.min(...traces.map((tr) => kept(tr.index).length))
  const d = traces[0].final.x.shape[0]
  const out = new Float64Array(chains * n * d)
  traces.forEach((tr, k) => {
    const x = data(tr.series.x)
    const rows = kept(tr.index)
    for (let r = 0; r < n; r++) out.set(x.subarray(rows[r][1] * d, (rows[r][1] + 1) * d), (k * n + r) * d)
  })
  return { traces, draws: fromData(out, [chains, n, d]), steps: keep.slice(0, n).map(([i]) => i) }
}

/** The result of `raoBlackwell`. */
export type RaoBlackwellEstimate = {
  /** The estimate: the average of the conditional expectation over every draw (length $k$). */
  mean: Vector
  /**
   * The conditional expectation at each draw, $m \times n \times k$ (chain, draw, component): a chain of its own, for
   * `monteCarloStandardError`, `effectiveSampleSize` and `splitRhat`.
   */
  values: Tensor
}

/**
 * The Rao–Blackwellised estimate of $\expect[h(\xvec)]$ (Gelfand and Smith, 1990; Casella and Robert, 1996): the
 * average over the draws of a conditional expectation $g(\xvec) = \expect[h(\xvec) \mid \xvec_{-B}]$, e.g.
 * `conditionalMean(blocks)` for $h(\xvec) = \xvec$. Both averages are unbiased; the conditional one removes the
 * variance of $h$ given $\xvec_{-B}$. `draws` is $m \times n \times d$ (as `sampleChains` returns) or $n \times d$,
 * and $g$ returns a number or $k$ numbers (an array or a vector). Throws `ShapeError` for draws of another rank, or
 * when $g$ changes how many values it returns.
 *
 * @param draws The draws, $m \times n \times d$ or (one chain) $n \times d$.
 * @param expectation The conditional expectation $g$, called once per draw with the draw as a vector of $d$ values.
 * @returns The estimate and the value of $g$ at every draw.
 *
 * @example The mean of a correlated Gaussian, two ways
 * const blocks = gaussianConditionals([2, 0], [[1, 0.9], [0.9, 1]])
 * const { draws } = sampleChains(gibbs(blocks), { x0: [0, 0] }, {
 *   chains: 2, steps: 200, warmup: 20, stream: stream(3),
 * })
 * const rb = raoBlackwell(draws, conditionalMean(blocks))
 * print('plain average =', mean(reshape(draws, [-1, 2]), 0))
 * print('Rao-Blackwellised =', rb.mean)
 * print('MCSE, plain =', monteCarloStandardError(draws))
 * print('MCSE, Rao-Blackwellised =', monteCarloStandardError(rb.values))
 *
 * @example A scalar expectation
 * // E[x0^2 | x1] = Var + mean^2 = (1 - 0.81) + (2 + 0.9 x1)^2 under the same target.
 * const blocks = gaussianConditionals([2, 0], [[1, 0.9], [0.9, 1]])
 * const { draws } = sampleChains(gibbs(blocks), { x0: [0, 0] }, {
 *   chains: 2, steps: 200, warmup: 20, stream: stream(3),
 * })
 * print('E[x0^2] =', raoBlackwell(draws, (x) => 0.19 + (2 + 0.9 * x.data[1]) ** 2).mean)
 */
export function raoBlackwell(
  draws: Tensor,
  expectation: (x: Vector) => number | ArrayLike<number> | Tensor,
): RaoBlackwellEstimate {
  const shape = draws.shape.length === 2 ? [1, ...draws.shape] : draws.shape
  if (shape.length !== 3)
    throw new ShapeError('raoBlackwell', `raoBlackwell: draws must be m×n×d or n×d, got rank ${draws.shape.length}`)
  const [m, n, d] = shape
  const x = data(draws)
  let k = -1
  let out = new Float64Array(0)
  let total = new Float64Array(0)
  for (let r = 0; r < m * n; r++) {
    const g = expectation(fromData(x.slice(r * d, (r + 1) * d), [d]) as Vector)
    const row = typeof g === 'number' ? [g] : isTensor(g) ? data(g) : g
    if (k < 0) {
      k = row.length
      out = new Float64Array(m * n * k)
      total = new Float64Array(k)
    } else if (row.length !== k)
      throw new ShapeError('raoBlackwell', `raoBlackwell: the expectation returned ${row.length} values, then ${k}`)
    for (let j = 0; j < k; j++) {
      out[r * k + j] = row[j]
      total[j] += row[j]
    }
  }
  return {
    mean: fromData(
      total.map((v) => v / (m * n)),
      [k],
    ) as Vector,
    values: fromData(out, [m, n, k]),
  }
}
