/**
 * Gibbs sampling (Geman & Geman, 1984) with systematic or random scan over user-supplied full conditionals, one
 * coordinate or one block of coordinates at a time (block Gibbs), the Gaussian full conditionals of any partition, the
 * conditional means that Rao–Blackwellised estimates average, and univariate slice sampling (Neal, 2003) as a Gibbs
 * step for any target.
 */

import { cholesky, inverse, solveTriangular } from 'aifn-compute/numerics/linalg'
import { child, integers, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { badLogDensity } from './metropolis'
import type { ChainStart, ChainState, LogDensity, VectorLike } from './types'
import { allFinite, data, logDensityAt, mat, standardNormals, toF64, vec, type F64 } from './util'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A full conditional: a draw of coordinate $i$ given the current point $\xvec$ (whose coordinate $i$ is ignored), from
 * the stream it is given.
 */
export type Conditional = (x: Vector, s: Stream) => number

/**
 * A block of a Gibbs sampler: the coordinates $B$ it updates, a joint draw
 * $\xvec_B \sim \pi(\xvec_B \mid \xvec_{-B})$ (the values of $\xvec$ at $B$ are ignored), and optionally the
 * conditional mean $\expect[\xvec_B \mid \xvec_{-B}]$, which Rao–Blackwellised estimates average (`conditionalMean`).
 */
export type Block = {
  /** The indices of the coordinates in $B$, in the order `draw` and `mean` return their values. */
  readonly coordinates: readonly number[]
  /** Draws $\xvec_B$ given the current point, one value per coordinate of $B$, from the stream it is given. */
  readonly draw: (x: Vector, s: Stream) => ArrayLike<number>
  /** $\expect[\xvec_B \mid \xvec_{-B}]$ at the current point, one value per coordinate of $B$. */
  readonly mean?: (x: Vector) => ArrayLike<number>
}

/** The state of `gibbs`. `logDensity` is NaN: Gibbs never evaluates the joint. */
export type GibbsState = ChainState & {
  /**
   * The path within the last step: the point before and after each block update, $(u + 1) \times d$ for $u$ updates
   * per step.
   */
  moves: Matrix
  /** The block updated at each move of the last step (int32; with one conditional per coordinate, the coordinate). */
  blocks: Tensor
  /** Block updates so far. */
  updates: number
}

/** Options for `gibbs`. */
export type GibbsOptions = {
  /**
   * `systematic` updates blocks $0, 1, \dots, b - 1$ in order (carrying on across steps where the last one stopped);
   * `random` picks each uniformly. Default systematic.
   */
  scan?: 'systematic' | 'random'
  /** Block updates per step (a sweep). Default the number of blocks. */
  updatesPerStep?: number
}

/**
 * Blocks from conditionals: a function at index $i$ is the conditional of coordinate $i$, a block of one.
 *
 * @param conditionals Conditionals and blocks, which may be mixed; a block is kept as it is.
 * @returns One block per entry, in order.
 */
function asBlocks(conditionals: readonly Conditional[] | readonly Block[]): Block[] {
  return conditionals.map((c, i) =>
    typeof c === 'function' ? { coordinates: [i], draw: (x: Vector, s: Stream) => [(c as Conditional)(x, s)] } : c,
  )
}

/**
 * The dimension the blocks cover, checking every coordinate below it is updated by some block. Throws `DomainError`
 * for an empty block, a coordinate that is not a non-negative integer, or a coordinate no block updates.
 *
 * @param blocks The blocks to check.
 * @param name The caller's name for error messages.
 * @returns $d$, one more than the largest coordinate of any block (0 with no blocks).
 */
function blockDimension(blocks: readonly Block[], name: string): number {
  const d = Math.max(-1, ...blocks.flatMap((b) => b.coordinates)) + 1
  const covered = new Uint8Array(d)
  for (const b of blocks) {
    if (b.coordinates.length === 0) throw new DomainError(name, `${name}: a block has no coordinates`)
    for (const i of b.coordinates) {
      if (!Number.isInteger(i) || i < 0) throw new DomainError(name, `${name}: coordinate ${i} is not an index`)
      covered[i] = 1
    }
  }
  const missing = covered.indexOf(0)
  if (missing >= 0) throw new DomainError(name, `${name}: no block updates coordinate ${missing}`)
  return d
}

/**
 * Gibbs sampling (Geman & Geman, 1984): replace one coordinate, or one block of coordinates, at a time by a draw from
 * its full conditional $\pi(\xvec_B \mid \xvec_{-B})$. Each update leaves $\pi$ invariant, so both scans do; the
 * random scan is also reversible. Pass one `Conditional` per coordinate, or `Block`s: drawing strongly correlated
 * coordinates jointly (block Gibbs; Liu, Wong and Kong, 1994) removes the slow zig-zag of one-at-a-time updates. One
 * step is one sweep of `updatesPerStep` updates, and `moves` holds the path it traced. Update $k$ of step $t$ draws
 * from `child(ctx.stream, k)`. Throws `DomainError` when the blocks do not cover coordinates $0, \dots, d - 1$, and
 * `ShapeError` when $\xvec_0$ or a block's draw has the wrong length.
 *
 * @param conditionals One `Conditional` per coordinate (entry $i$ draws coordinate $i$), or `Block`s whose coordinates
 *   together cover $0, \dots, d - 1$.
 * @param options The `scan` order and the number of `updatesPerStep`.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example One coordinate at a time on a correlated Gaussian
 * const { draws } = sampleChains(gibbs(bivariateGaussianConditionals(0.8)), { x0: [0, 0] }, {
 *   chains: 2, steps: 300, stream: stream(1),
 * })
 * const x = reshape(draws, [-1, 2])
 * print('mean =', mean(x, 0))
 * print('second moments =', div(matmul(transpose(x), x), x.shape[0]))
 *
 * @example The path of one sweep, and a joint block
 * const one = run(gibbs(bivariateGaussianConditionals(0.95)), { x0: [1, 1] }, 1)
 * print('coordinate-wise moves =', one.moves)
 * const both = run(gibbs(gaussianConditionals([0, 0], [[1, 0.95], [0.95, 1]], [[0, 1]])), { x0: [1, 1] }, 1)
 * print('block moves =', both.moves)
 */
export function gibbs(
  conditionals: readonly Conditional[] | readonly Block[],
  options: GibbsOptions = {},
): Algorithm<ChainStart, GibbsState> {
  const name = 'gibbs'
  const blocks = asBlocks(conditionals)
  const d = blockDimension(blocks, name)
  const { scan = 'systematic', updatesPerStep = blocks.length } = options
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      if (x.length !== d)
        throw new ShapeError(name, `${name}: x0 has ${x.length} values for blocks covering ${d} coordinates`)
      return {
        t: 0,
        x: vec(x),
        logDensity: NaN,
        moves: mat(Float64Array.from(x), 1, d),
        blocks: fromData(new Int32Array(0), [0]),
        updates: 0,
        diverged: !allFinite(x),
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      let x = data(s.x)
      const moves = new Float64Array((updatesPerStep + 1) * d)
      moves.set(x, 0)
      const chosen = new Int32Array(updatesPerStep)
      for (let k = 0; k < updatesPerStep; k++) {
        const u = child(draws, k)
        const j = scan === 'random' ? integers(child(u, 'coordinate'), blocks.length) : (s.updates + k) % blocks.length
        const block = blocks[j]
        const value = block.draw(vec(x), u)
        if (value.length !== block.coordinates.length)
          throw new ShapeError(
            name,
            `${name}: block ${j} drew ${value.length} values for ${block.coordinates.length} coordinates`,
          )
        const next = Float64Array.from(x)
        block.coordinates.forEach((i, m) => (next[i] = value[m]))
        x = next
        chosen[k] = j
        moves.set(x, (k + 1) * d)
      }
      return {
        ...s,
        t: s.t + 1,
        x: vec(x),
        moves: mat(moves, updatesPerStep + 1, d),
        blocks: fromData(chosen, [updatesPerStep]),
        updates: s.updates + updatesPerStep,
        diverged: !allFinite(x),
      }
    },
  }
}

/**
 * The conditional mean of every coordinate, $\xvec \mapsto (\expect[\xvec_B \mid \xvec_{-B}]$ for each block
 * $B)$, from blocks that all give `mean` (a coordinate in several blocks takes the first). Averaged over a chain's
 * draws (`raoBlackwell` in `./chains`), it is the Rao–Blackwellised estimate of $\expect[\xvec]$ (Gelfand and Smith,
 * 1990): unbiased like the plain average, and never of larger variance for independent draws, since
 * $\var \expect[\xvec_B \mid \xvec_{-B}] \le \var \xvec_B$. Throws `DomainError` when a block has no `mean` or
 * the blocks do not cover every coordinate.
 *
 * @param blocks Blocks covering coordinates $0, \dots, d - 1$, each with its `mean`.
 * @returns A function from a point $\xvec$ ($d$ values) to the vector of conditional means ($d$ values).
 *
 * @example The conditional mean at a point
 * const blocks = gaussianConditionals([0, 0], [[1, 0.8], [0.8, 1]])
 * print('conditional means at x = (1, 2) =', conditionalMean(blocks)(tensor([1, 2])))
 *
 * @example A Rao–Blackwellised estimate of the mean
 * const blocks = gaussianConditionals([1, -1], [[1, 0.8], [0.8, 1]])
 * const { draws } = sampleChains(gibbs(blocks), { x0: [0, 0] }, { chains: 2, steps: 200, stream: stream(4) })
 * print('plain average =', mean(reshape(draws, [-1, 2]), 0))
 * print('Rao-Blackwellised =', raoBlackwell(draws, conditionalMean(blocks)).mean)
 */
export function conditionalMean(blocks: readonly Block[]): (x: Vector) => Vector {
  const d = blockDimension(blocks, 'conditionalMean')
  blocks.forEach((b, j) => {
    if (!b.mean) throw new DomainError('conditionalMean', `conditionalMean: block ${j} has no mean`)
  })
  return (x) => {
    const out = new Float64Array(d)
    const set = new Uint8Array(d)
    for (const b of blocks) {
      const m = b.mean!(x)
      b.coordinates.forEach((i, k) => {
        if (!set[i]) {
          out[i] = m[k]
          set[i] = 1
        }
      })
    }
    return vec(out)
  }
}

/**
 * The full conditionals of $\Gauss(\muvec, \Sigmamat)$ for a partition of the coordinates into blocks (default: one
 * block per coordinate). With $\Lambdamat = \Sigmamat^{-1}$,
 * $\xvec_B \mid \xvec_{-B} \sim \Gauss(\mvec_B, \Lambdamat_{BB}^{-1})$ with
 * $\mvec_B = \muvec_B - \Lambdamat_{BB}^{-1} \Lambdamat_{B,-B} (\xvec_{-B} - \muvec_{-B})$ (Bishop, 2006, eqs. 2.73
 * and 2.75 in precision form); each block gives its conditional mean $\mvec_B$ too. For a bivariate
 * Gaussian with correlation $\rho$ and unit variances the single-coordinate conditionals are
 * $x_1 \mid x_2 \sim \Gauss(\rho x_2, 1 - \rho^2)$: the closer $\lvert \rho \rvert$ is to 1, the shorter each
 * zig-zag step and the slower the chain, unless both coordinates share a block. $\Sigmamat$ is inverted once; a
 * singular one throws from `inverse`, and a block whose precision $\Lambdamat_{BB}$ is not positive definite throws
 * `NumericalError`.
 *
 * @param mean The mean $\muvec$ ($d$ values).
 * @param covariance The covariance $\Sigmamat$ ($d \times d$), as a matrix or as rows.
 * @param partition The blocks, each a list of coordinate indices; left out, each coordinate is a block of its own.
 *   It is not checked to cover every coordinate here (`gibbs` checks).
 * @returns One `Block` per part, with `draw` and `mean`, for `gibbs` and `conditionalMean`.
 *
 * @example A conditional mean by eye
 * // With unit variances and correlation 0.8, E[x1 | x2] = 0.8 x2.
 * const [first] = gaussianConditionals([0, 0], [[1, 0.8], [0.8, 1]])
 * print('E[x1 | x2 = 1] =', first.mean(tensor([0, 1])))
 * print('E[x1 | x2 = -2] =', first.mean(tensor([0, -2])))
 *
 * @example Coordinates, or one joint block
 * // With one block covering both coordinates, each Gibbs step is an independent draw.
 * const covariance = [[1, 0.95], [0.95, 1]]
 * for (const partition of [undefined, [[0, 1]]]) {
 *   const { draws } = sampleChains(gibbs(gaussianConditionals([0, 0], covariance, partition)), { x0: [0, 0] }, {
 *     chains: 2, steps: 200, stream: stream(5),
 *   })
 *   print(partition ? 'one block:' : 'coordinates:', 'ESS =', effectiveSampleSize(draws))
 * }
 */
export function gaussianConditionals(
  mean: VectorLike,
  covariance: Tensor | readonly (readonly number[])[],
  partition?: readonly (readonly number[])[],
): Block[] {
  const name = 'gaussianConditionals'
  const mu = toF64(mean, name)
  const d = mu.length
  const cov = Array.isArray(covariance) ? tensor(covariance as number[][]) : (covariance as Tensor)
  const P = data(inverse(cov))
  const parts = partition ?? Array.from({ length: d }, (_, i) => [i])
  return parts.map((B) => {
    const nb = B.length
    const inB = new Set(B)
    const rest = Array.from({ length: d }, (_, i) => i).filter((i) => !inB.has(i))
    const Pbb = new Float64Array(nb * nb)
    B.forEach((i, a) => B.forEach((j, b) => (Pbb[a * nb + b] = P[i * d + j])))
    const chol = cholesky(fromData(Pbb, [nb, nb]))
    if (chol.failed)
      throw new NumericalError(
        name,
        `${name}: the precision block [${B.join(', ')}] is not positive definite`,
        'not-positive-definite',
      )
    const L = chol.L
    const conditionalMean = (x: Vector) => {
      const xs = data(x)
      // Λ_BB m = Λ_B,₋B (x₋B − μ₋B), then μ_B − m.
      const r = new Float64Array(nb)
      B.forEach((i, a) => {
        for (const j of rest) r[a] += P[i * d + j] * (xs[j] - mu[j])
      })
      const y = solveTriangular(L, fromData(r, [nb]))
      const shift = data(solveTriangular(L, y, { transpose: true }))
      return Float64Array.from(B, (i, a) => mu[i] - shift[a])
    }
    return {
      coordinates: B,
      mean: conditionalMean,
      draw: (x: Vector, s: Stream) => {
        // Λ_BB = L Lᵀ, so L⁻ᵀ z has covariance Λ_BB⁻¹.
        const z = data(solveTriangular(L, fromData(standardNormals(s, nb), [nb]), { transpose: true }))
        return conditionalMean(x).map((m, a) => m + z[a])
      },
    }
  })
}

/**
 * The full conditionals of the standard bivariate Gaussian with correlation $\rho$ (unit variances, zero means):
 * $x_1 \mid x_2 \sim \Gauss(\rho x_2, 1 - \rho^2)$ and the same with the coordinates swapped. Throws `DomainError`
 * unless $\lvert \rho \rvert < 1$.
 *
 * @param rho The correlation $\rho$ between the two coordinates.
 * @returns The two conditionals, of coordinate 0 and of coordinate 1, for `gibbs`.
 *
 * @example Strong correlation slows the chain
 * for (const rho of [0, 0.5, 0.95]) {
 *   const { draws } = sampleChains(gibbs(bivariateGaussianConditionals(rho)), { x0: [0, 0] }, {
 *     chains: 2, steps: 200, stream: stream(6),
 *   })
 *   print(`rho = ${rho}: ESS of 400 draws =`, effectiveSampleSize(draws))
 * }
 */
export function bivariateGaussianConditionals(rho: number): Conditional[] {
  if (!(Math.abs(rho) < 1))
    throw new DomainError('bivariateGaussianConditionals', 'bivariateGaussianConditionals: |ρ| must be below 1')
  const sd = Math.sqrt(1 - rho * rho)
  return [(x, s) => normal(s, rho * data(x)[1], sd), (x, s) => normal(s, rho * data(x)[0], sd)]
}

// ---------------------------------------------------------------------------------------------------------------------
// Slice sampling.

/** The state of `sliceSampler`. */
export type SliceState = ChainState & {
  /** $\log y$, the slice level drawn for each coordinate on the last sweep. */
  levels: Vector
  /**
   * The final bracket $[L, R]$ (after stepping out and shrinking) for each coordinate on the last sweep, $d \times 2$.
   */
  intervals: Matrix
  /** The bracket after stepping out and before shrinking, $d \times 2$. */
  steppedOut: Matrix
  /** Shrinkage steps on the last sweep, per coordinate. */
  shrinks: Vector
  /** Log-density evaluations so far. */
  evaluations: number
}

/** Options for `sliceSampler`. */
export type SliceOptions = {
  /** Initial bracket width $w$ (one number or one per coordinate). Default 1. */
  width?: number | ArrayLike<number>
  /**
   * Largest size $m$ of the stepped-out bracket, in widths (Neal, 2003, Fig. 3): at most $m - 1$ steps, split at
   * random between the two ends. Default 32.
   */
  maxSteps?: number
}

/**
 * Univariate slice sampling in turn along each coordinate (Neal, 2003, §4, stepping out and shrinkage, Figs. 3 and 5):
 * draw a level $\log y = \log \pi(\xvec) + \log u$, place a bracket of width $w$ at random around $x_i$, step it
 * out until both ends leave the slice, then draw uniformly in it, shrinking towards $x_i$ on each rejection. It needs
 * no step-size tuning beyond $w$ and is exact. One step is one sweep; sweep $t$, coordinate $i$ draws from
 * `child(ctx.stream, i)`. The start must lie in the support (`DomainError` otherwise).
 *
 * @param target The target $\pi$, through its (possibly unnormalised) `logDensity` and `dim`.
 * @param options The bracket `width` $w$ and the stepping-out limit `maxSteps`.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example A standard normal
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * const { draws, traces } = sampleChains(sliceSampler(target, { width: 2 }), { x0: [0] }, {
 *   chains: 2, steps: 200, stream: stream(7),
 * })
 * print('mean =', mean(draws))
 * print('variance =', variance(draws))
 * print('evaluations per step =', traces[0].final.evaluations / 200)
 *
 * @example One sweep's bracket
 * const target = { kind: 'log-density', dim: 1, normalised: false, logDensity: (x) => mul(-0.5, sum(mul(x, x))) }
 * const s = run(sliceSampler(target, { width: 0.5 }), { x0: [0] }, 1)
 * print('level log y =', s.levels)
 * print('stepped out =', s.steppedOut)
 * print('after shrinking =', s.intervals)
 * print('new point =', s.x)
 */
export function sliceSampler(target: LogDensity, options: SliceOptions = {}): Algorithm<ChainStart, SliceState> {
  const name = 'slice'
  const d = target.dim
  const { maxSteps = 32 } = options
  const width =
    typeof options.width === 'number' || options.width === undefined ? null : Float64Array.from(options.width)
  const w = (i: number) => (width ? width[i] : ((options.width as number | undefined) ?? 1))
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      if (x.length !== d) throw new ShapeError(name, `${name}: x0 has ${x.length} values for dimension ${d}`)
      const logDensity = logDensityAt(target, x)
      if (logDensity === -Infinity) throw new DomainError(name, `${name}: x0 is outside the support`)
      return {
        t: 0,
        x: vec(x),
        logDensity,
        levels: vec(new Float64Array(d).fill(NaN)),
        intervals: mat(new Float64Array(2 * d).fill(NaN), d, 2),
        steppedOut: mat(new Float64Array(2 * d).fill(NaN), d, 2),
        shrinks: vec(new Float64Array(d)),
        evaluations: 1,
        diverged: badLogDensity(logDensity) || !allFinite(x),
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      const x: F64 = Float64Array.from(data(s.x))
      let logDensity = s.logDensity
      let evaluations = s.evaluations
      const levels = new Float64Array(d)
      const intervals = new Float64Array(2 * d)
      const steppedOut = new Float64Array(2 * d)
      const shrinks = new Float64Array(d)
      const at = (i: number, v: number) => {
        const y = Float64Array.from(x)
        y[i] = v
        evaluations++
        return logDensityAt(target, y)
      }
      for (let i = 0; i < d; i++) {
        const u = child(draws, i)
        const level = logDensity + Math.log(uniform(u))
        const wi = w(i)
        const x0 = x[i]
        let L = x0 - wi * uniform(u)
        let R = L + wi
        // Stepping out, with at most maxSteps steps split at random between the ends.
        let J = Math.floor(maxSteps * uniform(u))
        let K = maxSteps - 1 - J
        while (J > 0 && at(i, L) > level) {
          L -= wi
          J--
        }
        while (K > 0 && at(i, R) > level) {
          R += wi
          K--
        }
        steppedOut[2 * i] = L
        steppedOut[2 * i + 1] = R
        let count = 0
        for (;;) {
          const x1 = L + uniform(u) * (R - L)
          const logX1 = at(i, x1)
          if (logX1 > level) {
            x[i] = x1
            logDensity = logX1
            break
          }
          count++
          if (x1 < x0) L = x1
          else R = x1
          if (R - L < 1e-12 * (1 + Math.abs(x0))) break // the bracket has collapsed onto x0: keep it
        }
        levels[i] = level
        intervals[2 * i] = L
        intervals[2 * i + 1] = R
        shrinks[i] = count
      }
      return {
        ...s,
        t: s.t + 1,
        x: vec(x),
        logDensity,
        levels: vec(levels),
        intervals: mat(intervals, d, 2),
        steppedOut: mat(steppedOut, d, 2),
        shrinks: vec(shrinks),
        evaluations,
        diverged: badLogDensity(logDensity) || !allFinite(x),
      }
    },
  }
}
