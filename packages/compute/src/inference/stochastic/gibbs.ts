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

/** A full conditional: a draw of coordinate i given the current point x (whose coordinate i is ignored). */
export type Conditional = (x: Vector, s: Stream) => number

/**
 * A block of a Gibbs sampler: the coordinates B it updates, a joint draw x_B ~ π(x_B | x₋B) (the values of x at B are
 * ignored), and optionally the conditional mean E[x_B | x₋B], which Rao–Blackwellised estimates average
 * (`conditionalMean`).
 */
export type Block = {
  readonly coordinates: readonly number[]
  readonly draw: (x: Vector, s: Stream) => ArrayLike<number>
  readonly mean?: (x: Vector) => ArrayLike<number>
}

/** The state of `gibbs`. `logDensity` is NaN: Gibbs never evaluates the joint. */
export type GibbsState = ChainState & {
  /** The path within the last step: the point before and after each block update, (updates + 1)×d. */
  moves: Matrix
  /** The block updated at each move of the last step (int32; with one conditional per coordinate, the coordinate). */
  blocks: Tensor
  /** Block updates so far. */
  updates: number
}

/** Options for `gibbs`. */
export type GibbsOptions = {
  /** `systematic` updates blocks 0, 1, …, b − 1 in order; `random` picks each uniformly. Default systematic. */
  scan?: 'systematic' | 'random'
  /** Block updates per step (a sweep). Default the number of blocks. */
  updatesPerStep?: number
}

/** Blocks from conditionals: a function at index i is the conditional of coordinate i, a block of one. */
function asBlocks(conditionals: readonly Conditional[] | readonly Block[]): Block[] {
  return conditionals.map((c, i) =>
    typeof c === 'function' ? { coordinates: [i], draw: (x: Vector, s: Stream) => [(c as Conditional)(x, s)] } : c,
  )
}

/** The dimension the blocks cover, checking every coordinate below it is updated by some block. */
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
 * its full conditional π(x_B | x₋B). Each update leaves π invariant, so both scans do; the random scan is also
 * reversible. Pass one `Conditional` per coordinate, or `Block`s: drawing strongly correlated coordinates jointly
 * (block Gibbs; Liu, Wong and Kong, 1994) removes the slow zig-zag of one-at-a-time updates. One step is one sweep of
 * `updatesPerStep` updates, and `moves` holds the path it traced. Update k of step t draws from
 * `child(ctx.stream, k)`.
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
 * The conditional mean of every coordinate, x ↦ (E[x_B | x₋B] for each block B), from blocks that all give `mean` (a
 * coordinate in several blocks takes the first). Averaged over a chain's draws (`raoBlackwell` in `./chains`), it is
 * the Rao–Blackwellised estimate of E[x] (Gelfand and Smith, 1990): unbiased like the plain average, and never of
 * larger variance for independent draws, since Var E[x_B | x₋B] ≤ Var x_B.
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
 * The full conditionals of N(μ, Σ) for a partition of the coordinates into blocks (default: one block per
 * coordinate). With Λ = Σ⁻¹, x_B | x₋B ~ N(μ_B − Λ_BB⁻¹ Λ_B,₋B (x₋B − μ₋B), Λ_BB⁻¹) (Bishop, 2006, eqs. 2.73 and 2.75
 * in precision form); each block gives its conditional mean too. For a bivariate Gaussian with correlation ρ and unit
 * variances the single-coordinate conditionals are x₁ | x₂ ~ N(ρx₂, 1 − ρ²): the closer |ρ| is to 1, the shorter each
 * zig-zag step and the slower the chain, unless both coordinates share a block.
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

/** The full conditionals of the standard bivariate Gaussian with correlation ρ (unit variances, zero means). */
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
  /** log y, the slice level drawn for each coordinate on the last sweep. */
  levels: Vector
  /** The final bracket [L, R] (after stepping out and shrinking) for each coordinate on the last sweep, d×2. */
  intervals: Matrix
  /** The bracket after stepping out and before shrinking, d×2. */
  steppedOut: Matrix
  /** Shrinkage steps on the last sweep, per coordinate. */
  shrinks: Vector
  /** Log-density evaluations so far. */
  evaluations: number
}

/** Options for `sliceSampler`. */
export type SliceOptions = {
  /** Initial bracket width w (one number or one per coordinate). Default 1. */
  width?: number | ArrayLike<number>
  /** Largest number of stepping-out steps m (Neal, 2003, Fig. 3). Default 32. */
  maxSteps?: number
}

/**
 * Univariate slice sampling in turn along each coordinate (Neal, 2003, §4, stepping out and shrinkage, Figs. 3 and 5):
 * draw a level log y = log π(x) + log u, place a bracket of width w at random around xᵢ, step it out until both ends
 * leave the slice, then draw uniformly in it, shrinking towards xᵢ on each rejection. It needs no step-size tuning
 * beyond w and is exact. One step is one sweep; sweep t, coordinate i draws from `child(ctx.stream, i)`.
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
