/**
 * GAM terms: specifications (`s`, `cyclic`, `te`, `linearTerm`, `factorTerm`) and their construction on training data
 * as model-matrix blocks with penalties. Smooths get a sum-to-zero identifiability constraint absorbed by a Householder
 * reflection, so each block has one column fewer than its basis and the intercept carries the mean (Wood, 2017,
 * "Generalized Additive Models", 2nd ed., §5.4.1).
 */

import {
  bsplineBasis,
  bsplineCount,
  cyclicBsplineBasis,
  cyclicDifferencePenalty,
  differenceMatrix,
  differencePenalty,
  tensorProductBasis,
  tensorProductPenalties,
  thinPlateRegressionBasis,
  uniformKnots,
} from 'aifn-compute/numerics/interpolate'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

type F64 = Float64Array
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

/** A shape constraint on a P-spline smooth, imposed on its coefficients' differences. */
export type ShapeConstraint = 'increasing' | 'decreasing' | 'convex' | 'concave'

/** Options of a one-dimensional smooth. */
export type SmoothOptions = {
  /** Basis functions before the centring constraint (default 10). */
  k?: number
  /** B-spline degree (default 3). */
  degree?: number
  /** Order of the difference penalty (default 2). */
  order?: number
  /** Range the basis covers (default: the training range). */
  range?: readonly [number, number]
  /** A fixed smoothing parameter λ (default: selected with the model's method). */
  lambda?: number
  /** Multiply the smooth by this feature (a varying-coefficient term), or with `byFactor` fit one smooth per level. */
  by?: number
  byFactor?: boolean
  /** A shape constraint (P-spline bases only). */
  constraint?: ShapeConstraint
  label?: string
}

/** A term specification. */
export type TermSpec =
  | ({ kind: 'smooth'; feature: number; basis: 'pspline' | 'cyclic' | 'thin-plate' } & SmoothOptions)
  | { kind: 'linear'; feature: number; label?: string }
  | { kind: 'factor'; feature: number; lambda?: number; label?: string }
  | {
      kind: 'tensor'
      features: [number, number]
      k?: [number, number]
      degree?: number
      order?: number
      lambda?: [number, number]
      label?: string
    }

/** A P-spline smooth of one feature: `s(0)`, `s(1, { k: 20, constraint: 'increasing' })`. */
export function s(feature: number, options: SmoothOptions = {}): TermSpec {
  return { kind: 'smooth', basis: 'pspline', feature, ...options }
}

/** A cyclic P-spline smooth on `range` (its ends join smoothly): seasonal or angular effects. */
export function cyclic(feature: number, options: SmoothOptions & { range: readonly [number, number] }): TermSpec {
  return { kind: 'smooth', basis: 'cyclic', feature, ...options }
}

/** A thin-plate regression spline smooth of one feature (Wood, 2003), rank `k`. */
export function thinPlate(feature: number, options: SmoothOptions = {}): TermSpec {
  return { kind: 'smooth', basis: 'thin-plate', feature, ...options }
}

/** A tensor-product smooth of two features with one penalty per direction (Wood, 2006). */
export function te(
  a: number,
  b: number,
  options: Omit<Extract<TermSpec, { kind: 'tensor' }>, 'kind' | 'features'> = {},
) {
  return { kind: 'tensor', features: [a, b], ...options } as TermSpec
}

/** A linear (parametric) term βx, centred on the training mean. */
export function linearTerm(feature: number, label?: string): TermSpec {
  return { kind: 'linear', feature, label }
}

/** A factor (categorical) term on integer codes, sum-to-zero coded; `lambda` > 0 shrinks it as a random effect. */
export function factorTerm(feature: number, options: { lambda?: number; label?: string } = {}): TermSpec {
  return { kind: 'factor', feature, ...options }
}

/** A term built on training data: a block of the model matrix with its penalties. */
export type BuiltTerm = {
  readonly label: string
  readonly spec: TermSpec
  /** Features the term reads. */
  readonly features: number[]
  /** The raw basis at rows of X (row-major [m, d]) → [m, rawSize]. */
  raw(X: F64, m: number, d: number): F64
  readonly rawSize: number
  /** The constraint map [rawSize, size]: β_raw = Zβ. */
  readonly Z: F64
  readonly size: number
  /** Penalties [size, size], already mapped by Z; one smoothing parameter each. */
  readonly penalties: F64[]
  /** The same penalties on the raw coefficients, [rawSize, rawSize] (Sₖ before the constraint: penalties = ZᵀSₖZ). */
  readonly rawPenalties: F64[]
  /** Fixed smoothing parameters (NaN where the model selects them), one per penalty. */
  readonly fixedLambda: number[]
  /** Rows of a difference operator on the raw coefficients that a shape constraint wants ≥ 0 ([rows, rawSize]). */
  readonly shape?: { matrix: F64; rows: number }
  /** For `by` terms: the level (factor by) or null (numeric by). */
  readonly byLevel?: number | null
}

/** Householder reflection H = I − 2vvᵀ/vᵀv with Hc ∝ e₁; its last p − 1 columns span {β : cᵀβ = 0}. */
export function sumToZero(c: F64): F64 {
  const p = c.length
  const norm = Math.hypot(...c)
  const v = Float64Array.from(c)
  v[0] += (c[0] >= 0 ? 1 : -1) * norm
  const vv = v.reduce((a, b) => a + b * b, 0)
  const Z = new Float64Array(p * (p - 1))
  for (let i = 0; i < p; i++)
    for (let j = 1; j < p; j++) Z[i * (p - 1) + j - 1] = (i === j ? 1 : 0) - (vv > 0 ? (2 * v[i] * v[j]) / vv : 0)
  return Z
}

function identity(p: number): F64 {
  const I = new Float64Array(p * p)
  for (let i = 0; i < p; i++) I[i * p + i] = 1
  return I
}

/** ZᵀSZ for S [r, r] and Z [r, q]. */
export function congruence(S: F64, Z: F64, r: number, q: number): F64 {
  const SZ = new Float64Array(r * q)
  for (let i = 0; i < r; i++)
    for (let k = 0; k < r; k++) {
      const v = S[i * r + k]
      if (v === 0) continue
      for (let j = 0; j < q; j++) SZ[i * q + j] += v * Z[k * q + j]
    }
  const out = new Float64Array(q * q)
  for (let i = 0; i < r; i++)
    for (let a = 0; a < q; a++) for (let b = 0; b < q; b++) out[a * q + b] += Z[i * q + a] * SZ[i * q + b]
  return out
}

/** A times B for row-major [m, r] and [r, q]. */
export function times(A: F64, m: number, r: number, B: F64, q: number): F64 {
  const out = new Float64Array(m * q)
  for (let i = 0; i < m; i++)
    for (let k = 0; k < r; k++) {
      const v = A[i * r + k]
      if (v === 0) continue
      for (let j = 0; j < q; j++) out[i * q + j] += v * B[k * q + j]
    }
  return out
}

function column(X: F64, m: number, d: number, j: number): F64 {
  if (j < 0 || j >= d) throw new DomainError('gam', `gam: feature ${j} out of range (${d} features)`)
  return Float64Array.from({ length: m }, (_, i) => X[i * d + j])
}

function range(v: F64): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const u of v) {
    if (u < lo) lo = u
    if (u > hi) hi = u
  }
  return hi > lo ? [lo, hi] : [lo - 0.5, hi + 0.5]
}

/**
 * Build the terms of a model on training data X [n, d]; a factor `by` expands into one term per level.
 */
export function buildTerms(specs: readonly TermSpec[], X: F64, n: number, d: number): BuiltTerm[] {
  const out: BuiltTerm[] = []
  for (const spec of specs) {
    if (spec.kind === 'smooth' && spec.by !== undefined && spec.byFactor) {
      const levels = [...new Set(column(X, n, d, spec.by))].sort((a, b) => a - b)
      for (const level of levels) out.push(buildSmooth(spec, X, n, d, level))
    } else if (spec.kind === 'smooth') out.push(buildSmooth(spec, X, n, d, spec.by === undefined ? undefined : null))
    else if (spec.kind === 'linear') out.push(buildLinear(spec, X, n, d))
    else if (spec.kind === 'factor') out.push(buildFactor(spec, X, n, d))
    else out.push(buildTensor(spec, X, n, d))
  }
  return out
}

/** Column sums of a raw design [n, r] (the centring constraint's c). */
function columnSums(B: F64, n: number, r: number): F64 {
  const c = new Float64Array(r)
  for (let i = 0; i < n; i++) for (let j = 0; j < r; j++) c[j] += B[i * r + j]
  return c
}

function buildSmooth(
  spec: Extract<TermSpec, { kind: 'smooth' }>,
  X: F64,
  n: number,
  d: number,
  byLevel: number | null | undefined,
): BuiltTerm {
  const { feature, k = 10, degree = 3, order = 2 } = spec
  const xs = column(X, n, d, feature)
  const [lo, hi] = spec.range ?? range(xs)
  let basis: (v: F64) => F64
  let rawSize: number
  let S: F64
  if (spec.basis === 'cyclic') {
    rawSize = k
    basis = (v) => f64(cyclicBsplineBasis(fromData(v, [v.length]), lo, hi, k, degree))
    S = f64(cyclicDifferencePenalty(k, order))
  } else if (spec.basis === 'thin-plate') {
    const tprs = thinPlateRegressionBasis(fromData(xs, [n]), k)
    rawSize = k
    basis = (v) => f64(tprs.evaluate(fromData(v, [v.length])))
    S = f64(tprs.penalty)
  } else {
    const knots = uniformKnots(lo, hi, k - degree, degree)
    rawSize = bsplineCount(knots, degree)
    basis = (v) => f64(bsplineBasis(fromData(v, [v.length]), knots, degree))
    S = f64(differencePenalty(rawSize, order))
  }
  const multiplier = (Xm: F64, m: number, dm: number): F64 | null => {
    if (spec.by === undefined) return null
    const b = column(Xm, m, dm, spec.by)
    return byLevel === null || byLevel === undefined ? b : b.map((v) => (v === byLevel ? 1 : 0))
  }
  const raw = (Xm: F64, m: number, dm: number) => {
    const B = basis(column(Xm, m, dm, feature))
    const mult = multiplier(Xm, m, dm)
    if (mult) for (let i = 0; i < m; i++) for (let j = 0; j < rawSize; j++) B[i * rawSize + j] *= mult[i]
    return B
  }
  // A numeric `by` smooth is not centred (its level is identifiable); every other smooth sums to zero on the data.
  const centred = byLevel !== null
  const Z = centred ? sumToZero(columnSums(raw(X, n, d), n, rawSize)) : identity(rawSize)
  const size = centred ? rawSize - 1 : rawSize
  let shape: BuiltTerm['shape']
  if (spec.constraint) {
    if (spec.basis !== 'pspline') throw new DomainError('gam', 'gam: shape constraints need a P-spline basis')
    const ord = spec.constraint === 'increasing' || spec.constraint === 'decreasing' ? 1 : 2
    const sign = spec.constraint === 'decreasing' || spec.constraint === 'concave' ? -1 : 1
    const D = f64(differenceMatrix(rawSize, ord)).map((v) => sign * v)
    shape = { matrix: D, rows: rawSize - ord }
  }
  const byName = spec.by === undefined ? '' : byLevel === null ? `, by = x${spec.by}` : `, x${spec.by} = ${byLevel}`
  return {
    label: spec.label ?? `s(x${feature}${byName})`,
    spec,
    features: spec.by === undefined ? [feature] : [feature, spec.by],
    raw,
    rawSize,
    Z,
    size,
    penalties: [congruence(S, Z, rawSize, size)],
    rawPenalties: [S],
    fixedLambda: [spec.lambda ?? NaN],
    shape,
    byLevel,
  }
}

function buildLinear(spec: Extract<TermSpec, { kind: 'linear' }>, X: F64, n: number, d: number): BuiltTerm {
  const xs = column(X, n, d, spec.feature)
  const mean = xs.reduce((a, b) => a + b, 0) / n
  return {
    label: spec.label ?? `x${spec.feature}`,
    spec,
    features: [spec.feature],
    raw: (Xm, m, dm) => column(Xm, m, dm, spec.feature).map((v) => v - mean),
    rawSize: 1,
    Z: identity(1),
    size: 1,
    penalties: [],
    rawPenalties: [],
    fixedLambda: [],
  }
}

function buildFactor(spec: Extract<TermSpec, { kind: 'factor' }>, X: F64, n: number, d: number): BuiltTerm {
  const levels = [...new Set(column(X, n, d, spec.feature))].sort((a, b) => a - b)
  const L = levels.length
  if (L < 2) throw new DomainError('gam', `gam: factor x${spec.feature} has fewer than two levels`)
  const raw = (Xm: F64, m: number, dm: number) => {
    const v = column(Xm, m, dm, spec.feature)
    const B = new Float64Array(m * L)
    v.forEach((u, i) => {
      const j = levels.indexOf(u)
      if (j < 0) throw new DomainError('gam', `gam: level ${u} of x${spec.feature} was not in the training data`)
      B[i * L + j] = 1
    })
    return B
  }
  const Z = sumToZero(columnSums(raw(X, n, d), n, L))
  const ridge = spec.lambda !== undefined && spec.lambda > 0
  return {
    label: spec.label ?? `factor(x${spec.feature})`,
    spec,
    features: [spec.feature],
    raw,
    rawSize: L,
    Z,
    size: L - 1,
    penalties: ridge ? [congruence(identity(L), Z, L, L - 1)] : [],
    rawPenalties: ridge ? [identity(L)] : [],
    fixedLambda: ridge ? [spec.lambda!] : [],
  }
}

function buildTensor(spec: Extract<TermSpec, { kind: 'tensor' }>, X: F64, n: number, d: number): BuiltTerm {
  const [a, b] = spec.features
  const [ka, kb] = spec.k ?? [6, 6]
  const degree = spec.degree ?? 3
  const order = spec.order ?? 2
  const knots = [a, b].map((f, i) => {
    const [lo, hi] = range(column(X, n, d, f))
    return uniformKnots(lo, hi, (i === 0 ? ka : kb) - degree, degree)
  })
  const sizes = knots.map((t) => bsplineCount(t, degree))
  const rawSize = sizes[0] * sizes[1]
  const raw = (Xm: F64, m: number, dm: number) => {
    const Ba = bsplineBasis(fromData(column(Xm, m, dm, a), [m]), knots[0], degree)
    const Bb = bsplineBasis(fromData(column(Xm, m, dm, b), [m]), knots[1], degree)
    return f64(tensorProductBasis(Ba, Bb))
  }
  const [S1, S2] = tensorProductPenalties(differencePenalty(sizes[0], order), differencePenalty(sizes[1], order))
  const Z = sumToZero(columnSums(raw(X, n, d), n, rawSize))
  const size = rawSize - 1
  return {
    label: spec.label ?? `te(x${a}, x${b})`,
    spec,
    features: [a, b],
    raw,
    rawSize,
    Z,
    size,
    penalties: [congruence(f64(S1), Z, rawSize, size), congruence(f64(S2), Z, rawSize, size)],
    rawPenalties: [f64(S1), f64(S2)],
    fixedLambda: spec.lambda ? [...spec.lambda] : [NaN, NaN],
  }
}

/**
 * Rows of the feature matrix [m, d] that evaluate a term on a grid of its own feature(s): the grid ([m], or [m, 2] for
 * a tensor) in the term's feature columns, a `by` variable set to 1 (numeric) or to the term's level (factor), every
 * other column 0.
 */
export function termGridRows(term: BuiltTerm, grid: Tensor, d: number): { rows: F64; m: number } {
  const g = f64(grid)
  const m = grid.shape[0]
  const width = grid.shape.length === 2 ? grid.shape[1] : 1
  const rows = new Float64Array(m * d)
  const spec = term.spec
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < Math.min(width, spec.kind === 'tensor' ? 2 : 1); k++)
      rows[i * d + term.features[k]] = g[i * width + k]
    if (spec.kind === 'smooth' && spec.by !== undefined)
      rows[i * d + spec.by] = term.byLevel === null || term.byLevel === undefined ? 1 : term.byLevel
  }
  return { rows, m }
}

/** A term's basis on a grid: the raw columns, the constrained columns (raw · Z) and the penalties. */
export type TermBasis = {
  /** The raw basis [m, rawSize]: B-splines, cyclic B-splines or thin-plate columns before the sum-to-zero constraint. */
  raw: Tensor
  /** The model-matrix columns [m, size] = raw · Z: the basis after the constraint (what the coefficients multiply). */
  constrained: Tensor
  /** The constraint map Z [rawSize, size]. */
  Z: Tensor
  /** Each penalty on the raw coefficients, [rawSize, rawSize]. */
  rawPenalties: Tensor[]
  /** Each penalty on the constrained coefficients, [size, size]. */
  penalties: Tensor[]
}

/** The basis of a built term on a grid of its feature(s) ([m], or [m, 2] for a tensor); `d` is the feature count. */
export function termBasis(term: BuiltTerm, grid: Tensor, d: number): TermBasis {
  const { rows, m } = termGridRows(term, grid, d)
  const raw = term.raw(rows, m, d)
  const square = (S: F64, p: number) => fromData(S, [p, p])
  return {
    raw: fromData(raw, [m, term.rawSize]),
    constrained: fromData(times(raw, m, term.rawSize, term.Z, term.size), [m, term.size]),
    Z: fromData(term.Z, [term.rawSize, term.size]),
    rawPenalties: term.rawPenalties.map((S) => square(S, term.rawSize)),
    penalties: term.penalties.map((S) => square(S, term.size)),
  }
}
