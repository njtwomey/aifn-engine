/**
 * GAM terms: specifications (`s`, `cyclic`, `thinPlate`, `te`, `linearTerm`, `factorTerm`) and their construction on
 * training data as blocks of the model matrix with their penalties.
 *
 * A term's raw basis $\Bmat$ ($n \times r$) is constrained to sum to zero over the training data,
 * $\ones^\top\Bmat\betavec_{\text{raw}} = 0$, by writing $\betavec_{\text{raw}} = \Zmat\betavec$ with $\Zmat$
 * ($r \times (r - 1)$) the last $r - 1$ columns of a Householder reflection, so each block has one column fewer than
 * its basis and the intercept carries the mean (Wood, 2017, "Generalized Additive Models", 2nd ed., §5.4.1). Smooths,
 * tensor products and factors are constrained this way; a smooth multiplied by a numeric `by` variable is not (its
 * level is identifiable), and a linear term is centred on the training mean instead. A penalty $\Smat$ on the raw
 * coefficients becomes $\Zmat^\top\Smat\Zmat$ on the constrained ones. Matrices are row-major `Float64Array`s
 * throughout.
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
/**
 * A tensor's values, flattened in row-major order, as a fresh `Float64Array`.
 *
 * @param t The tensor to copy; it is not modified.
 * @returns Its entries in row-major order.
 */
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

/** A shape constraint on a P-spline smooth, imposed on its coefficients' differences. */
export type ShapeConstraint = 'increasing' | 'decreasing' | 'convex' | 'concave'

/** Options of a one-dimensional smooth. */
export type SmoothOptions = {
  /** Basis functions before the centring constraint (default 10); for a thin-plate smooth, the rank of the basis. */
  k?: number
  /** B-spline degree of a P-spline or cyclic smooth (default 3); not read by a thin-plate smooth. */
  degree?: number
  /**
   * Order of the difference penalty of a P-spline or cyclic smooth (default 2); a thin-plate smooth has its own
   * penalty.
   */
  order?: number
  /**
   * The interval the basis covers (default: the training range of the feature). Required by `cyclic`, whose ends join
   * there; not read by a thin-plate smooth.
   */
  range?: readonly [number, number]
  /** A fixed smoothing parameter $\lambda$ (default: selected with the model's method). */
  lambda?: number
  /**
   * The column of a feature to multiply the smooth by (a varying-coefficient term); with `byFactor`, that column holds
   * a factor's codes and one smooth is built per level.
   */
  by?: number
  /** Treat `by` as a factor: one centred smooth per level seen in training, each zero outside its level. */
  byFactor?: boolean
  /** A shape constraint (P-spline bases only; any other basis throws `DomainError` when built). */
  constraint?: ShapeConstraint
  /** The term's name in labels (default `s(x<feature>)`, with the `by` variable or level appended). */
  label?: string
}

/**
 * A term specification, as the constructors return it. `feature` is a column index of the data's $\Xmat$
 * ($n \times d$). A smooth names its basis and carries `SmoothOptions`; a linear term only a `label`; a factor term an
 * optional ridge `lambda` and a `label`.
 */
export type TermSpec =
  | ({ kind: 'smooth'; feature: number; basis: 'pspline' | 'cyclic' | 'thin-plate' } & SmoothOptions)
  | { kind: 'linear'; feature: number; label?: string }
  | { kind: 'factor'; feature: number; lambda?: number; label?: string }
  | {
      kind: 'tensor'
      /** The two feature columns. */
      features: [number, number]
      /** B-spline basis functions per direction before the constraint (default `[6, 6]`). */
      k?: [number, number]
      /** B-spline degree of both marginals (default 3). */
      degree?: number
      /** Order of both difference penalties (default 2). */
      order?: number
      /** Fixed $\lambda$ for the two directions' penalties (default: both selected with the model's method). */
      lambda?: [number, number]
      /** The term's name in labels (default `te(x<a>, x<b>)`). */
      label?: string
    }

/**
 * A P-spline smooth of one feature (Eilers and Marx, 1996): `k` B-splines (cubic by default) on equally spaced knots
 * over the training range, with a difference penalty (second order by default) on their coefficients, as pygam's `s`
 * and mgcv's `s(x, bs = "ps")`. Only the specification is made here; the basis is built when a model is fitted.
 *
 * @param feature The column of the data's $\Xmat$ the smooth is a function of.
 * @param options The basis size, degree, penalty order, range, a fixed $\lambda$, a `by` variable, a shape
 *   constraint and the label (see `SmoothOptions`).
 * @returns The term specification, to list in a model's `terms`.
 *
 * @example A smooth term recovers a sine from noisy points
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const model = gam({ terms: [s(0)] }).fit({ x, y })
 * const grid = tensor([0.5, 1.5, 3, 4.5, 5.5])
 * print('f(x) =', add(model.intercept, model.partial(0, grid).fit))
 * print('sin x =', sin(grid))
 *
 * @example A shape constraint keeps a wiggly fit increasing
 * const r = stream(0)
 * const x = uniform(r, 0, 3, { shape: [30, 1] })
 * const y = add(reshape(x, [30]), normals(r, 30, 0, 0.5))
 * const free = gam({ terms: [s(0, { lambda: 0.01 })] }).fit({ x, y })
 * const up = gam({ terms: [s(0, { lambda: 0.01, constraint: 'increasing' })] }).fit({ x, y })
 * const grid = reshape(linspace(0, 3, 61), [61, 1])
 * const falls = (model) => {
 *   const f = toArray(model.decide(grid))
 *   return f.slice(1).filter((v, i) => v < f[i]).length
 * }
 * print('grid steps where the fit falls: free', falls(free), 'increasing', falls(up))
 * print('active rows =', up.shape.active, 'violations =', up.shape.violations)
 */
export function s(feature: number, options: SmoothOptions = {}): TermSpec {
  return { kind: 'smooth', basis: 'pspline', feature, ...options }
}

/**
 * A cyclic P-spline smooth on `range`, whose value and derivatives at the two ends agree: seasonal or angular effects
 * (mgcv's `bs = "cp"`). The basis has `k` cyclic B-splines and a cyclic difference penalty.
 *
 * @param feature The column of the data's $\Xmat$ the smooth is a function of.
 * @param options As for `s`, with `range` required: the period's start and end, which the basis joins.
 * @returns The term specification.
 *
 * @example A periodic effect has the same value at both ends of its period
 * const r = stream(0)
 * const x = uniform(r, 0, 1, { shape: [40, 1] })
 * const y = add(cos(mul(2 * Math.PI, reshape(x, [40]))), normals(r, 40, 0, 0.2))
 * const model = gam({ terms: [cyclic(0, { range: [0, 1] })] }).fit({ x, y })
 * print('f(0) =', model.decide(tensor([[0]])), 'f(1) =', model.decide(tensor([[1]])))
 * print('f(0.5) =', model.decide(tensor([[0.5]])))
 */
export function cyclic(feature: number, options: SmoothOptions & { range: readonly [number, number] }): TermSpec {
  return { kind: 'smooth', basis: 'cyclic', feature, ...options }
}

/**
 * A thin-plate regression spline smooth of one feature (Wood, 2003), mgcv's default `s(x)`: the rank-`k` truncation
 * of the thin-plate spline's eigenbasis on the training values, with its own penalty. `degree`, `order` and `range`
 * are not read.
 *
 * @param feature The column of the data's $\Xmat$ the smooth is a function of.
 * @param options The rank `k` (default 10), a fixed $\lambda$, a `by` variable and the label (see `SmoothOptions`);
 *   a shape constraint is rejected when the model is built.
 * @returns The term specification.
 *
 * @example A rank-8 thin-plate smooth of a noisy sine
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.2))
 * const model = gam({ terms: [thinPlate(0, { k: 8 })] }).fit({ x, y })
 * print('f(1.5), f(4.5) =', model.decide(tensor([[1.5], [4.5]])))
 * print('sin 1.5, sin 4.5 =', [Math.sin(1.5), Math.sin(4.5)])
 * print('edf =', model.edf)
 */
export function thinPlate(feature: number, options: SmoothOptions = {}): TermSpec {
  return { kind: 'smooth', basis: 'thin-plate', feature, ...options }
}

/**
 * A tensor-product smooth of two features with one penalty per direction (Wood, 2006), as mgcv's `te` and pygam's
 * `te`: the row-wise Kronecker product of two cubic B-spline bases, each on its feature's training range, so each
 * direction gets its own smoothing parameter.
 *
 * @param a The column of the first feature.
 * @param b The column of the second feature.
 * @param options The basis size per direction, degree, penalty order, fixed $\lambda$ pair and label (see
 *   `TermSpec`).
 * @returns The term specification.
 *
 * @example The interaction x0 x1, which no sum of one-dimensional smooths can fit
 * const r = stream(0)
 * const x = uniform(r, -1, 1, { shape: [40, 2] })
 * const y = add(tensor(toArray(x).map(([u, v]) => u * v)), normals(r, 40, 0, 0.1))
 * const model = gam({ terms: [te(0, 1, { k: [4, 4] })] }).fit({ x, y })
 * print('label =', model.labels[0], 'lambdas =', model.lambdas)
 * print('f(1/2, 1/2), f(1/2, -1/2) =', model.decide(tensor([[0.5, 0.5], [0.5, -0.5]])))
 */
export function te(
  a: number,
  b: number,
  options: Omit<Extract<TermSpec, { kind: 'tensor' }>, 'kind' | 'features'> = {},
) {
  return { kind: 'tensor', features: [a, b], ...options } as TermSpec
}

/**
 * A linear (parametric, unpenalised) term $\beta (x - \bar{x})$, centred on the training mean $\bar{x}$ so the
 * intercept keeps the mean.
 *
 * @param feature The column of the data's $\Xmat$.
 * @param label The term's name in labels (default `x<feature>`).
 * @returns The term specification.
 *
 * @example The slope of a straight line
 * const r = stream(0)
 * const x = uniform(r, 0, 1, { shape: [40, 1] })
 * const y = add(mul(2, reshape(x, [40])), normals(r, 40, 0, 0.1))
 * const model = gam({ terms: [linearTerm(0)] }).fit({ x, y })
 * print('slope =', toArray(model.coefficients)[1], 'edf =', model.edf)
 */
export function linearTerm(feature: number, label?: string): TermSpec {
  return { kind: 'linear', feature, label }
}

/**
 * A factor (categorical) term: one indicator column per level seen in training, sum-to-zero coded. Unpenalised by
 * default; `lambda` $> 0$ adds the fixed ridge penalty $\lambda \lVert \betavec_{\text{raw}} \rVert^2$ on the
 * level effects, which shrinks them towards each other as a random effect would. Building it throws `DomainError` for
 * fewer than two levels, and predicting at a level not seen in training throws too.
 *
 * @param feature The column of the data's $\Xmat$ holding the level codes (any numbers; each distinct value is a
 *   level).
 * @param options `lambda`, the fixed ridge weight (none when left out or $\le 0$), and `label`, the term's name
 *   (default `factor(x<feature>)`).
 * @returns The term specification.
 *
 * @example Group means, and the same means shrunk towards the grand mean
 * const x = tensor([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
 * const y = tensor([1, 2, 3, 5, 6, 7, 2, 2, 2])
 * const fixed = gam({ terms: [factorTerm(0)] }).fit({ x, y })
 * const shrunk = gam({ terms: [factorTerm(0, { lambda: 5 })] }).fit({ x, y })
 * print('group means =', fixed.decide(tensor([[0], [1], [2]])))
 * print('shrunk =', shrunk.decide(tensor([[0], [1], [2]])))
 */
export function factorTerm(feature: number, options: { lambda?: number; label?: string } = {}): TermSpec {
  return { kind: 'factor', feature, ...options }
}

/** A term built on training data: a block of the model matrix with its penalties. */
export type BuiltTerm = {
  /** The term's name, e.g. `s(x0)`, `s(x0, x1 = 2)`, `te(x0, x1)`. */
  readonly label: string
  /** The specification it was built from. */
  readonly spec: TermSpec
  /** Features the term reads (for a `by` smooth, the feature and then the `by` column). */
  readonly features: number[]
  /**
   * The raw basis at the rows of a feature matrix ($m \times d$, row-major): $m \times$ `rawSize`, row-major. A `by`
   * smooth's rows are multiplied by the `by` value (or its level's indicator).
   */
  raw(X: F64, m: number, d: number): F64
  /** Columns of the raw basis. */
  readonly rawSize: number
  /** The constraint map $\Zmat$ (`rawSize` $\times$ `size`, row-major): $\betavec_{\text{raw}} = \Zmat\betavec$. */
  readonly Z: F64
  /** Columns of the term's block of the model matrix, after the constraint. */
  readonly size: number
  /** Penalties (`size` $\times$ `size`), already mapped by $\Zmat$; one smoothing parameter each. */
  readonly penalties: F64[]
  /**
   * The same penalties $\Smat_k$ on the raw coefficients (`rawSize` $\times$ `rawSize`), before the constraint:
   * `penalties` are $\Zmat^\top\Smat_k\Zmat$.
   */
  readonly rawPenalties: F64[]
  /** Fixed smoothing parameters (NaN where the model selects them), one per penalty. */
  readonly fixedLambda: number[]
  /**
   * Rows of a difference operator on the raw coefficients that a shape constraint wants $\ge 0$ (`rows` $\times$
   * `rawSize`, row-major); absent without a constraint.
   */
  readonly shape?: { matrix: F64; rows: number }
  /** For `by` terms: the level (factor `by`) or null (numeric `by`); undefined for every other term. */
  readonly byLevel?: number | null
}

/**
 * The constraint map of $\cvec^\top\betavec = 0$: the Householder reflection
 * $\Hmat = \Imat - 2\vvec\vvec^\top / \vvec^\top\vvec$ with $\Hmat\cvec \propto \evec_1$, whose last $p - 1$
 * columns span $\{\betavec : \cvec^\top\betavec = 0\}$. For $\cvec = \zeros$ they are the last $p - 1$
 * columns of $\Imat$.
 *
 * @param c The constraint vector $\cvec$ of $p$ values (here the column sums of a raw basis); not modified.
 * @returns $\Zmat$, the last $p - 1$ columns of $\Hmat$, as a row-major $p \times (p - 1)$ array.
 */
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

/**
 * The identity matrix.
 *
 * @param p Its number of rows and columns.
 * @returns $\Imat$ ($p \times p$), row-major.
 */
function identity(p: number): F64 {
  const I = new Float64Array(p * p)
  for (let i = 0; i < p; i++) I[i * p + i] = 1
  return I
}

/**
 * The congruence $\Zmat^\top\Smat\Zmat$: a penalty on raw coefficients carried to the constrained ones.
 *
 * @param S $\Smat$, a row-major $r \times r$ array; not modified.
 * @param Z $\Zmat$, a row-major $r \times q$ array; not modified.
 * @param r The rows of $\Zmat$ (and size of $\Smat$).
 * @param q The columns of $\Zmat$.
 * @returns $\Zmat^\top\Smat\Zmat$, a row-major $q \times q$ array.
 */
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

/**
 * The matrix product $\Amat\Bmat$ of row-major arrays, skipping zero entries of $\Amat$ (bases are sparse).
 *
 * @param A $\Amat$, $m \times r$; not modified.
 * @param m The rows of $\Amat$.
 * @param r The columns of $\Amat$ and rows of $\Bmat$.
 * @param B $\Bmat$, $r \times q$; not modified.
 * @param q The columns of $\Bmat$.
 * @returns $\Amat\Bmat$, a row-major $m \times q$ array.
 */
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

/**
 * One column of a row-major feature matrix; throws `DomainError` when there is no such column.
 *
 * @param X The feature matrix, $m \times d$, row-major; not modified.
 * @param m Its number of rows.
 * @param d Its number of columns (features).
 * @param j The column wanted, from 0.
 * @returns Column `j`, $m$ values.
 */
function column(X: F64, m: number, d: number, j: number): F64 {
  if (j < 0 || j >= d) throw new DomainError('gam', `gam: feature ${j} out of range (${d} features)`)
  return Float64Array.from({ length: m }, (_, i) => X[i * d + j])
}

/**
 * The range of a feature's training values, widened to $[v - 0.5, v + 0.5]$ when every value is $v$ (so a basis has
 * an interval to cover).
 *
 * @param v The values.
 * @returns `[lo, hi]` with `hi > lo`.
 */
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
 * Build the terms of a model on the training data: bases, constraint maps and penalties. A smooth with a factor `by`
 * expands into one term per level, in increasing order of level.
 *
 * @param specs The term specifications, in model order.
 * @param X The training features $\Xmat$, $n \times d$, row-major; not modified.
 * @param n The number of training rows.
 * @param d The number of features.
 * @returns The built terms, in the order their blocks take in the model matrix.
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

/**
 * Column sums of a raw design: the centring constraint's $\cvec = \Bmat^\top\ones$.
 *
 * @param B The raw design $\Bmat$, $n \times r$, row-major; not modified.
 * @param n Its number of rows.
 * @param r Its number of columns.
 * @returns The $r$ column sums.
 */
function columnSums(B: F64, n: number, r: number): F64 {
  const c = new Float64Array(r)
  for (let i = 0; i < n; i++) for (let j = 0; j < r; j++) c[j] += B[i * r + j]
  return c
}

/**
 * Build a one-dimensional smooth (P-spline, cyclic or thin-plate) on the training data, with its `by` multiplier,
 * centring and shape-constraint rows. Throws `DomainError` for a shape constraint on a basis other than P-splines.
 *
 * @param spec The smooth's specification.
 * @param X The training features, $n \times d$, row-major; not modified.
 * @param n The number of training rows.
 * @param d The number of features.
 * @param byLevel Undefined without a `by` variable; null for a numeric `by` (the smooth is multiplied by it and not
 *   centred); a level for a factor `by` (multiplied by that level's indicator, and centred).
 * @returns The built term.
 */
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

/**
 * Build a linear term: the feature minus its training mean, one unpenalised column.
 *
 * @param spec The term's specification.
 * @param X The training features, $n \times d$, row-major; not modified.
 * @param n The number of training rows.
 * @param d The number of features.
 * @returns The built term.
 */
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

/**
 * Build a factor term: indicator columns of the training levels, sum-to-zero constrained, with a ridge penalty when
 * `spec.lambda` $> 0$. Throws `DomainError` for fewer than two levels.
 *
 * @param spec The term's specification.
 * @param X The training features, $n \times d$, row-major; not modified.
 * @param n The number of training rows.
 * @param d The number of features.
 * @returns The built term; its `raw` throws `DomainError` for a level not seen in training.
 */
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

/**
 * Build a tensor-product smooth: B-spline marginals on each feature's training range, their row-wise Kronecker
 * product, the two directional difference penalties, and the centring constraint.
 *
 * @param spec The term's specification.
 * @param X The training features, $n \times d$, row-major; not modified.
 * @param n The number of training rows.
 * @param d The number of features.
 * @returns The built term, with two penalties.
 */
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
 * Rows of a feature matrix ($m \times d$) that evaluate a term on a grid of its own feature(s): the grid in the term's
 * feature columns, a `by` variable set to 1 (numeric) or to the term's level (factor), every other column 0.
 *
 * @param term The built term.
 * @param grid The grid: $m$ values, or $m \times 2$ for a tensor product (a one-dimensional term reads only the first
 *   column of a two-column grid).
 * @param d The number of features the term was built on.
 * @returns `rows`, the row-major $m \times d$ feature matrix, and `m`, its number of rows.
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

/** A term's basis on a grid: the raw columns, the constrained columns $\Bmat\Zmat$ and the penalties. */
export type TermBasis = {
  /**
   * The raw basis $\Bmat$ ($m \times$ `rawSize`): B-splines, cyclic B-splines or thin-plate columns before the
   * sum-to-zero constraint.
   */
  raw: Tensor
  /**
   * The model-matrix columns $\Bmat\Zmat$ ($m \times$ `size`): the basis after the constraint (what the coefficients
   * multiply).
   */
  constrained: Tensor
  /** The constraint map $\Zmat$ (`rawSize` $\times$ `size`). */
  Z: Tensor
  /** Each penalty on the raw coefficients, `rawSize` $\times$ `rawSize`. */
  rawPenalties: Tensor[]
  /** Each penalty on the constrained coefficients, `size` $\times$ `size`. */
  penalties: Tensor[]
}

/**
 * The basis of a built term on a grid of its feature(s): what a figure of a term's basis functions and penalties
 * draws. Get a built term from a design (`gamDesign(...).terms`) or a fitted model (`model.terms`).
 *
 * @param term The built term.
 * @param grid The grid: $m$ values, or $m \times 2$ for a tensor product.
 * @param d The number of features the term was built on.
 * @returns The raw and constrained columns, $\Zmat$ and the penalties, as tensors.
 *
 * @example Six cubic B-splines, and one column fewer after the constraint
 * const x = reshape(linspace(0, 1, 20), [20, 1])
 * const A = gamDesign([s(0, { k: 6 })], x)
 * const b = termBasis(A.terms[0], tensor([0, 0.5, 1]), 1)
 * print('raw =', b.raw.shape, 'constrained =', b.constrained.shape, 'Z =', b.Z.shape)
 * print('B-splines sum to 1:', sum(b.raw, 1))
 * print('penalty =', b.penalties[0].shape)
 */
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
