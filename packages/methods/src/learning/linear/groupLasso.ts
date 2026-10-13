/**
 * The group lasso (Yuan and Lin, 2006) by block coordinate descent, one fit at a time or along a warm-started path of
 * penalties, in the scaling of `elasticNet`.
 *
 * The columns are partitioned into groups $g$ of sizes $p_g$, and the fit minimises
 *
 * $$
 * \frac{1}{2n}\norm{\yvec - \Xmat\wvec - b\ones}^2 + \alpha\sum_g \sqrt{p_g}\,\norm{\wvec_g}_2,
 * $$
 *
 * which sets whole groups to zero together (the $\ell_2$ norm is not squared, so it has a corner at $\wvec_g = \zeros$)
 * and, with every group a single column, is the lasso. A group is zero exactly when
 * $\norm{\zvec_g}_2 \le n\alpha\sqrt{p_g}$, with $\zvec_g = \Xmat_g^\top\rvec_{(g)}$ the correlation of its columns
 * with the partial residual $\rvec_{(g)}$ that leaves the group out. Otherwise its block is solved exactly, not by the
 * orthonormal-group shortcut of Yuan and Lin: with $\Xmat_g^\top\Xmat_g = \Qmat\operatorname{diag}(e_i)\Qmat^\top$,
 * $\cvec = \Qmat^\top\zvec_g$ and $\lambda = n\alpha\sqrt{p_g}$, the block minimiser is
 * $\wvec_g = \Qmat\operatorname{diag}(c_i / (e_i + \nu))$ with $\nu = \lambda / \norm{\wvec_g}$ the root of
 * $\sum_i (c_i\nu / (e_i + \nu))^2 = \lambda^2$, found by safeguarded Newton steps. The sweeps work on the Gram
 * matrix, and a run stops on scikit-learn's two tests, as `elasticNet`: a sweep's largest change below `tolerance`
 * times the largest weight, then a duality gap below `tolerance` times $\yvec^\top\yvec$; the gap rescales the residual
 * by $\min(1, \lambda_1 / \max_g \norm{(\Xmat^\top\rvec)_g} / \sqrt{p_g})$ into the dual feasible set.
 */

import { eigh } from 'aifn-compute/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Decides, Estimator, Fitted, Supervised } from 'aifn-compute/learning/estimators'
import { defineModel, matrixShape } from 'aifn-compute/learning/estimators'
import { bool, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { centre, gram, type Gram } from './elasticNet'

/** The groups of a design: the columns of each, and the eigendecomposition of each diagonal block of the Gram. */
interface Blocks {
  /** The group labels, in order of first appearance. */
  readonly labels: number[]
  /** The columns of each group, increasing. */
  readonly members: number[][]
  /** The eigenvalues $e_i \ge 0$ of each block $\Xmat_g^\top\Xmat_g$. */
  readonly e: Float64Array[]
  /** The eigenvectors of each block, $p_g \times p_g$ row-major, as columns. */
  readonly Q: Float64Array[]
}

/**
 * The columns of each group from a label per column. Throws `ShapeError` when there is not one label per column, and
 * `DomainError` for a label that is not an integer.
 *
 * @param groups One integer label per column; columns sharing a label form a group. Left out, each column is a group.
 * @param d The number of columns.
 * @param where The caller's name, for error messages.
 * @returns The labels in order of first appearance and the columns of each.
 */
function partition(
  groups: readonly number[] | undefined,
  d: number,
  where: string,
): { labels: number[]; members: number[][] } {
  if (groups === undefined)
    return { labels: Array.from({ length: d }, (_, j) => j), members: Array.from({ length: d }, (_, j) => [j]) }
  if (groups.length !== d) throw new ShapeError(where, `${where}: ${groups.length} group labels for ${d} columns`)
  const labels: number[] = []
  const members: number[][] = []
  groups.forEach((label, j) => {
    if (!Number.isInteger(label)) throw new DomainError(where, `${where}: group labels must be integers`)
    let g = labels.indexOf(label)
    if (g < 0) {
      g = labels.length
      labels.push(label)
      members.push([])
    }
    members[g].push(j)
  })
  return { labels, members }
}

/**
 * Decompose the diagonal blocks of the Gram matrix.
 *
 * @param g The problem in Gram form.
 * @param groups One integer label per column (each column its own group when left out).
 * @param where The caller's name, for error messages.
 * @returns The groups and their block decompositions.
 */
function blocks(g: Gram, groups: readonly number[] | undefined, where: string): Blocks {
  const { d, H } = g
  const { labels, members } = partition(groups, d, where)
  const e: Float64Array[] = []
  const Q: Float64Array[] = []
  for (const cols of members) {
    const p = cols.length
    const B = new Float64Array(p * p)
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) B[a * p + b] = H[cols[a] * d + cols[b]]
    const { values, vectors } = eigh(fromData(B, [p, p]))
    e.push(Float64Array.from(dense.data(values), (v) => Math.max(0, v)))
    Q.push(Float64Array.from(dense.data(vectors)))
  }
  return { labels, members, e, Q }
}

/**
 * Solve one block: minimise $\frac12\wvec^\top\Bmat\wvec - \zvec^\top\wvec + \lambda\norm{\wvec}$ over the group's
 * weights, $\Bmat = \Qmat\operatorname{diag}(e_i)\Qmat^\top$, written into `out`.
 *
 * @param z The correlations with the partial residual, $p_g$ values.
 * @param e The eigenvalues of the block.
 * @param Q The eigenvectors of the block, as columns.
 * @param lambda The penalty $\lambda = n\alpha\sqrt{p_g}$.
 * @param out The group's weights, overwritten.
 */
function solveBlock(z: Float64Array, e: Float64Array, Q: Float64Array, lambda: number, out: Float64Array): void {
  const p = z.length
  let zz = 0
  for (const v of z) zz += v * v
  if (zz <= lambda * lambda) {
    out.fill(0)
    return
  }
  const c = new Float64Array(p)
  for (let i = 0; i < p; i++) for (let a = 0; a < p; a++) c[i] += Q[a * p + i] * z[a]
  // h(nu) = sum (c_i nu / (e_i + nu))^2 - lambda^2 rises from -lambda^2 at 0 to |z|^2 - lambda^2 > 0: one root.
  const h = (nu: number): [number, number] => {
    let f = -lambda * lambda
    let df = 0
    for (let i = 0; i < p; i++) {
      const r = nu / (e[i] + nu)
      f += c[i] * c[i] * r * r
      df += (2 * c[i] * c[i] * r * e[i]) / ((e[i] + nu) * (e[i] + nu))
    }
    return [f, df]
  }
  // A bracket [lo, hi] of the root: once nu >= lambda max(e) / (|z| - lambda), every ratio nu / (e_i + nu) is at
  // least lambda / |z|, so h(nu) >= 0. The doubling only guards against rounding.
  let maxE = 0
  for (const v of e) maxE = Math.max(maxE, v)
  let lo = 0
  let hi = Math.max(1e-300, (lambda * (maxE + 1)) / (Math.sqrt(zz) - lambda))
  while (h(hi)[0] < 0) {
    lo = hi
    hi *= 2
  }
  let nu = hi
  for (let it = 0; it < 200; it++) {
    const [f, df] = h(nu)
    if (f === 0) break
    if (f > 0) hi = nu
    else lo = nu
    let next = df > 0 ? nu - f / df : NaN
    if (!(next > lo && next < hi)) next = (lo + hi) / 2
    if (Math.abs(next - nu) <= 1e-15 * nu) {
      nu = next
      break
    }
    nu = next
  }
  out.fill(0)
  for (let i = 0; i < p; i++) {
    const s = c[i] / (e[i] + nu)
    for (let a = 0; a < p; a++) out[a] += Q[a * p + i] * s
  }
}

/** The outcome of a block-coordinate-descent run. */
interface GroupSweeps {
  /** Sweeps made. */
  readonly steps: number
  /** The duality gap at the end, of the objective scaled by $n$. */
  readonly gap: number
  /** Whether the gap fell below the tolerance before `maxSteps`. */
  readonly converged: boolean
}

/**
 * The duality gap of the group lasso in Gram form, of the objective scaled by $n$:
 * $\frac12\norm{\rvec}^2 + \lambda_1\sum_g\sqrt{p_g}\norm{\wvec_g}$ with $\lambda_1 = n\alpha$.
 *
 * @param g The problem in Gram form.
 * @param b The groups.
 * @param w The weights.
 * @param l1 The penalty $\lambda_1 = n\alpha$.
 * @returns The gap, non-negative up to rounding.
 */
function groupGap(g: Gram, b: Blocks, w: Float64Array, l1: number): number {
  const { d, H, q, yy } = g
  const grad = new Float64Array(d)
  let wHw = 0
  let qw = 0
  for (let j = 0; j < d; j++) {
    let hw = 0
    for (let k = 0; k < d; k++) hw += H[j * d + k] * w[k]
    grad[j] = q[j] - hw
    wHw += w[j] * hw
    qw += q[j] * w[j]
  }
  let dual = 0
  let penalty = 0
  for (const cols of b.members) {
    let gg = 0
    let ww = 0
    for (const j of cols) {
      gg += grad[j] * grad[j]
      ww += w[j] * w[j]
    }
    const root = Math.sqrt(cols.length)
    dual = Math.max(dual, Math.sqrt(gg) / root)
    penalty += root * Math.sqrt(ww)
  }
  const rr = Math.max(0, yy - 2 * qw + wHw)
  const c = dual > l1 ? l1 / dual : 1
  return 0.5 * rr * (1 + c * c) + l1 * penalty - c * (yy - qw)
}

/**
 * Cyclic block coordinate descent on the group lasso in Gram form, in place, with the stopping rule of
 * `coordinateSweeps`.
 *
 * @param g The problem in Gram form.
 * @param b The groups.
 * @param w The starting weights, $d$ values; overwritten with the solution.
 * @param l1 The penalty $\lambda_1 = n\alpha$.
 * @param tolerance The relative tolerance of both tests.
 * @param maxSteps The most sweeps.
 * @returns The sweeps made, the final gap and whether it converged.
 */
function groupSweeps(
  g: Gram,
  b: Blocks,
  w: Float64Array,
  l1: number,
  tolerance: number,
  maxSteps: number,
): GroupSweeps {
  const { d, H, q, yy } = g
  const Hw = new Float64Array(d)
  const refresh = (): void => {
    Hw.fill(0)
    for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) Hw[j] += H[j * d + k] * w[k]
  }
  refresh()
  const gapTolerance = tolerance * yy
  let gap = Infinity
  for (let step = 1; step <= maxSteps; step++) {
    let wMax = 0
    let dMax = 0
    b.members.forEach((cols, gi) => {
      const p = cols.length
      const z = new Float64Array(p)
      for (let a = 0; a < p; a++) {
        const j = cols[a]
        let s = q[j] - Hw[j]
        for (let c = 0; c < p; c++) s += H[j * d + cols[c]] * w[cols[c]]
        z[a] = s
      }
      const next = new Float64Array(p)
      solveBlock(z, b.e[gi], b.Q[gi], l1 * Math.sqrt(p), next)
      for (let a = 0; a < p; a++) {
        const j = cols[a]
        const delta = next[a] - w[j]
        if (delta !== 0) {
          for (let k = 0; k < d; k++) Hw[k] += H[k * d + j] * delta
          w[j] = next[a]
        }
        dMax = Math.max(dMax, Math.abs(delta))
        wMax = Math.max(wMax, Math.abs(next[a]))
      }
    })
    if (wMax === 0 || dMax / wMax < tolerance || step === maxSteps) {
      gap = groupGap(g, b, w, l1)
      if (gap <= gapTolerance) return { steps: step, gap, converged: true }
      refresh()
    }
  }
  return { steps: maxSteps, gap, converged: false }
}

/** Hyperparameters of `groupLasso`. */
export interface GroupLassoParams {
  /** The penalty strength $\alpha > 0$. */
  alpha: number
  /**
   * One integer label per column; columns sharing a label form a group, of any size. Default every column its own
   * group, which is the lasso.
   */
  groups?: readonly number[]
  /** Fit an unpenalised intercept (default true). */
  intercept?: boolean
  /** The relative tolerance of the stopping tests (default $10^{-10}$). */
  tolerance?: number
  /** The most sweeps (default 10000). */
  maxSteps?: number
  /** Starting weights, $d$ values (default zeros). */
  warmStart?: Tensor
}

/** A fitted group lasso $y = \xvec^\top\wvec + b$. */
export interface GroupLassoModel extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor> {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'group-lasso'
  /** The weights $\wvec$, $d$ values; a group the penalty removed is exactly 0. */
  readonly weights: Tensor
  /** The intercept $b$ (0 without an intercept). */
  readonly intercept: number
  /** The penalty strength $\alpha$. */
  readonly alpha: number
  /** The group labels, in order of first appearance. */
  readonly groups: number[]
  /** $\norm{\wvec_g}_2$ for each group, in the order of `groups`. */
  readonly groupNorms: number[]
  /** The number of groups that are not zero. */
  readonly activeGroups: number
  /** The objective at the solution, in the $\frac{1}{2n}$ scaling above. */
  readonly objective: number
  /** The sweeps of block coordinate descent made. */
  readonly steps: number
  /** The duality gap at the end, in the scaling of `objective`. */
  readonly gap: number
  /** Whether the stopping tests passed before `maxSteps`. */
  readonly converged: boolean
}

/**
 * Check the penalty strength.
 *
 * @param alpha The penalty strength; must be positive and finite.
 * @param where The caller's name, for error messages.
 */
function checkAlpha(alpha: number, where: string): void {
  if (!(alpha > 0) || !Number.isFinite(alpha))
    throw new DomainError(where, `${where}: alpha must be positive and finite`)
}

/**
 * The group lasso of Yuan and Lin (2006) by block coordinate descent: $\wvec$ and $b$ minimise
 * $\frac{1}{2n}\norm{\yvec - \Xmat\wvec - b\ones}^2 + \alpha\sum_g\sqrt{p_g}\norm{\wvec_g}_2$, which selects whole
 * groups of columns (the dummy columns of a factor, the basis of one input's expansion) in or out together. Each block
 * is solved exactly, so the groups need not be orthonormal; within a group that is in, the weights are shrunk
 * together but none is zero by itself. With every group a single column it is `elasticNet`'s lasso. Standardise the
 * columns first for the penalty to treat them alike. Throws `DomainError` at once for $\alpha \le 0$, and from `fit`
 * `ShapeError` for mismatched shapes or a `groups` or `warmStart` of the wrong length, and `DomainError` for a label
 * that is not an integer. A run that reaches `maxSteps` returns its last weights with `converged` false.
 *
 * Capabilities: `forward` and `decide` (the prediction $\xvec^\top\wvec + b$, $m$ values).
 *
 * @param params The penalty `alpha`, the `groups`, the intercept, the stopping rule and a warm start, as
 *   `GroupLassoParams`.
 * @returns The estimator: `fit({ x, y })`, with `x` $n \times d$ and `y` $n$ values, returns a `GroupLassoModel`.
 *
 * @example A group of two noise columns is dropped whole
 * // Columns 0-1 form group 0 and carry the signal; columns 2-3 (group 1) are noise.
 * const x = normals(stream(0), [100, 4])
 * const y = tensor(toArray(x).map((r, i) => 2 * r[0] - r[1] + 0.2 * Math.sin(i)))
 * const fit = groupLasso({ alpha: 0.2, groups: [0, 0, 1, 1] }).fit({ x, y })
 * print('weights =', fit.weights)
 * print('group norms =', fit.groupNorms, ' active groups =', fit.activeGroups)
 *
 * @example Singleton groups give the lasso
 * const x = normals(stream(0), [80, 3])
 * const y = tensor(toArray(x).map((r) => r[0] - 0.5 * r[1] + 0.1 * r[2]))
 * print('group lasso:', groupLasso({ alpha: 0.1, groups: [0, 1, 2] }).fit({ x, y }).weights)
 * print('lasso:      ', elasticNet({ alpha: 0.1 }).fit({ x, y }).weights)
 */
export function groupLasso(params: GroupLassoParams): Estimator<Supervised<Tensor, Tensor>, GroupLassoModel> {
  const { alpha, groups, intercept = true, tolerance = 1e-10, maxSteps = 10000, warmStart } = params
  checkAlpha(alpha, 'groupLasso')
  return {
    name: 'group-lasso',
    params: { alpha, groups: groups && [...groups], intercept, tolerance, maxSteps },
    fit({ x, y }) {
      const c = centre(x, y, intercept, 'groupLasso')
      const { n, d } = c
      const g = gram(c)
      const b = blocks(g, groups, 'groupLasso')
      const w = new Float64Array(d)
      if (warmStart) {
        const start = dense.data(warmStart)
        if (start.length !== d)
          throw new ShapeError('groupLasso', `groupLasso: warmStart has ${start.length} values for ${d} features`)
        w.set(start)
      }
      const run = groupSweeps(g, b, w, n * alpha, tolerance, maxSteps)
      let bias = c.yMean
      for (let j = 0; j < d; j++) bias -= c.xMean[j] * w[j]
      let rss = 0
      for (let i = 0; i < n; i++) {
        let r = c.y[i]
        for (let j = 0; j < d; j++) r -= c.x[i * d + j] * w[j]
        rss += r * r
      }
      const groupNorms = b.members.map((cols) => Math.sqrt(cols.reduce((s, j) => s + w[j] * w[j], 0)))
      const penalty = groupNorms.reduce((s, v, gi) => s + Math.sqrt(b.members[gi].length) * v, 0)
      const forward = (input: Tensor): Tensor => {
        const [m, cols] = matrixShape(input, 'groupLasso.forward')
        if (cols !== d) throw new ShapeError('groupLasso', `groupLasso: fitted on ${d} features, given ${cols}`)
        const Z = dense.data(input)
        const out = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          let f = bias
          for (let j = 0; j < d; j++) f += Z[i * d + j] * w[j]
          out[i] = f
        }
        return fromData(out, [m])
      }
      return {
        kind: 'model',
        name: 'group-lasso',
        weights: fromData(w, [d]),
        intercept: bias,
        alpha,
        groups: b.labels,
        groupNorms,
        activeGroups: groupNorms.filter((v) => v !== 0).length,
        objective: rss / (2 * n) + alpha * penalty,
        steps: run.steps,
        gap: run.gap / n,
        converged: run.converged,
        forward,
        decide: forward,
      }
    },
  }
}

/** Options of `groupLassoPath`. */
export interface GroupLassoPathOptions {
  /** One integer label per column; columns sharing a label form a group (default every column its own group). */
  groups?: readonly number[]
  /** The penalties, any order (they are fitted from largest to smallest); default the log-spaced grid below. */
  alphas?: readonly number[]
  /** The number of penalties of the default grid (default 81). */
  count?: number
  /** The decades the default grid spans below $\alpha_{\max}$ (default 4). */
  span?: number
  /** Fit an unpenalised intercept, centring the data (default true). */
  intercept?: boolean
  /** The relative tolerance of each fit's stopping tests (default $10^{-10}$). */
  tolerance?: number
  /** The most sweeps of each fit (default 10000). */
  maxSteps?: number
}

/** A group-lasso path: one fit per penalty, the largest penalty first. */
export interface GroupLassoPath {
  /** The penalties $\alpha$, decreasing. */
  readonly alphas: number[]
  /** The weights, $d \times k$: column $t$ is the fit at `alphas[t]`. */
  readonly coef: Tensor
  /** The group norms, $G \times k$: row $g$ is the norm of group `groups[g]` along the path. */
  readonly groupNorms: Tensor
  /** The group labels, in order of first appearance. */
  readonly groups: number[]
  /** The intercept of each fit (0 without an intercept). */
  readonly intercept: number[]
  /** The number of groups that are not zero in each fit. */
  readonly activeGroups: number[]
  /** The sweeps each fit took from the previous one's weights. */
  readonly steps: number[]
  /** Whether every fit met its stopping tests. */
  readonly converged: boolean
}

/**
 * The group-lasso regularisation path: the fits of `groupLasso` at a decreasing sequence of penalties, each started
 * from the previous one's weights, as `elasticNetPath` is for the lasso. The default grid runs from
 * $\alpha_{\max} = \max_g \norm{\Xmat_g^\top\yvec}_2 / (n\sqrt{p_g})$ (on the centred data), the smallest penalty
 * at which every group is zero, down `span` decades in `count` log-spaced steps. Whole groups join the path as the
 * penalty falls. Throws `DomainError` for a penalty that is not positive or a bad `count`, and `ShapeError` for
 * mismatched shapes or a `groups` of the wrong length.
 *
 * @param x The design, $n \times d$; standardise it first for the penalty to treat the columns alike.
 * @param y The target, $n$ values.
 * @param options The groups, the grid (or explicit penalties), the intercept and the stopping rule, as
 *   `GroupLassoPathOptions`.
 * @returns The path: the penalties, the weights ($d \times k$), the group norms ($G \times k$), intercepts, active
 *   group counts and sweeps per fit.
 *
 * @example Groups join the path whole
 * const x = normals(stream(0), [100, 5])
 * const y = tensor(toArray(x).map((r) => 2 * r[0] + r[1] + 0.5 * r[4]))
 * const path = groupLassoPath(x, y, { groups: [0, 0, 1, 1, 2], count: 7, span: 2 })
 * print('alphas:', path.alphas.map((a) => a.toFixed(3)))
 * print('active groups:', path.activeGroups)
 * print('group norms:', toArray(path.groupNorms).map((row) => row.map((v) => v.toFixed(2))))
 */
export function groupLassoPath(x: Tensor, y: Tensor, options: GroupLassoPathOptions = {}): GroupLassoPath {
  const { groups, count = 81, span = 4, intercept = true, tolerance = 1e-10, maxSteps = 10000 } = options
  const c = centre(x, y, intercept, 'groupLassoPath')
  const g = gram(c)
  const { n, d } = c
  const b = blocks(g, groups, 'groupLassoPath')
  let alphas: number[]
  if (options.alphas) alphas = [...options.alphas].sort((u, v) => v - u)
  else {
    if (!(count >= 1 && Number.isInteger(count)))
      throw new DomainError('groupLassoPath', 'groupLassoPath: count must be a positive integer')
    let top = 0
    for (const cols of b.members) {
      const norm = Math.sqrt(cols.reduce((s, j) => s + g.q[j] * g.q[j], 0))
      top = Math.max(top, norm / (n * Math.sqrt(cols.length)))
    }
    const max = Math.max(top, 1e-15)
    alphas = Array.from({ length: count }, (_, t) => max * 10 ** (count === 1 ? 0 : (-span * t) / (count - 1)))
  }
  for (const a of alphas) checkAlpha(a, 'groupLassoPath')
  const k = alphas.length
  const G = b.members.length
  const coef = new Float64Array(d * k)
  const norms = new Float64Array(G * k)
  const w = new Float64Array(d)
  const intercepts: number[] = []
  const activeGroups: number[] = []
  const steps: number[] = []
  let converged = true
  alphas.forEach((alpha, t) => {
    const run = groupSweeps(g, b, w, n * alpha, tolerance, maxSteps)
    converged &&= run.converged
    steps.push(run.steps)
    let bias = c.yMean
    for (let j = 0; j < d; j++) {
      coef[j * k + t] = w[j]
      bias -= c.xMean[j] * w[j]
    }
    let active = 0
    b.members.forEach((cols, gi) => {
      const norm = Math.sqrt(cols.reduce((s, j) => s + w[j] * w[j], 0))
      norms[gi * k + t] = norm
      if (norm !== 0) active++
    })
    intercepts.push(bias)
    activeGroups.push(active)
  })
  return {
    alphas,
    coef: fromData(coef, [d, k]),
    groupNorms: fromData(norms, [G, k]),
    groups: b.labels,
    intercept: intercepts,
    activeGroups,
    steps,
    converged,
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'groupLasso',
    module: 'learning/linear',
    name: 'Group lasso',
    summary: 'Least squares with a sum of unsquared L2 norms over groups of weights, which selects whole groups.',
    task: 'regression',
    capabilities: ['forward', 'decide'],
    hyper: space({
      alpha: real(1e-4, 100, { default: 1, scale: 'log', label: '\\alpha' }),
      intercept: bool({ default: true }),
    }),
    cite: ['yuan2006'],
  },
  groupLasso,
)
