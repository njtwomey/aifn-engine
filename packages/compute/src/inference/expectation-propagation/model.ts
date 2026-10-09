/**
 * Expectation propagation over a model of the model language (`aifn-compute/inference/model`) whose latent variables
 * are linear-Gaussian, with interval constraints and Gaussian observations as evidence: the message passing of
 * TrueSkill and Infer.NET (Minka 2001, "Expectation propagation for approximate Bayesian inference", UAI; Herbrich,
 * Minka & Graepel 2007, "TrueSkill", NIPS, §2.3; Minka et al. 2018, Infer.NET).
 *
 * The model is compiled into factors on latent scalar instances $\zvec$, each a function of one linear form
 * $u = \wvec^\top \zvec$:
 *
 * - a latent `Normal(mean, sd)` whose mean is an affine form $\cvec^\top \zvec + b$ of other latents:
 *   $\psi(z_i - \cvec^\top \zvec) = \Gauss(\cdot; b, \text{sd}^2)$ (a latent with a constant mean is its own Gaussian
 *   prior, not a factor);
 * - an observed `Normal(mean, sd)` with data $y$: $\psi(\avec^\top \zvec) = \Gauss(\cdot; y - b, \text{sd}^2)$;
 * - an observed `Bernoulli` of an `interval` node with data 1:
 *   $\psi(\avec^\top \zvec) = \indicator(l - b < \avec^\top \zvec < u - b)$, $l$ and $u$ the bounds (0 states the
 *   complement, allowed when one bound is infinite).
 *
 * Deterministic `sum`, `difference`, `product` (by constants) and constant `index` nodes are folded into the linear
 * forms. The approximation is fully factorised, $q(\zvec) = \prod_i \Gauss(z_i; m_i, v_i)$, with one Gaussian site
 * per (factor, variable) pair. A Gaussian factor sends its exact message (belief propagation): to $z_j$,
 * $\Gauss(w_j z_j; \mu - \sum_{i \ne j} w_i m_i, s^2 + \sum_{i \ne j} w_i^2 v_i)$ from the cavities. An interval factor
 * moment-matches: with the cavity of $u$, $\Gauss(m_u, v_u)$, and the truncated-normal moments $(\hat{m}, \hat{v})$
 * of $u$, each $z_i$'s tilted marginal has mean $m_i + w_i v_i (\hat{m} - m_u)/v_u$ and variance
 * $v_i + (w_i v_i)^2 (\hat{v} - v_u)/v_u^2$, and the site is that marginal divided by the cavity.
 *
 * Schedule: each step is one sweep. For every evidence factor in model order, the Gaussian factors of the latents in
 * its scope pass messages in, the evidence factor updates, and those Gaussian factors pass messages out (TrueSkill's
 * schedule: a game's performances, its outcome, then back to the skills); Gaussian factors next to no evidence update
 * once per sweep.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  argValue,
  distOf,
  environment,
  expandModel,
  resolveRef,
  type Arg,
  type Bindings,
  type ExpandedModel,
  type Instance,
  type Model,
  type NodeRef,
  type NodeValue,
} from 'aifn-compute/inference/model'
import { intervalTilted } from './tilted'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * An affine form $\sum_k c_k z_k + b$ over latent indices: `terms` maps a latent's index $k$ to its coefficient $c_k$,
 * and `constant` is $b$.
 */
type Affine = { terms: Map<number, number>; constant: number }

/**
 * One compiled factor: $\psi(\wvec^\top \zvec_\text{scope})$, Gaussian $\Gauss(\cdot; \text{mean}, \text{variance})$
 * or an interval indicator $\indicator(\text{lower} < \cdot < \text{upper})$.
 */
type Factor = {
  /** The key of the instance the factor came from. */
  key: string
  /** For a latent's own conditional, that latent's index; null for evidence. */
  own: number | null
  /** The latent indices the factor's linear form reads. */
  scope: number[]
  /** The coefficient of each latent of `scope` in the linear form. */
  weights: number[]
} & ({ kind: 'gaussian'; mean: number; variance: number } | { kind: 'interval'; lower: number; upper: number })

/** A model compiled for {@link modelExpectationPropagation}. */
export interface CompiledGaussianModel {
  /** The latent instance keys, in model order (the order of `means` and `variances`). */
  keys: string[]
  /** The prior precision of each latent with a constant mean (0 for the others). */
  priorPrecision: number[]
  /** The prior shift (precision times mean) of each latent with a constant mean (0 for the others). */
  priorShift: number[]
  /** The compiled factors: the latents' conditionals on other latents, then evidence, in model order. */
  factors: Factor[]
}

/**
 * Throw the module's `DomainError`.
 *
 * @param what What is wrong, appended to the function's name in the message.
 * @returns Never: it always throws.
 */
const fail = (what: string): never => {
  throw new DomainError('modelExpectationPropagation', `modelExpectationPropagation: ${what}`)
}

/**
 * Whether an argument is a reference to another node.
 *
 * @param a The argument.
 * @returns True for a `{ kind: 'ref' }` object.
 */
const isRef = (a: Arg): a is NodeRef => typeof a === 'object' && a !== null && 'kind' in a && a.kind === 'ref'

/**
 * A value as a number; throws `DomainError` when it is not a scalar (a number or a one-element tensor).
 *
 * @param v The value.
 * @param what The name of the value, for the error message.
 * @returns The number.
 */
function scalar(v: NodeValue, what: string): number {
  if (typeof v === 'number') return v
  if (isTensor(v) && v.shape.reduce((a, b) => a * b, 1) === 1) return toFlat(v)[0]
  return fail(`${what} must be a scalar`)
}

/**
 * The affine form of a constant.
 *
 * @param c The constant.
 * @returns The form with no terms and constant $c$.
 */
const constantForm = (c: number): Affine => ({ terms: new Map(), constant: c })

/**
 * The affine form $a + s\,b$.
 *
 * @param a The first form; not modified.
 * @param b The second form; not modified.
 * @param scale The multiplier $s$ of `b` (default 1; $-1$ for a difference).
 * @returns A new form.
 */
function combine(a: Affine, b: Affine, scale = 1): Affine {
  const terms = new Map(a.terms)
  for (const [k, c] of b.terms) terms.set(k, (terms.get(k) ?? 0) + scale * c)
  return { terms, constant: a.constant + scale * b.constant }
}

/**
 * The affine form $s\,a$.
 *
 * @param a The form; not modified.
 * @param s The multiplier.
 * @returns A new form.
 */
function scaled(a: Affine, s: number): Affine {
  return { terms: new Map([...a.terms].map(([k, c]) => [k, s * c])), constant: s * a.constant }
}

/**
 * Compile a model and its bindings for EP: latent instances, their Gaussian priors, and the factors. Throws
 * `DomainError`, naming the instance, when the model is not linear-Gaussian with interval and Gaussian evidence: a
 * latent standard deviation or bound, a product of two latents, a non-constant selector, or another family or link.
 * Observed nodes without data are latents.
 *
 * @param m The model, or one already expanded by `expandModel` (then `bindings` is not used).
 * @param bindings The sizes, constants and data the model is expanded with.
 * @returns The latent keys, their priors and the factors.
 *
 * @example Two observations of a weight, one through a product
 * // w ~ N(0, 1), y1 ~ N(w, 1) and y2 ~ N(2w, 1); the product by a constant folds into the factor's weight.
 * const ref = (node) => ({ kind: 'ref', node })
 * const node = (name, role, data) => ({ name, role, group: null, data })
 * const regression = {
 *   kind: 'graph', name: 'regression', directed: true, nodes: 4, groups: [], sizes: [],
 *   labels: ['w', 'y1', 'w2', 'y2'],
 *   edges: [[0, 1], [0, 2], [2, 3]].map(([from, to]) => ({ from, to, directed: true })),
 *   attributes: [
 *     node('w', 'latent', { dist: { family: 'Normal', args: [0, 1] } }),
 *     node('y1', 'observed', { dist: { family: 'Normal', args: [ref('w'), 1] } }),
 *     node('w2', 'deterministic', { op: 'product', args: [ref('w'), 2] }),
 *     node('y2', 'observed', { dist: { family: 'Normal', args: [ref('w2'), 1] } }),
 *   ],
 * }
 * const c = compileGaussianModel(regression, { data: { y1: 1, y2: 4 } })
 * print('latents =', c.keys, 'prior precision', c.priorPrecision)
 * print('factors =', c.factors)
 */
export function compileGaussianModel(m: Model | ExpandedModel, bindings: Bindings = {}): CompiledGaussianModel {
  const em = 'instances' in m ? m : expandModel(m, bindings)
  const latents = em.instances.filter(
    (i) => i.node.role === 'latent' || (i.node.role === 'observed' && !em.fixed.has(i.key)),
  )
  const index = new Map(latents.map((inst, k) => [inst.key, k]))
  const env = environment(em, new Map())
  const forms = new Map<string, Affine>()

  /** The affine form of an instance's value. */
  const formOf = (key: string): Affine => {
    const known = forms.get(key)
    if (known) return known
    const inst = em.byKey.get(key)!
    let form: Affine
    if (index.has(key)) form = { terms: new Map([[index.get(key)!, 1]]), constant: 0 }
    else if (inst.node.role === 'deterministic') form = deterministicForm(inst)
    else form = constantForm(scalar(env(key), key))
    forms.set(key, form)
    return form
  }

  /** The affine form of an argument of `inst`. */
  const argForm = (inst: Instance, a: Arg): Affine => {
    if (!isRef(a)) return constantForm(scalar(argValue(em, inst, a, env), `an argument of ${inst.key}`))
    const r = resolveRef(em, inst, a)
    if (r.selector !== null && !em.fixed.has(r.selector))
      fail(`${inst.key} selects by ${r.selector}, which is not a constant`)
    if (r.indexesValue) return constantForm(scalar(argValue(em, inst, a, env), `an argument of ${inst.key}`))
    return formOf(r.choose(env))
  }

  const deterministicForm = (inst: Instance): Affine => {
    const { op, args = [] } = inst.node.data!
    const fs = () => args.map((a) => argForm(inst, a))
    switch (op) {
      case 'sum':
        return fs().reduce((s, f) => combine(s, f))
      case 'difference': {
        const [a, b] = fs()
        return combine(a, b, -1)
      }
      case 'product':
        return fs().reduce((s, f) => {
          if (s.terms.size === 0) return scaled(f, s.constant)
          if (f.terms.size === 0) return scaled(s, f.constant)
          return fail(`${inst.key} multiplies two latent quantities`)
        })
      case 'index':
        return constantForm(scalar(env(inst.key), inst.key))
      default:
        return fail(`${inst.key} (${op}) is not linear in the latent variables`)
    }
  }

  const priorPrecision = latents.map(() => 0)
  const priorShift = latents.map(() => 0)
  const factors: Factor[] = []
  const linear = (form: Affine) => ({ scope: [...form.terms.keys()], weights: [...form.terms.values()] })

  for (const inst of em.instances) {
    if (inst.node.role === 'deterministic' || inst.node.role === 'parameter') continue
    const d = distOf(inst)
    const observed = em.fixed.has(inst.key)
    if (d.family === 'Normal') {
      const mean = argForm(inst, d.args[0])
      const sd = argForm(inst, d.args[1])
      if (sd.terms.size) fail(`${inst.key} has a latent standard deviation`)
      const variance = sd.constant * sd.constant
      if (observed) {
        if (mean.terms.size === 0) continue // a constant: no information about the latents
        factors.push({
          key: inst.key,
          own: null,
          ...linear(mean),
          kind: 'gaussian',
          mean: scalar(em.fixed.get(inst.key)!, inst.key) - mean.constant,
          variance,
        })
      } else if (mean.terms.size === 0) {
        const k = index.get(inst.key)!
        priorPrecision[k] = 1 / variance
        priorShift[k] = mean.constant / variance
      } else {
        // z_i − cᵀz ~ N(b, sd²)
        const form = combine(formOf(inst.key), mean, -1)
        factors.push({
          key: inst.key,
          own: index.get(inst.key)!,
          ...linear(form),
          kind: 'gaussian',
          mean: mean.constant,
          variance,
        })
      }
      continue
    }
    if (d.family === 'Bernoulli' && observed && isRef(d.args[0])) {
      const target = em.byKey.get(resolveRef(em, inst, d.args[0]).choose(env))!
      if (target.node.data?.op === 'interval') {
        const [x, lo, hi] = target.node.data.args!.map((a) => argForm(target, a))
        if (lo.terms.size || hi.terms.size) fail(`${target.key} has latent bounds`)
        let lower = lo.constant - x.constant
        let upper = hi.constant - x.constant
        if (scalar(em.fixed.get(inst.key)!, inst.key) === 0) {
          if (lower === -Infinity) [lower, upper] = [upper, Infinity]
          else if (upper === Infinity) [lower, upper] = [-Infinity, lower]
          else fail(`${inst.key} = 0 states the complement of a bounded interval`)
        }
        factors.push({ key: inst.key, own: null, ...linear(x), kind: 'interval', lower, upper })
        continue
      }
    }
    fail(`${inst.key} (${d.family}) is neither a linear-Gaussian latent nor interval or Gaussian evidence`)
  }
  return { keys: latents.map((i) => i.key), priorPrecision, priorShift, factors }
}

/** Options of {@link modelExpectationPropagation}. */
export interface ModelEpOptions {
  /** Weight of the old site in each update, in $[0, 1)$. Default 0; outside $[0, 1)$ throws `DomainError`. */
  damping?: number
  /** A sweep that moves no site parameter more than this has converged. Default 1e-8. */
  tolerance?: number
}

/** The state of {@link modelExpectationPropagation}: `t` counts sweeps. */
export interface ModelEpState extends Status {
  /** The sweeps run. */
  t: Size
  /** The latent instance keys, the order of `means` and `variances`. */
  keys: string[]
  /** The posterior mean of each latent. */
  means: Tensor
  /** The posterior variance of each latent. */
  variances: Tensor
  /** Site precisions, one per (factor, scope variable), factors in order. */
  sitePrecision: Tensor
  /** Site shifts, one per (factor, scope variable), factors in order. */
  siteShift: Tensor
  /** Largest change of a site parameter in the last sweep ($\infty$ at the start). */
  change: number
  /**
   * Interval updates skipped in the last sweep because a cavity was improper or the tilted moments were not usable.
   */
  skipped: number
  /** Whether the last sweep moved no site parameter by more than `tolerance` (true at once with no factors). */
  converged: boolean
}

/**
 * EP over a linear-Gaussian model with interval and Gaussian evidence (see the file comment), as a traceable
 * algorithm: one sweep per step, `converged` when a sweep moves no site by more than `tolerance`. The marginals are
 * `means` and `variances`, in the order of `keys`. No start. The model is compiled by `compileGaussianModel` when the
 * algorithm is made, so a model it cannot handle throws `DomainError` then, as does a `damping` outside $[0, 1)$.
 *
 * @param m The model, or one already expanded by `expandModel`.
 * @param bindings The sizes, constants and data (default none); evidence is the observed nodes with data.
 * @param options The damping and tolerance.
 * @returns The algorithm, to run with `run(alg, undefined, steps)`.
 *
 * @example One game: the winner's skill rises
 * // Skills a, b ~ N(0, 1) and the observation that a − b > 0. The posterior of a is the probit tilt of N(0, 1),
 * // since p(a − b > 0 | a) = Φ(a).
 * const ref = (node) => ({ kind: 'ref', node })
 * const node = (name, role, data) => ({ name, role, group: null, data })
 * const game = {
 *   kind: 'graph', name: 'game', directed: true, nodes: 5, groups: [], sizes: [],
 *   labels: ['a', 'b', 'd', 'won', 'y'],
 *   edges: [[0, 2], [1, 2], [2, 3], [3, 4]].map(([from, to]) => ({ from, to, directed: true })),
 *   attributes: [
 *     node('a', 'latent', { dist: { family: 'Normal', args: [0, 1] } }),
 *     node('b', 'latent', { dist: { family: 'Normal', args: [0, 1] } }),
 *     node('d', 'deterministic', { op: 'difference', args: [ref('a'), ref('b')] }),
 *     node('won', 'deterministic', { op: 'interval', args: [ref('d'), 0, Infinity] }),
 *     node('y', 'observed', { dist: { family: 'Bernoulli', args: [ref('won')] } }),
 *   ],
 * }
 * const s = run(modelExpectationPropagation(game, { data: { y: 1 } }), undefined, 100)
 * print(s.keys, 'means', s.means, 'variances', s.variances)
 * print('probit tilt:', probitTilted(0, 1, 1))
 * print('sweeps =', s.t, 'converged =', s.converged)
 */
export function modelExpectationPropagation(
  m: Model | ExpandedModel,
  bindings: Bindings = {},
  options: ModelEpOptions = {},
): Algorithm<void, ModelEpState> {
  const { keys, priorPrecision, priorShift, factors } = compileGaussianModel(m, bindings)
  const damping = options.damping ?? 0
  if (!(damping >= 0 && damping < 1))
    throw new DomainError('modelExpectationPropagation', 'modelExpectationPropagation: damping must be in [0, 1)')
  const tolerance = options.tolerance ?? 1e-8
  const offsets: number[] = []
  let sites = 0
  for (const f of factors) {
    offsets.push(sites)
    sites += f.scope.length
  }

  // The schedule: every evidence factor wrapped in the Gaussian conditionals of its scope; then the rest.
  const conditionalOf = new Map<number, number>()
  factors.forEach((f, k) => {
    if (f.own !== null) conditionalOf.set(f.own, k)
  })
  const schedule: number[] = []
  const used = new Set<number>()
  factors.forEach((f, k) => {
    if (f.own !== null) return
    const around = f.scope.map((v) => conditionalOf.get(v)).filter((c): c is number => c !== undefined)
    around.forEach((c) => used.add(c))
    schedule.push(...around, k, ...around)
  })
  factors.forEach((f, k) => {
    if (f.own !== null && !used.has(k)) schedule.push(k)
  })

  const marginals = (tau: Float64Array, nu: Float64Array) => {
    const P = Float64Array.from(priorPrecision)
    const N = Float64Array.from(priorShift)
    factors.forEach((f, k) =>
      f.scope.forEach((v, j) => {
        P[v] += tau[offsets[k] + j]
        N[v] += nu[offsets[k] + j]
      }),
    )
    return { P, N }
  }

  const stateOf = (
    t: number,
    tau: Float64Array,
    nu: Float64Array,
    change: number,
    skipped: number,
    converged: boolean,
  ) => {
    const { P, N } = marginals(tau, nu)
    return {
      t,
      keys,
      means: fromData(
        Float64Array.from(P, (p, i) => N[i] / p),
        [P.length],
      ),
      variances: fromData(
        Float64Array.from(P, (p) => 1 / p),
        [P.length],
      ),
      sitePrecision: fromData(tau, [sites]),
      siteShift: fromData(nu, [sites]),
      change,
      skipped,
      converged,
    }
  }

  return {
    name: 'ep.model',
    init: () => stateOf(0, new Float64Array(sites), new Float64Array(sites), Infinity, 0, factors.length === 0),
    step: (s) => {
      if (s.converged) return { ...s, t: s.t + 1 }
      const tau = Float64Array.from(s.sitePrecision.data)
      const nu = Float64Array.from(s.siteShift.data)
      const { P, N } = marginals(tau, nu)
      let change = 0
      let skipped = 0
      for (const k of schedule) {
        const f = factors[k]
        const o = offsets[k]
        const cp = f.scope.map((v, j) => P[v] - tau[o + j])
        const cn = f.scope.map((v, j) => N[v] - nu[o + j])
        const next = updateFactor(f, cp, cn)
        if (next === null) {
          skipped++
          continue
        }
        f.scope.forEach((v, j) => {
          const t = (1 - damping) * next.precision[j] + damping * tau[o + j]
          const n = (1 - damping) * next.shift[j] + damping * nu[o + j]
          change = Math.max(change, Math.abs(t - tau[o + j]), Math.abs(n - nu[o + j]))
          P[v] += t - tau[o + j]
          N[v] += n - nu[o + j]
          tau[o + j] = t
          nu[o + j] = n
        })
      }
      return stateOf(s.t + 1, tau, nu, change, skipped, change < tolerance)
    },
  }
}

/**
 * New sites of one factor from its cavities (natural parameters per scope variable), or null when an interval
 * factor meets an improper cavity or unusable tilted moments. A Gaussian factor's messages are exact; one to $z_j$ is
 * flat when another variable's cavity is.
 *
 * @param f The factor.
 * @param cp The cavity precision of each variable of its scope.
 * @param cn The cavity shift of each variable of its scope.
 * @returns The new site precision and shift for each variable of the scope, or null.
 */
function updateFactor(
  f: Factor,
  cp: readonly number[],
  cn: readonly number[],
): { precision: number[]; shift: number[] } | null {
  const w = f.weights
  if (f.kind === 'gaussian') {
    // Exact messages; a variable with a flat cavity makes the messages to the others flat.
    const precision: number[] = []
    const shift: number[] = []
    w.forEach((wj, j) => {
      let mean = f.mean
      let variance = f.variance
      let flat = false
      w.forEach((wi, i) => {
        if (i === j) return
        if (!(cp[i] > 0)) flat = true
        else {
          mean -= (wi * cn[i]) / cp[i]
          variance += (wi * wi) / cp[i]
        }
      })
      precision.push(flat ? 0 : (wj * wj) / variance)
      shift.push(flat ? 0 : (wj * mean) / variance)
    })
    return { precision, shift }
  }
  if (!cp.every((p) => p > 0)) return null
  const m = cn.map((n, i) => n / cp[i])
  const v = cp.map((p) => 1 / p)
  let mu = 0
  let vu = 0
  w.forEach((wi, i) => {
    mu += wi * m[i]
    vu += wi * wi * v[i]
  })
  const t = intervalTilted(mu, vu, f.lower, f.upper)
  if (!(t.variance > 0 && Number.isFinite(t.mean))) return null
  const precision: number[] = []
  const shift: number[] = []
  w.forEach((wi, i) => {
    const k = (wi * v[i]) / vu
    const mean = m[i] + k * (t.mean - mu)
    const variance = v[i] + k * k * (t.variance - vu)
    precision.push(1 / variance - cp[i])
    shift.push(mean / variance - cn[i])
  })
  return { precision, shift }
}
