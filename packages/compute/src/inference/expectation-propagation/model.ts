/**
 * Expectation propagation over a model of the model language (`aifn-compute/inference/model`) whose latent variables are
 * linear-Gaussian, with interval constraints and Gaussian observations as evidence: the message passing of TrueSkill
 * and Infer.NET (Minka 2001, "Expectation propagation for approximate Bayesian inference", UAI; Herbrich, Minka &
 * Graepel 2007, "TrueSkill", NIPS, §2.3; Minka et al. 2018, Infer.NET).
 *
 * The model is compiled into factors on latent scalar instances z, each a function of one linear form u = wᵀz:
 *
 * - a latent `Normal(mean, sd)` whose mean is an affine form cᵀz + b of other latents: ψ(z_i − cᵀz) = N(·; b, sd²)
 *   (a latent with a constant mean is its own Gaussian prior, not a factor);
 * - an observed `Normal(mean, sd)` with data y: ψ(aᵀz) = N(·; y − b, sd²);
 * - an observed `Bernoulli` of an `interval` node with data 1: ψ(aᵀz) = 𝟙(lower − b < aᵀz < upper − b) (0 states the
 *   complement, allowed when one bound is infinite).
 *
 * Deterministic `sum`, `difference`, `product` (by constants), `linear` and constant `index` nodes are folded into the
 * linear forms. The approximation is fully factorised, q(z) = Πᵢ N(zᵢ; mᵢ, vᵢ), with one Gaussian site per (factor,
 * variable) pair. A Gaussian factor sends its exact message (belief propagation): to z_j, N(w_j z_j; μ − Σ_{i≠j} wᵢmᵢ,
 * s² + Σ_{i≠j} wᵢ²vᵢ) from the cavities. An interval factor moment-matches: with the cavity of u, N(m_u, v_u), and the
 * truncated-normal moments (m̂, v̂) of u, each z_i's tilted marginal has mean mᵢ + wᵢvᵢ(m̂ − m_u)/v_u and variance
 * vᵢ + (wᵢvᵢ)²(v̂ − v_u)/v_u², and the site is that marginal divided by the cavity.
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

/** An affine form Σ coefficients[k]·z_k + constant over latent indices. */
type Affine = { terms: Map<number, number>; constant: number }

/** One compiled factor: ψ(wᵀz_scope), Gaussian N(·; mean, variance) or an interval indicator. */
type Factor = {
  key: string
  /** For a latent's own conditional, that latent's index; null for evidence. */
  own: number | null
  scope: number[]
  weights: number[]
} & ({ kind: 'gaussian'; mean: number; variance: number } | { kind: 'interval'; lower: number; upper: number })

/** A model compiled for {@link modelExpectationPropagation}. */
export interface CompiledGaussianModel {
  /** The latent instance keys, in model order (the order of `means` and `variances`). */
  keys: string[]
  /** The Gaussian priors of latents with a constant mean (precision 0 for the others). */
  priorPrecision: number[]
  priorShift: number[]
  factors: Factor[]
}

const fail = (what: string): never => {
  throw new DomainError('modelExpectationPropagation', `modelExpectationPropagation: ${what}`)
}

const isRef = (a: Arg): a is NodeRef => typeof a === 'object' && a !== null && 'kind' in a && a.kind === 'ref'

function scalar(v: NodeValue, what: string): number {
  if (typeof v === 'number') return v
  if (isTensor(v) && v.shape.reduce((a, b) => a * b, 1) === 1) return toFlat(v)[0]
  return fail(`${what} must be a scalar`)
}

const constantForm = (c: number): Affine => ({ terms: new Map(), constant: c })

function combine(a: Affine, b: Affine, scale = 1): Affine {
  const terms = new Map(a.terms)
  for (const [k, c] of b.terms) terms.set(k, (terms.get(k) ?? 0) + scale * c)
  return { terms, constant: a.constant + scale * b.constant }
}

function scaled(a: Affine, s: number): Affine {
  return { terms: new Map([...a.terms].map(([k, c]) => [k, s * c])), constant: s * a.constant }
}

/**
 * Compile a model and its bindings for EP: latent instances, their Gaussian priors, and the factors. Throws, naming
 * the instance, when the model is not linear-Gaussian with interval and Gaussian evidence.
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
  /** Weight of the old site in each update, in [0, 1). Default 0. */
  damping?: number
  /** A sweep that moves no site parameter more than this has converged. Default 1e-8. */
  tolerance?: number
}

/** The state of {@link modelExpectationPropagation}: `t` counts sweeps. */
export interface ModelEpState extends Status {
  t: Size
  /** The latent instance keys, the order of `means` and `variances`. */
  keys: string[]
  means: Tensor
  variances: Tensor
  /** Site precisions and shifts, one per (factor, scope variable), factors in order. */
  sitePrecision: Tensor
  siteShift: Tensor
  /** Largest change of a site parameter in the last sweep. */
  change: number
  /** Interval updates skipped in the last sweep because a cavity was improper. */
  skipped: number
  converged: boolean
}

/**
 * EP over a linear-Gaussian model with interval and Gaussian evidence (see the module comment), as a traceable
 * algorithm: one sweep per step, `converged` when a sweep moves no site by more than `tolerance`. The marginals are
 * `means` and `variances`, in the order of `keys`. No start.
 *
 * @example
 * const s = run(modelExpectationPropagation(trueSkillModel(), bindings), undefined, 100)
 * s.means // posterior skill and performance means
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
 * factor meets an improper cavity.
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
