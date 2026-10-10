/**
 * Realised paths of a stochastic vector field mixture and the work of solving them (Twomey, Kozłowski &
 * Santos-Rodríguez, 2020, §3 and §4.3). One realisation per instance freezes its randomness for the whole solve, as the
 * paper does by recording the random state at the start and resetting it whenever samples are drawn: a uniform picks
 * the component by the inverse CDF of $\pivec$ (once under pick and stick, at every grid time under forward filtering,
 * where $\pivec$ is filtered on the path itself), and normals sample the SVF's direction and length (the
 * reparameterisation trick). `mean` realisations take the most probable component and the mean VF.
 *
 * The number of function evaluations (NFE) of a solve is measured per instance with `dormandPrinceRows` (each instance
 * solved alone, as in fig. 11) and for the whole batch as one system (the "traditional" NFE: one step size for all,
 * chosen on the root-mean-square error over every coordinate). Instances are rows of a row-major `Float64Array`
 * throughout.
 */

import { dormandPrinceRows, type RowsRhs } from 'aifn-compute/dynamics/ode'
import { child, normals, units, type Stream } from 'aifn-compute/foundation/random'
import { fromData, mul, ones, toFlat } from 'aifn-compute/foundation/tensor'
import { flatOf, stackedMlp, type Svfm, type SvfmParams } from './model'

/** Sampled paths (`'sample'`), or mean paths (`'mean'`: the most probable component and the mean VF). */
export type RealisationMode = 'sample' | 'mean'

/** The frozen randomness of one realisation per instance. */
export type Realisation = {
  /** A uniform per instance, for the component draws. */
  u: Float64Array
  /** Normals per instance, $[n \times (S + 1)]$: $S$ for the direction, one for the log length. */
  eps: Float64Array
}

/**
 * Draw a realisation for $n$ instances of state dimension $S$.
 *
 * @param s The stream to draw from (its children `'components'` and `'field'`).
 * @param n The number of instances.
 * @param S The state dimension, `stateDim` of the model.
 * @returns A uniform and $S + 1$ normals per instance.
 *
 * @example Three instances of a 2-d state
 * const r = realisation(stream(0), 3, 2)
 * print('uniforms:', r.u, ' normals:', r.eps.length)
 */
export function realisation(s: Stream, n: number, S: number): Realisation {
  return {
    u: units(child(s, 'components'), n),
    eps: Float64Array.from(toFlat(normals(child(s, 'field'), [n * (S + 1)]))),
  }
}

/**
 * The component an instance follows: the inverse CDF of $\pivec$ at its uniform, or the most probable one.
 *
 * @param pi The weights of every instance, $[n \times K]$.
 * @param at The offset of this instance's $K$ weights in `pi`.
 * @param K The number of components.
 * @param u The instance's uniform (unused for `'mean'`).
 * @param mode `'sample'` for the inverse CDF, `'mean'` for the most probable component.
 * @returns The component's index; the last one if rounding leaves the cumulative weights below `u`.
 */
function pick(pi: Float64Array, at: number, K: number, u: number, mode: RealisationMode): number {
  if (mode === 'mean') {
    let best = 0
    for (let k = 1; k < K; k++) if (pi[at + k] > pi[at + best]) best = k
    return best
  }
  let acc = 0
  for (let k = 0; k < K; k++) {
    acc += pi[at + k]
    if (u < acc) return k
  }
  return K - 1
}

/**
 * The realised VF of each row: component `comps[row]`, sampled with the row's frozen normals (or its mean). For an SVF
 * unit the direction is $\uvec = \operatorname{normalise}(\muvec^{(u)} + \sqrt{\tau^{(u)}} \Pmat \epsilonvec)$ with
 * $\Pmat = \Imat - \muvec^{(u)} \muvec^{(u)\top}$ and the length $\norm{\avec} e^{\sqrt{\tau^{(v)}} \varepsilon}$; the
 * mean VF is $\avec \, e^{\tau^{(v)}/2}$, as `moments` has it. Plain arithmetic: not differentiable.
 *
 * @param model The SVFM.
 * @param params Its parameters.
 * @param comps The component each instance follows, $[n]$.
 * @param real The frozen randomness of the $n$ instances.
 * @param mode `'sample'` for the realised VF, `'mean'` for the mean VF.
 * @param context The instances' context, $[n \times C]$, or null.
 * @returns A right-hand side for `dormandPrinceRows`: given each row's time, the rows' states ($[m \times S]$) and
 *   which instances they are, the VF at each row, $[m \times S]$.
 *
 * @example The mean VF of each row is the `moments` mean of its component
 * const model = svfm({ dim: 2, components: 2, stochastic: true, learnVariance: false, varianceBias: 0 })
 * const params = model.init(stream(0))
 * const real = realisation(stream(1), 2, 2)
 * const comps = Int32Array.of(0, 1)
 * const at = [Float64Array.of(0, 0), Float64Array.of(1, 0, 0, 1), Int32Array.of(0, 1)]
 * print('sampled VF:', realisedRhs(model, params, comps, real, 'sample', null)(...at))
 * print('mean VF:', realisedRhs(model, params, comps, real, 'mean', null)(...at))
 * const z = tensor([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
 * print('moments (component 0, row 0; component 1, row 1):', model.moments(params, 0, z, null).mean)
 */
export function realisedRhs(
  model: Svfm,
  params: SvfmParams,
  comps: Int32Array,
  real: Realisation,
  mode: RealisationMode,
  context: Float64Array | null,
): RowsRhs {
  const { stochastic, context: C, activation, maxVariance, learnVariance, varianceBias, timeDependent } = model.options
  const S = model.stateDim
  const inF = S + 1 + C
  const out = stochastic ? S + 2 : S
  return (t, x, rows) => {
    const m = rows.length
    const input = new Float64Array(m * inF)
    for (let a = 0; a < m; a++) {
      for (let j = 0; j < S; j++) input[a * inF + j] = x[a * S + j]
      input[a * inF + S] = timeDependent ? t[a] : 0
      for (let j = 0; j < C; j++) input[a * inF + S + 1 + j] = context![rows[a] * C + j]
    }
    const raw = flatOf(stackedMlp(params.fields, fromData(input, [m, inF]), activation)) // [K, m, out]
    const f = new Float64Array(m * S)
    for (let a = 0; a < m; a++) {
      const r = rows[a]
      const base = (comps[r] * m + a) * out
      if (!stochastic) {
        for (let j = 0; j < S; j++) f[a * S + j] = raw[base + j]
        continue
      }
      let n2 = 0
      for (let j = 0; j < S; j++) n2 += raw[base + j] ** 2
      const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))
      const tauU = maxVariance * sigmoid(learnVariance ? raw[base + S] : varianceBias)
      const tauV = maxVariance * sigmoid(learnVariance ? raw[base + S + 1] : varianceBias)
      if (mode === 'mean' || n2 === 0) {
        for (let j = 0; j < S; j++) f[a * S + j] = raw[base + j] * Math.exp(0.5 * tauV)
        continue
      }
      // u = normalise(μ⁽ᵘ⁾ + √τᵘ P ε) with P = I − μ⁽ᵘ⁾μ⁽ᵘ⁾ᵀ; L = ‖a‖ exp(√τᵛ ε_ℓ).
      const nrm = Math.sqrt(n2)
      const e = real.eps.subarray(r * (S + 1), r * (S + 1) + S)
      let dot = 0
      for (let j = 0; j < S; j++) dot += (e[j] * raw[base + j]) / nrm
      const u = new Float64Array(S)
      let un = 0
      for (let j = 0; j < S; j++) {
        const mu = raw[base + j] / nrm
        u[j] = mu + Math.sqrt(tauU) * (e[j] - dot * mu)
        un += u[j] ** 2
      }
      const L = nrm * Math.exp(Math.sqrt(tauV) * real.eps[r * (S + 1) + S])
      for (let j = 0; j < S; j++) f[a * S + j] = (L * u[j]) / Math.sqrt(un)
    }
    return f
  }
}

/**
 * $\pivec(t_0)$ for every instance (eq. 2).
 *
 * @param model The SVFM.
 * @param params Its parameters.
 * @param h The instances' lifted states, $[n \times S]$.
 * @param n The number of instances.
 * @param context Their context, $[n \times C]$, or null.
 * @returns The weights, $[n \times K]$.
 */
function priorOf(model: Svfm, params: SvfmParams, h: Float64Array, n: number, context: Float64Array | null) {
  const S = model.stateDim
  const c = context ? fromData(context, [n, model.options.context]) : null
  return Float64Array.from(flatOf(model.prior(params, fromData(h, [n, S]), c)), Math.exp)
}

/**
 * The forward-filtering update (eq. 5) on realised states: every component is taken to be at the instance's state, so
 * $\psivec$ and $\Psimat$ are evaluated on the path itself.
 *
 * @param model The SVFM, with forward filtering.
 * @param params Its parameters.
 * @param t The grid time of the update.
 * @param pi The weights before it, $[n \times K]$.
 * @param h The instances' states at $t$, $[n \times S]$.
 * @param n The number of instances.
 * @param context Their context, $[n \times C]$, or null.
 * @returns The weights after it, $[n \times K]$.
 */
function filterOn(
  model: Svfm,
  params: SvfmParams,
  t: number,
  pi: Float64Array,
  h: Float64Array,
  n: number,
  context: Float64Array | null,
): Float64Array {
  const { components: K } = model.options
  const S = model.stateDim
  const c = context ? fromData(context, [n, model.options.context]) : null
  // Every component "at" the realised state: ψ and Ψ evaluated on h itself.
  const states = mul(ones([K, 1, 1]), fromData(h, [1, n, S]))
  const logPi = fromData(
    Float64Array.from(pi, (v) => Math.log(Math.max(v, 1e-300))),
    [n, K],
  )
  const f = model.filter(params, t, logPi, states, fromData(new Float64Array(K * n), [K, n]), c)
  return Float64Array.from(flatOf(f.logWeights), Math.exp)
}

/** Options of {@link samplePaths} and {@link instanceWork}. */
export type RealisedOptions = {
  /** Sampled or mean paths. Default `'sample'`. */
  mode?: RealisationMode
  /** Dormand–Prince's relative tolerance. Default $10^{-4}$. */
  rtol?: number
  /** Dormand–Prince's absolute tolerance. Default $10^{-6}$. */
  atol?: number
  /** Frames per grid interval of the recorded path. Default 4. */
  framesPerInterval?: number
  /** Accepted steps per solve before an instance is given up (its work is then a lower bound). Default 2000. */
  maxSteps?: number
}

/** Realised paths of $n$ instances. */
export type RealisedPaths = {
  /** Frame times on $[0, 1]$. */
  times: Float64Array
  /** States at each frame, $[\text{frames} \times n \times S]$. */
  paths: Float64Array
  /** The component followed in each grid interval, $[T \times n]$. */
  components: Int32Array
  /** $\pivec$ at each grid time, $[(T + 1) \times n \times K]$. */
  weights: Float64Array
}

/**
 * The realised paths of instances, recorded at `framesPerInterval` frames per grid interval, each frame interval
 * solved by Dormand–Prince with every instance's step size controlled on its own. Under forward filtering $\pivec$ is
 * filtered on the path at every grid time and the component redrawn from the instance's frozen uniform; a solve
 * continues across the grid time where the component is kept, and restarts its step size where it changes.
 *
 * @param model The SVFM.
 * @param params Its parameters.
 * @param x The starts, $[n \times D]$ row-major.
 * @param context The instances' context, $[n \times C]$, or null.
 * @param real The frozen randomness of the $n$ instances, from `realisation`.
 * @param options The mode, the tolerances, the frames per interval and the step limit; see `RealisedOptions`.
 * @returns The frame times, the states, the components followed and $\pivec$ along the paths.
 *
 * @example Under forward filtering $\pivec$ moves along the path, and an instance may switch component
 * const model = svfm({ dim: 2, components: 2, selection: 'forward-filtering', grid: 4 })
 * const x = Float64Array.of(1, 0, 0, 1, -1, -1)
 * const real = realisation(stream(1), 3, 2)
 * const paths = samplePaths(model, model.init(stream(0)), x, null, real, { framesPerInterval: 2 })
 * print('frame times:', paths.times)
 * print('component of each point per interval:', paths.components)
 * print('pi of the first point at t_0 ... t_4:', [0, 1, 2, 3, 4].map((i) => paths.weights.slice(i * 6, i * 6 + 2)))
 * print('end states:', paths.paths.slice(-6))
 */
export function samplePaths(
  model: Svfm,
  params: SvfmParams,
  x: Float64Array,
  context: Float64Array | null,
  real: Realisation,
  options: RealisedOptions = {},
): RealisedPaths {
  const { mode = 'sample', rtol = 1e-4, atol = 1e-6, framesPerInterval = 4, maxSteps = 2000 } = options
  const { dim: D, components: K, grid: T, selection } = model.options
  const S = model.stateDim
  const n = x.length / D
  let h = flatOf(model.lift(fromData(x, [n, D])))
  const F = T * framesPerInterval
  const times = Float64Array.from({ length: F + 1 }, (_, f) => f / F)
  const paths = new Float64Array((F + 1) * n * S)
  const components = new Int32Array(T * n)
  const weights = new Float64Array((T + 1) * n * K)
  paths.set(h, 0)
  let pi: Float64Array = priorOf(model, params, h, n, context)
  weights.set(pi, 0)
  const comps = Int32Array.from({ length: n }, (_, i) => pick(pi, i * K, K, real.u[i], mode))
  let resume: { nextStepSize: Float64Array; derivative: Float64Array } | undefined
  for (let i = 0; i < T; i++) {
    components.set(comps, i * n)
    const rhs = realisedRhs(model, params, comps, real, mode, context)
    for (let f = 0; f < framesPerInterval; f++) {
      const g = i * framesPerInterval + f
      const sol = dormandPrinceRows(rhs, h, S, { t0: times[g], tEnd: times[g + 1], rtol, atol, maxSteps, resume })
      h = sol.x
      resume = sol
      paths.set(h, (g + 1) * n * S)
    }
    if (selection === 'forward-filtering' && K > 1) {
      pi = filterOn(model, params, (i + 1) / T, pi, h, n, context)
      const step = Float64Array.from(resume!.nextStepSize)
      for (let r = 0; r < n; r++) {
        const k = pick(pi, r * K, K, real.u[r], mode)
        if (k !== comps[r]) step[r] = NaN
        comps[r] = k
      }
      resume = { nextStepSize: step, derivative: resume!.derivative }
    }
    weights.set(pi, (i + 1) * n * K)
  }
  return { times, paths, components, weights }
}

/** The work of solving $n$ instances. */
export type InstanceWork = {
  /** NFE of each instance solved alone (fig. 11's histograms). */
  perInstance: Int32Array
  /** NFE of the batch solved as one system: one step size for all, error measured over every coordinate. */
  batch: number
  /** End states of the instances solved alone, $[n \times S]$. */
  final: Float64Array
}

/**
 * The NFE of solving each instance's realised path on $[0, 1]$ with Dormand–Prince, alone and as one batch. Pick and
 * stick and a single VF solve the whole interval at once; forward filtering stops at each grid time to filter $\pivec$
 * on the path, and continues the solve (step size and last evaluation kept) where the instance keeps its component,
 * restarting it where the component changes (in the batch, where any instance's does).
 *
 * @param model The SVFM.
 * @param params Its parameters.
 * @param x The starts, $[n \times D]$ row-major.
 * @param context The instances' context, $[n \times C]$, or null.
 * @param real The frozen randomness of the $n$ instances, from `realisation`.
 * @param options The mode, the tolerances and the step limit (`framesPerInterval` is not used); see
 *   `RealisedOptions`.
 * @returns The NFE of each instance alone and of the batch, and the end states.
 *
 * @example Solved alone, most instances of an SVFM need fewer evaluations than the batch
 * const model = svfm({ dim: 2, components: 3, stochastic: true, varianceBias: 0 })
 * const x = Float64Array.from(toArray(normal(stream(2), 0, 1, { shape: [12, 2] })).flat())
 * const w = instanceWork(model, model.init(stream(0)), x, null, realisation(stream(1), 12, 2))
 * print('NFE per instance:', w.perInstance)
 * print('NFE of the batch as one system:', w.batch)
 */
export function instanceWork(
  model: Svfm,
  params: SvfmParams,
  x: Float64Array,
  context: Float64Array | null,
  real: Realisation,
  options: RealisedOptions = {},
): InstanceWork {
  const { mode = 'sample', rtol = 1e-4, atol = 1e-6, maxSteps = 2000 } = options
  const { dim: D, components: K, grid: T, selection } = model.options
  const S = model.stateDim
  const n = x.length / D
  const h0 = flatOf(model.lift(fromData(x, [n, D])))
  const filtering = selection === 'forward-filtering' && K > 1
  const cuts = filtering ? Array.from({ length: T + 1 }, (_, i) => i / T) : [0, 1]
  const run = (batched: boolean) => {
    let h = h0
    let pi: Float64Array = priorOf(model, params, h, n, context)
    const comps = Int32Array.from({ length: n }, (_, i) => pick(pi, i * K, K, real.u[i], mode))
    const work = new Int32Array(n)
    let batchWork = 0
    // A solve continues across a grid time where an instance keeps its component (its field is unchanged).
    let resume: { nextStepSize: Float64Array; derivative: Float64Array } | undefined
    for (let i = 0; i + 1 < cuts.length; i++) {
      const rhs = realisedRhs(model, params, comps, real, mode, context)
      const tol = { t0: cuts[i], tEnd: cuts[i + 1], rtol, atol, maxSteps }
      if (batched) {
        // The batch as one state of n·S coordinates.
        const all = Int32Array.from({ length: n }, (_, r) => r)
        const sol = dormandPrinceRows((t, z) => rhs(new Float64Array(n).fill(t[0]), z, all), h, n * S, {
          ...tol,
          resume,
        })
        batchWork += sol.evaluations[0]
        h = sol.x
        resume = sol
      } else {
        const sol = dormandPrinceRows(rhs, h, S, { ...tol, resume })
        for (let r = 0; r < n; r++) work[r] += sol.evaluations[r]
        h = sol.x
        resume = sol
      }
      if (filtering && i + 2 < cuts.length) {
        pi = filterOn(model, params, cuts[i + 1], pi, h, n, context)
        const before = Int32Array.from(comps)
        for (let r = 0; r < n; r++) comps[r] = pick(pi, r * K, K, real.u[r], mode)
        const step = Float64Array.from(resume.nextStepSize)
        if (batched) {
          if (comps.some((k, r) => k !== before[r])) step[0] = NaN
        } else for (let r = 0; r < n; r++) if (comps[r] !== before[r]) step[r] = NaN
        resume = { nextStepSize: step, derivative: resume.derivative }
      }
    }
    return { work, batchWork, h }
  }
  const alone = run(false)
  const together = run(true)
  return { perInstance: alone.work, batch: together.batchWork, final: alone.h }
}

/**
 * The variance of each instance's realised VF along its path at the grid times (eq. 9 per instance):
 * $\frac{1}{T} \sum_{i=1}^T \norm{\fvec_i - \bar\fvec}^2$, with $\fvec_i$ the VF of the component followed in
 * interval $i$ at the state reached at $t_i$, and $\bar\fvec$ their mean.
 *
 * @param model The SVFM.
 * @param params Its parameters.
 * @param paths The realised paths, from `samplePaths`.
 * @param context The instances' context, $[n \times C]$, or null.
 * @param real The realisation the paths were drawn with.
 * @param framesPerInterval The frames per grid interval the paths were recorded at (so grid time $t_i$ is frame
 *   $i \cdot$ `framesPerInterval`).
 * @param mode Sampled or mean VFs, as the paths were drawn.
 * @returns The variance of each instance, $[n]$.
 *
 * @example A constant field has none; the field $f = t$ has the variance of $t_i = i/4$
 * const model = svfm({ dim: 1, layers: 0, grid: 4 })
 * const constant = { ...model.init(stream(0)), fields: [{ weight: zeros([1, 2, 1]), bias: tensor([[[0.5]]]) }] }
 * const timed = { ...constant, fields: [{ weight: tensor([[[0], [1]]]), bias: tensor([[[0]]]) }] }
 * const real = realisation(stream(1), 2, 1)
 * for (const [name, params] of [['f = 0.5:', constant], ['f = t:', timed]]) {
 *   const paths = samplePaths(model, params, Float64Array.of(0, 1), null, real, { framesPerInterval: 2 })
 *   print(name, realisedVariance(model, params, paths, null, real, 2))
 * }
 */
export function realisedVariance(
  model: Svfm,
  params: SvfmParams,
  paths: RealisedPaths,
  context: Float64Array | null,
  real: Realisation,
  framesPerInterval: number,
  mode: RealisationMode = 'sample',
): Float64Array {
  const { grid: T } = model.options
  const S = model.stateDim
  const n = paths.components.length / T
  const fields: Float64Array[] = []
  for (let i = 1; i <= T; i++) {
    const comps = paths.components.subarray((i - 1) * n, i * n)
    const rhs = realisedRhs(model, params, Int32Array.from(comps), real, mode, context)
    const at = paths.paths.subarray(i * framesPerInterval * n * S, (i * framesPerInterval + 1) * n * S)
    fields.push(
      Float64Array.from(
        rhs(
          new Float64Array(n).fill(i / T),
          Float64Array.from(at),
          Int32Array.from({ length: n }, (_, r) => r),
        ),
      ),
    )
  }
  const out = new Float64Array(n)
  for (let r = 0; r < n; r++) {
    for (let j = 0; j < S; j++) {
      let avg = 0
      for (const f of fields) avg += f[r * S + j] / T
      for (const f of fields) out[r] += (f[r * S + j] - avg) ** 2 / T
    }
  }
  return out
}
