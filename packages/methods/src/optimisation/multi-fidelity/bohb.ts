/**
 * BOHB (Falkner, Klein and Hutter, 2018): Hyperband whose configurations come from a model of the results so far, as
 * a step-through algorithm over a learning-curve function of the points of a search space.
 *
 * The brackets and rungs are Hyperband's (or one successive-halving bracket's), run synchronously as
 * `multiFidelitySteps` runs them. When a bracket starts, each of its configurations is drawn at random with
 * probability $\rho$ and otherwise from the model (Falkner et al., 2018, Algorithm 2). The model is built at the
 * largest budget $b$ with $\lvert D_b \rvert \ge N_{\min} + 2$ results: the best
 * $N_\ell = \max(N_{\min}, \lfloor q \lvert D_b \rvert \rfloor)$ form the good group and the worst
 * $N_g = \max(N_{\min}, \lvert D_b \rvert - N_\ell)$ the bad group, and each group gets a product-kernel density
 * $\ell(\xvec)$ or $g(\xvec)$, as statsmodels' `KDEMultivariate` with normal-reference bandwidths (which the authors'
 * HpBandSter uses):
 *
 * $$ h_j = 1.06\,\hat\sigma_j\,N^{-1/(4 + d)}, $$
 *
 * $\hat\sigma_j$ the standard deviation (divisor $N$) of coordinate $j$ in the group of $N$ points, clipped below at
 * `minBandwidth`. A real or integer dimension is modelled in its unit-cube coordinate (`encode` in
 * `aifn-compute/foundation/space`) with a Gaussian kernel, a choice or boolean by its category index with the
 * Aitchison–Aitken kernel ($1 - h$ on the observed category, $h / (C - 1)$ on each other, with $h$ kept at most
 * $(C - 1) / C$). `samples` candidates are drawn from $\ell$ with the continuous bandwidths widened by the bandwidth
 * factor (a truncated normal on $[0, 1]$ about a random good point; a categorical coordinate kept with probability
 * $1 - h$, else uniform), and the one with the smallest $g(\xvec) / \ell(\xvec)$ is chosen. Without a model, every
 * configuration is random.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { decode, encode, sample, type Space, type SpaceValues, type ValuesOf } from 'aifn-compute/foundation/space'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'
import type { MultiFidelityEvaluation, MultiFidelityIncumbent, MultiFidelityPromotion, MultiFidelityState } from './run'
import type { MultiFidelitySchedule } from './schedules'

/**
 * The learning curves BOHB searches: the observed loss of a configuration after a budget, given a stream for its
 * noise; or that function as `at` with the noise-free loss as `clean` (a `learningCurveSurface` of
 * `aifn-methods/data/synthetic` is one), so the true best is known.
 */
export type BohbCurve<S extends Space> =
  | ((point: ValuesOf<S>, budget: number, stream: Stream) => number)
  | {
      /** The observed loss of a configuration after a budget, its noise drawn from the stream. */
      at: (point: ValuesOf<S>, budget: number, stream: Stream) => number
      /** The noise-free loss, when known. */
      clean?: (point: ValuesOf<S>, budget: number) => number
    }

/** Options of `bohbSteps`. */
export interface BohbOptions {
  /** The fraction $\rho$ of configurations drawn at random (default $1/3$). */
  rho?: number
  /** The fraction $q$ of the results in the good group (default 0.15). */
  q?: number
  /** The number of candidates drawn from $\ell$ per model-based configuration (default 64). */
  samples?: number
  /** The factor the continuous bandwidths are widened by when drawing candidates (default 3). */
  bandwidthFactor?: number
  /** The smallest bandwidth (default $10^{-3}$). */
  minBandwidth?: number
  /** $N_{\min}$, the fewest points in a group (default $d + 1$ for $d$ dimensions). */
  minPoints?: number
  /** The number of workers evaluating a rung in parallel (default 1). */
  workers?: number
  /** Whether a promoted configuration resumes its training, costing the increase in budget (default false). */
  resume?: boolean
}

/** One state of `bohbSteps`: a state of `multiFidelitySteps` with the configurations and the model behind them. */
export interface BohbState extends MultiFidelityState {
  /** Every configuration drawn so far; a configuration's index (`config` elsewhere in the state) is its position. */
  points: SpaceValues[]
  /** Whether each configuration was drawn at random (true) or from the model (false). */
  random: boolean[]
  /** The configurations drawn at this step (for the bracket that starts next; empty otherwise). */
  sampled: number[]
  /** The budget of the model behind the latest draws (`null` when there was no model). */
  modelBudget: number | null
  /** The configurations in that model's good group. */
  good: number[]
  /** The configurations in that model's bad group. */
  bad: number[]
  /**
   * The stream the curves' noise comes from: configuration $i$ after budget $t$ is observed with the noise of
   * `child(noise, i, t)`, so the same evaluation always sees the same value.
   */
  noise: Stream
}

/** A dimension as the model sees it. */
type Coordinate = { kind: 'continuous' } | { kind: 'categorical'; count: number }

/** A product-kernel density of one group: its points (model coordinates) and per-dimension bandwidths. */
interface Kde {
  /** The points, model coordinates (a unit coordinate, or a category index). */
  data: number[][]
  /** The bandwidth of each dimension. */
  bw: number[]
}

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)

/**
 * BOHB as a step-through algorithm (see the file comment), minimising the loss at the schedule's maximum budget.
 * Synchronously, each step evaluates one rung of the current bracket and promotes the best
 * $\lfloor n_k / \eta \rfloor$; the step that ends a bracket also draws the next bracket's configurations from the
 * model of every result so far (the initial state draws the first bracket's, at random). The states are those of
 * `multiFidelitySteps`, with the configurations, whether each was random, and the model's budget and groups. The true
 * best of a bracket is by the noise-free final loss when the curve gives it (`clean`), else by the observed one. All
 * randomness comes from the runner's streams. Throws `DomainError` for a space with a conditional, nested or variants
 * dimension, $\rho$ outside $[0, 1]$, $q$ outside $(0, 1]$, a non-positive bandwidth or factor, or a count that is not
 * a positive integer.
 *
 * @param curve The learning curves over the space, as `BohbCurve`.
 * @param space The search space: real (linear or log scale), integer, choice and boolean dimensions.
 * @param schedule The schedule, from `hyperbandSchedule` or `successiveHalvingSchedule`.
 * @param options The model's settings, the workers and resuming, as `BohbOptions`.
 * @returns The algorithm, to run with `run` or `trace` from no start (`undefined`).
 *
 * @example BOHB over Hyperband's brackets for R = 27: random draws first, then the model's
 * import { real, space } from 'aifn-compute/foundation/space'
 * // Loss after budget t: a bowl at x = 0.3 that each configuration approaches as 1 / t
 * const curve = ({ x }, t) => (x - 0.3) ** 2 + 0.5 / t
 * const s = run(bohbSteps(curve, space({ x: real(0, 1) }), hyperbandSchedule({ maxBudget: 27 })), undefined, 20, {
 *   stream: stream(0),
 * })
 * print('configurations drawn at random:', s.random.filter((r) => r).length, 'of', s.points.length)
 * print('model budget:', s.modelBudget, ' incumbent x =', s.points[s.incumbent.config].x.toFixed(3))
 */
export function bohbSteps<const S extends Space>(
  curve: BohbCurve<S>,
  space: S,
  schedule: MultiFidelitySchedule,
  options: BohbOptions = {},
): Algorithm<undefined, BohbState> {
  const where = 'bohbSteps'
  const coords: Coordinate[] = Object.entries(space.dims).map(([key, dim]) => {
    if (dim.when !== undefined) throw new DomainError(where, `${where}: '${key}' is conditional; not supported`)
    if (dim.type === 'real' || dim.type === 'int') return { kind: 'continuous' }
    if (dim.type === 'choice') return { kind: 'categorical', count: dim.options.length }
    if (dim.type === 'bool') return { kind: 'categorical', count: 2 }
    throw new DomainError(where, `${where}: '${key}' is a ${dim.type} dimension; not supported`)
  })
  const d = coords.length
  if (d === 0) throw new DomainError(where, `${where}: the space has no dimensions`)
  const {
    rho = 1 / 3,
    q = 0.15,
    samples = 64,
    bandwidthFactor = 3,
    minBandwidth = 1e-3,
    minPoints = d + 1,
    workers = 1,
    resume = false,
  } = options
  if (!(rho >= 0 && rho <= 1)) throw new DomainError(where, `${where}: rho must lie in [0, 1]`)
  if (!(q > 0 && q <= 1)) throw new DomainError(where, `${where}: q must lie in (0, 1]`)
  if (!(bandwidthFactor > 0 && minBandwidth > 0))
    throw new DomainError(where, `${where}: the bandwidth factor and the smallest bandwidth must be positive`)
  for (const [name, v] of [
    ['samples', samples],
    ['minPoints', minPoints],
    ['workers', workers],
  ] as const)
    if (!(Number.isInteger(v) && v >= 1)) throw new DomainError(where, `${where}: ${name} must be a positive integer`)

  const at = typeof curve === 'function' ? curve : curve.at
  const clean = typeof curve === 'function' ? undefined : curve.clean
  const { maxBudget, brackets } = schedule
  const sizes = brackets.map((b) => b.rungs[0].configurations)
  const offsets = sizes.map((_, b) => sizes.slice(0, b).reduce((a, n) => a + n, 0))
  const configsOf = (b: number) => Array.from({ length: sizes[b] }, (_, i) => offsets[b] + i)

  // Model coordinates of a point: the unit coordinate, or the category's index.
  const toModel = (p: SpaceValues): number[] =>
    Array.from(encode(space, p), (u, j) => {
      const c = coords[j]
      return c.kind === 'continuous' ? u : Math.min(c.count - 1, Math.max(0, Math.floor(u * c.count)))
    })
  const toUnit = (x: readonly number[]): number[] =>
    x.map((v, j) => {
      const c = coords[j]
      return c.kind === 'continuous' ? v : (v + 0.5) / c.count
    })

  /** The KDE of a group, with normal-reference bandwidths. */
  const fitKde = (data: number[][]): Kde => {
    const n = data.length
    const bw = coords.map((c, j) => {
      const mean = data.reduce((a, r) => a + r[j], 0) / n
      const sd = Math.sqrt(data.reduce((a, r) => a + (r[j] - mean) ** 2, 0) / n)
      const h = Math.max(minBandwidth, 1.06 * sd * n ** (-1 / (4 + d)))
      return c.kind === 'categorical' ? Math.min(h, (c.count - 1) / c.count) : h
    })
    return { data, bw }
  }
  const pdf = (kde: Kde, x: readonly number[]): number => {
    let total = 0
    for (const r of kde.data) {
      let logK = 0
      coords.forEach((c, j) => {
        const h = kde.bw[j]
        if (c.kind === 'continuous') {
          const z = (x[j] - r[j]) / h
          logK += -0.5 * z * z - LOG_SQRT_2PI - Math.log(h)
        } else logK += Math.log(x[j] === r[j] ? 1 - h : h / Math.max(1, c.count - 1))
      })
      total += Math.exp(logK)
    }
    return total / kde.data.length
  }
  /** One candidate from the good density, with the continuous bandwidths widened. */
  const drawCandidate = (kde: Kde, s: Stream): number[] => {
    const u = units(s, 1 + d)
    const r = kde.data[Math.min(kde.data.length - 1, Math.floor(u[0] * kde.data.length))]
    return coords.map((c, j) => {
      const h = kde.bw[j]
      if (c.kind === 'categorical') {
        const [keep, other] = units(child(s, 'category', j), 2)
        return keep < 1 - h ? r[j] : Math.min(c.count - 1, Math.floor(other * c.count))
      }
      const b = h * bandwidthFactor
      const lo = normalCdf(-r[j] / b) as number
      const hi = normalCdf((1 - r[j]) / b) as number
      return Math.min(1, Math.max(0, r[j] + b * (normalQuantile(lo + u[1 + j] * (hi - lo)) as number)))
    })
  }

  /** The model of every result so far: the largest budget with at least $N_{\min} + 2$ results, and its groups. */
  const modelOf = (
    evaluations: readonly MultiFidelityEvaluation[],
    points: readonly SpaceValues[],
  ): { budget: number; good: number[]; bad: number[]; l: Kde; g: Kde } | null => {
    const byBudget = new Map<number, MultiFidelityEvaluation[]>()
    for (const e of evaluations) byBudget.set(e.budget, [...(byBudget.get(e.budget) ?? []), e])
    const budgets = [...byBudget.keys()].filter((b) => byBudget.get(b)!.length >= minPoints + 2)
    if (budgets.length === 0) return null
    const budget = Math.max(...budgets)
    const rank = (v: number) => (Number.isNaN(v) ? Infinity : v)
    const ranked = [...byBudget.get(budget)!].sort((a, b) => rank(a.loss) - rank(b.loss) || a.config - b.config)
    const N = ranked.length
    const nGood = Math.max(minPoints, Math.floor(q * N + 1e-12))
    const nBad = Math.max(minPoints, N - nGood)
    const good = ranked.slice(0, nGood).map((e) => e.config)
    const bad = ranked.slice(N - nBad).map((e) => e.config)
    return {
      budget,
      good,
      bad,
      l: fitKde(good.map((c) => toModel(points[c]))),
      g: fitKde(bad.map((c) => toModel(points[c]))),
    }
  }

  /** Draws the configurations of bracket b, appending them to the points. */
  const drawBracket = (
    b: number,
    evaluations: readonly MultiFidelityEvaluation[],
    points: readonly SpaceValues[],
    random: readonly boolean[],
    s: Stream,
  ) => {
    const model = modelOf(evaluations, points)
    const next = [...points]
    const flags = [...random]
    for (const c of configsOf(b)) {
      const [u] = units(child(s, 'rho', c), 1)
      if (model === null || u < rho) {
        next.push(sample(child(s, 'random', c), space) as SpaceValues)
        flags.push(true)
        continue
      }
      const cs = child(s, 'candidates', c)
      let best: number[] | null = null
      let bestRatio = Infinity
      for (let k = 0; k < samples; k++) {
        const x = drawCandidate(model.l, child(cs, k))
        const ratio = Math.max(1e-32, pdf(model.g, x)) / Math.max(1e-32, pdf(model.l, x))
        if (best === null || ratio < bestRatio) {
          best = x
          bestRatio = ratio
        }
      }
      next.push(decode(space, toUnit(best!)) as SpaceValues)
      flags.push(false)
    }
    return {
      points: next,
      random: flags,
      sampled: configsOf(b),
      modelBudget: model?.budget ?? null,
      good: model?.good ?? [],
      bad: model?.bad ?? [],
    }
  }

  /** The observed loss of configuration `config` after `budget`, with its own noise. */
  const observe = (points: readonly SpaceValues[], noise: Stream, config: number, budget: number): number =>
    at(points[config] as ValuesOf<S>, budget, child(noise, config, budget))

  /** The true loss of a configuration at the maximum budget: noise-free when known, else observed. */
  const trueFinal = (points: readonly SpaceValues[], noise: Stream, config: number): number =>
    clean ? clean(points[config] as ValuesOf<S>, maxBudget) : observe(points, noise, config, maxBudget)

  /** The true and observed best of bracket b, and whether the true best is still in the race. */
  const bests = (
    state: BohbState,
    b: number,
  ): Pick<BohbState, 'trueBest' | 'observedBest' | 'bestRung' | 'survived'> => {
    const configs = configsOf(b)
    const argmin = (loss: (c: number) => number) => {
      let best = configs[0]
      let value = loss(best)
      for (const c of configs.slice(1)) {
        const v = loss(c)
        if (v < value) [best, value] = [c, v]
      }
      return best
    }
    const trueBest = argmin((c) => trueFinal(state.points, state.noise, c))
    const observedBest = argmin((c) => observe(state.points, state.noise, c, maxBudget))
    const reached = new Map<number, number>()
    const reach = (c: number, k: number) => reached.set(c, Math.max(reached.get(c) ?? -1, k))
    for (const e of state.evaluations) if (e.bracket === b) reach(e.config, e.rung)
    for (const p of state.promotions) if (p.bracket === b) reach(p.config, p.to)
    const top = Math.max(-1, ...reached.values())
    const bestRung = reached.get(trueBest) ?? -1
    return { trueBest, observedBest, bestRung, survived: bestRung === -1 || bestRung === top }
  }

  return {
    name: 'bohb',
    init: (_start, stream) => {
      const noise = child(stream, 'curve')
      const drawn = drawBracket(0, [], [], [], child(stream, 'sample'))
      const state: BohbState = {
        t: 0,
        bracket: 0,
        rung: -1,
        budget: 0,
        evaluated: [],
        losses: [],
        promoted: [],
        evaluations: [],
        promotions: [],
        spent: 0,
        time: 0,
        incumbent: null,
        trueBest: 0,
        observedBest: 0,
        bestRung: -1,
        survived: true,
        queue: { bracket: 0, rung: 0, configurations: configsOf(0) },
        running: [],
        started: 0,
        ...drawn,
        noise,
      }
      return { ...state, ...bests(state, 0) }
    },
    step: (state, ctx) => {
      if (state.queue === null) return { ...state, t: state.t + 1, sampled: [], terminated: true }
      const { bracket: b, rung: k, configurations } = state.queue
      const g = brackets[b].rungs[k]
      const free = new Float64Array(workers).fill(state.time)
      const added: MultiFidelityEvaluation[] = []
      for (const config of configurations) {
        let w = 0
        for (let v = 1; v < workers; v++) if (free[v] < free[w]) w = v
        let last = 0
        if (resume) for (const e of state.evaluations) if (e.config === config && e.budget > last) last = e.budget
        const cost = resume ? Math.max(0, g.budget - last) : g.budget
        added.push({
          config,
          bracket: b,
          rung: k,
          budget: g.budget,
          loss: observe(state.points, state.noise, config, g.budget),
          cost,
          worker: w,
          start: free[w],
          end: free[w] + cost,
        })
        free[w] += cost
      }
      const byLoss = (x: MultiFidelityEvaluation, y: MultiFidelityEvaluation) => x.loss - y.loss || x.config - y.config
      const promoted = [...added]
        .sort(byLoss)
        .slice(0, g.keep)
        .map((e) => e.config)
      const time = Math.max(state.time, ...free)
      const promotions: MultiFidelityPromotion[] = [
        ...state.promotions,
        ...promoted.map((config) => ({ config, bracket: b, from: k, to: k + 1, time })),
      ]
      const evaluations = [...state.evaluations, ...added]
      let incumbent: MultiFidelityIncumbent | null = state.incumbent
      for (const e of added)
        if (incumbent === null || e.loss < incumbent.loss || (e.loss === incumbent.loss && e.budget > incumbent.budget))
          incumbent = {
            config: e.config,
            loss: e.loss,
            budget: e.budget,
            final: trueFinal(state.points, state.noise, e.config),
          }
      let drawn: Pick<BohbState, 'points' | 'random' | 'sampled' | 'modelBudget' | 'good' | 'bad'> = {
        points: state.points,
        random: state.random,
        sampled: [],
        modelBudget: state.modelBudget,
        good: state.good,
        bad: state.bad,
      }
      let queue: BohbState['queue'] = null
      if (g.keep > 0) queue = { bracket: b, rung: k + 1, configurations: promoted }
      else if (b + 1 < brackets.length) {
        drawn = drawBracket(b + 1, evaluations, state.points, state.random, child(ctx.stream, 'sample'))
        queue = { bracket: b + 1, rung: 0, configurations: configsOf(b + 1) }
      }
      const next: BohbState = {
        ...state,
        ...drawn,
        t: state.t + 1,
        bracket: b,
        rung: k,
        budget: g.budget,
        evaluated: added.map((e) => e.config),
        losses: added.map((e) => e.loss),
        promoted,
        evaluations,
        promotions,
        spent: state.spent + added.reduce((a, e) => a + e.cost, 0),
        time,
        incumbent,
        queue,
        terminated: queue === null,
      }
      return { ...next, ...bests(next, b) }
    },
  }
}
