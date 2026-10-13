/**
 * Population-based training (Jaderberg et al., 2017) as a step-through simulator: a population trained in parallel,
 * each member with its own hyperparameters, where every few steps the weaker members copy stronger ones (exploit)
 * and then change their hyperparameters (explore).
 *
 * The default problem is the toy of Jaderberg et al. (2017), Figure 2: maximise
 * $Q(\thetavec) = 1.2 - \theta_0^2 - \theta_1^2$ when training can only climb the surrogate
 *
 * $$ \hat Q(\thetavec \mid \hvec) = 1.2 - h_0\theta_0^2 - h_1\theta_1^2, $$
 *
 * by gradient steps $\thetavec \leftarrow \thetavec + \alpha \nabla_{\thetavec} \hat Q$, with $\hvec$ the
 * hyperparameters. Two members start at $\thetavec = (0.9, 0.9)$ with $\hvec = (1, 0)$ and $(0, 1)$: trained apart
 * (grid search) each can move one coordinate only and stalls at $Q = 1.2 - 0.81 = 0.39$, while with exploitation they
 * take turns, each copying the other's solution and moving the coordinate its own $\hvec$ allows, and reach the
 * optimum $Q = 1.2$. Exploit is truncation selection (Section 4.1.1): a member in the bottom fraction of the
 * population by score copies one drawn uniformly from the top fraction. Explore either perturbs the hyperparameters
 * or resamples them from the prior.
 */

import { units, type Stream } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Status } from 'aifn-compute/foundation/trace'

/** The problem a population is trained on: the training step, the score, and the hyperparameters' distribution. */
export interface PbtProblem {
  /** The parameters $\thetavec$ every member starts from. */
  theta: readonly number[]
  /** One training step of a member: its next parameters given its parameters and hyperparameters. */
  train: (theta: readonly number[], h: readonly number[]) => number[]
  /** The score of a member's parameters (higher is better): the evaluation PBT ranks by. */
  score: (theta: readonly number[]) => number
  /** What training itself climbs, reported beside the score when given. */
  surrogate?: (theta: readonly number[], h: readonly number[]) => number
  /** The hyperparameters of member `member` of a population of `size` at the start, drawn from `s` if random. */
  initial: (member: number, size: number, s: Stream) => number[]
  /** A draw from the hyperparameters' prior, for `explore: 'resample'`. */
  sample: (s: Stream) => number[]
  /** A perturbation of hyperparameters, for `explore: 'perturb'`. */
  perturb: (h: readonly number[], s: Stream) => number[]
}

/**
 * The toy problem of Jaderberg et al. (2017), Figure 2 (see the file comment): $Q$ scored, $\hat Q$ climbed by
 * gradient steps of size $\alpha$. A population of two starts from the paper's $\hvec = (1, 0)$ and $(0, 1)$; a larger
 * one starts from draws of the prior, uniform on $[0, 1]^2$. Explore's perturbation adds independent
 * $\Gauss(0, \sigma^2)$ noise to each $h_j$ (the paper's "slightly perturbs its update direction"), clipped at 0.
 *
 * @param options The step size and the perturbation.
 * @param options.stepSize The gradient step size $\alpha$ (default 0.05).
 * @param options.noise The standard deviation $\sigma$ of the perturbation (default 0.1).
 * @returns The problem.
 *
 * @example One gradient step from the start, and the scores
 * const p = pbtToyProblem()
 * const next = p.train([0.9, 0.9], [1, 0])
 * print('theta after one step with h = (1, 0):', next)
 * print('Q =', p.score(next), ' surrogate =', p.surrogate(next, [1, 0]))
 */
export function pbtToyProblem({
  stepSize = 0.05,
  noise = 0.1,
}: { stepSize?: number; noise?: number } = {}): PbtProblem {
  if (!(stepSize > 0)) throw new DomainError('pbtToyProblem', 'pbtToyProblem: stepSize must be positive')
  if (!(noise >= 0)) throw new DomainError('pbtToyProblem', 'pbtToyProblem: noise must be non-negative')
  const normal = (s: Stream) => {
    const [u1, u2] = units(s, 2)
    return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2)
  }
  return {
    theta: [0.9, 0.9],
    // The gradient of the surrogate is (−2 h0 θ0, −2 h1 θ1).
    train: (theta, h) => theta.map((v, j) => v - 2 * stepSize * h[j] * v),
    score: (theta) => 1.2 - theta[0] ** 2 - theta[1] ** 2,
    surrogate: (theta, h) => 1.2 - h[0] * theta[0] ** 2 - h[1] * theta[1] ** 2,
    initial: (member, size, s) => (size === 2 ? (member === 0 ? [1, 0] : [0, 1]) : Array.from(units(s, 2))),
    sample: (s) => Array.from(units(s, 2)),
    perturb: (h, s) => h.map((v) => Math.max(0, v + noise * normal(s))),
  }
}

/** Options of `populationBasedTrainingSteps`. */
export interface PbtOptions {
  /** The number of members (default 2). */
  population?: number
  /**
   * How members exploit: `'truncation'` (default), the bottom `fraction` copies from the top `fraction`; `'none'`,
   * never (with `explore: 'none'` too this is grid or random search: members trained apart).
   */
  exploit?: 'truncation' | 'none'
  /**
   * How a member that exploited explores: `'perturb'` (default) through the problem's perturbation, `'resample'` from
   * its prior, `'none'` not at all. With `exploit: 'none'` every member explores when ready.
   */
  explore?: 'perturb' | 'resample' | 'none'
  /** The training steps between exploit and explore, the paper's "ready" criterion (default 4). */
  readyInterval?: number
  /**
   * The fraction of the population in the top and the bottom for truncation selection (default 0.2, as the paper):
   * $\lceil \text{fraction} \cdot P \rceil$ members, at most $\lfloor P / 2 \rfloor$.
   */
  fraction?: number
  /**
   * What a member copies when it exploits: only its source's parameters (`'weights'`, default: the toy of Figure 2,
   * where the weaker worker "copies the solution of the better one" and keeps its own update direction), or the
   * parameters and the hyperparameters (`'all'`, as in the experiments of Section 4.1.1).
   */
  inherit?: 'all' | 'weights'
}

/** One exploit or explore event. */
export interface PbtEvent {
  /** The member it happened to. */
  member: number
  /** `'exploit'`: the member copied `source`; `'explore'`: its hyperparameters changed. */
  kind: 'exploit' | 'explore'
  /** The member copied from (exploit only). */
  source?: number
  /** The member's hyperparameters before. */
  before: number[]
  /** Its hyperparameters after. */
  after: number[]
}

/** One state of `populationBasedTrainingSteps`. */
export interface PbtState extends Status {
  /** Training steps taken. */
  t: number
  /** The members' parameters, $P \times d$ (one row per member). */
  theta: Tensor
  /** The members' hyperparameters, $P \times m$. */
  h: Tensor
  /** The members' scores, $P$ values (after this step's exploit and explore). */
  scores: Tensor
  /** The surrogate each member climbs, under its own hyperparameters ($P$ values; NaN when the problem has none). */
  surrogates: Tensor
  /** The best score in the population. */
  best: number
  /** The member with the best score. */
  bestMember: number
  /** Whether this step ended with exploit and explore. */
  ready: boolean
  /** This step's exploit and explore events, in the order they happened. */
  events: PbtEvent[]
}

/**
 * Population-based training as a step-through algorithm (see the file comment). Each step trains every member one
 * step; every `readyInterval` steps the members are ready together, and then (with `exploit: 'truncation'`) each
 * member of the bottom fraction, by score, copies a member drawn uniformly from the top fraction and explores from
 * there. Ties in score go to the lower member index. The draws come from the step's stream (and the initial
 * hyperparameters, where random, from the initial stream). The state holds every member's parameters,
 * hyperparameters and score and the step's events, so a trace draws the trajectories. A state whose scores are not
 * finite is flagged `diverged`. Throws `DomainError` for a population below 2, a `readyInterval` that is not a
 * positive integer, or a fraction outside $(0, 0.5]$.
 *
 * @param options The population, how it exploits and explores, and how often, as `PbtOptions`.
 * @param problem The problem (default `pbtToyProblem()`, the paper's toy).
 * @returns The algorithm, to run with `run` or `trace`; it takes no start (`undefined`).
 *
 * @example The toy of Jaderberg et al.: PBT against grid search
 * for (const exploit of ['truncation', 'none']) {
 *   const explore = exploit === 'none' ? 'none' : 'perturb'
 *   const s = run(populationBasedTrainingSteps({ exploit, explore }), undefined, 60, { stream: stream(0) })
 *   print(exploit === 'none' ? 'grid search:' : 'PBT:', ' best Q =', s.best.toFixed(4), ' h =', s.h)
 * }
 *
 * @example The events of the first ready step
 * const tr = trace(populationBasedTrainingSteps({}), undefined, 4, { stream: stream(1), keep: 'all' })
 * for (const e of tr.final.events) print(e.kind, 'member', e.member, e.source ?? '', e.before, '->', e.after)
 */
export function populationBasedTrainingSteps(
  options: PbtOptions = {},
  problem: PbtProblem = pbtToyProblem(),
): Algorithm<undefined, PbtState> {
  const where = 'populationBasedTrainingSteps'
  const {
    population = 2,
    exploit = 'truncation',
    explore = 'perturb',
    readyInterval = 4,
    fraction = 0.2,
    inherit = 'weights',
  } = options
  if (!(Number.isInteger(population) && population >= 2))
    throw new DomainError(where, `${where}: the population needs at least 2 members`)
  if (!(Number.isInteger(readyInterval) && readyInterval >= 1))
    throw new DomainError(where, `${where}: readyInterval must be a positive integer`)
  if (!(fraction > 0 && fraction <= 0.5)) throw new DomainError(where, `${where}: fraction must lie in (0, 0.5]`)
  const m = Math.min(Math.floor(population / 2), Math.ceil(fraction * population - 1e-12))
  const d = problem.theta.length

  const stateOf = (t: number, theta: number[][], h: number[][], ready: boolean, events: PbtEvent[]): PbtState => {
    const scores = theta.map((th) => problem.score(th))
    const surrogates = theta.map((th, i) => (problem.surrogate ? problem.surrogate(th, h[i]) : NaN))
    let bestMember = 0
    scores.forEach((v, i) => {
      if (v > scores[bestMember]) bestMember = i
    })
    const k = h[0].length
    return {
      t,
      theta: fromData(Float64Array.from(theta.flat()), [population, d]),
      h: fromData(Float64Array.from(h.flat()), [population, k]),
      scores: fromData(Float64Array.from(scores), [population]),
      surrogates: fromData(Float64Array.from(surrogates), [population]),
      best: scores[bestMember],
      bestMember,
      ready,
      events,
      diverged: !scores.every(Number.isFinite),
    }
  }
  const rows = (x: Tensor, cols: number): number[][] => {
    const v = Array.from(toFlat(x) as ArrayLike<number>)
    return Array.from({ length: x.shape[0] }, (_, i) => v.slice(i * cols, i * cols + cols))
  }

  return {
    name: 'population-based-training',
    init: (_start, s) => {
      const h = Array.from({ length: population }, (_, i) => problem.initial(i, population, s))
      return stateOf(
        0,
        h.map(() => [...problem.theta]),
        h,
        false,
        [],
      )
    },
    step: (state, ctx) => {
      const k = state.h.shape[1]
      const h = rows(state.h, k)
      let theta = rows(state.theta, d).map((th, i) => problem.train(th, h[i]))
      const t = state.t + 1
      const ready = t % readyInterval === 0
      const events: PbtEvent[] = []
      const exploreMember = (i: number) => {
        if (explore === 'none') return
        const before = [...h[i]]
        h[i] = explore === 'perturb' ? problem.perturb(h[i], ctx.stream) : problem.sample(ctx.stream)
        events.push({ member: i, kind: 'explore', before, after: [...h[i]] })
      }
      if (ready && exploit === 'truncation') {
        const scores = theta.map((th) => problem.score(th))
        const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b)
        const top = order.slice(0, m)
        const bottom = order.slice(population - m)
        const copied = theta.map((th) => [...th])
        for (const i of bottom) {
          const source = top[Math.min(m - 1, Math.floor(units(ctx.stream, 1)[0] * m))]
          const before = [...h[i]]
          copied[i] = [...theta[source]]
          if (inherit === 'all') h[i] = [...h[source]]
          events.push({ member: i, kind: 'exploit', source, before, after: [...h[i]] })
          exploreMember(i)
        }
        theta = copied
      } else if (ready && exploit === 'none') {
        for (let i = 0; i < population; i++) exploreMember(i)
      }
      return stateOf(t, theta, h, ready, events)
    },
  }
}
