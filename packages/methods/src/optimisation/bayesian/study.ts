/**
 * A multi-seed study of Bayesian optimisation on a registered objective: every arm (a set of loop options, such as an
 * acquisition or a confidence-bound weight) is run from every seed, and the curves a regret figure plots are returned
 * as plain tensors, so the whole study can run in a worker.
 *
 * With $f^*$ the objective's known minimum and $f(\xvec_i)$ its noise-free value at the $i$-th evaluation, the
 * instantaneous regret is $r_i = f(\xvec_i) - f^*$, the simple regret after $n$ evaluations is
 * $\min_{i \le n} f(\xvec_i) - f^*$ (the best-so-far value less the minimum) and the cumulative regret is
 * $R_n = \sum_{i \le n} r_i$ (Srinivas et al., 2010), each counted over every evaluation, the initial design included.
 */

import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { DomainError } from 'aifn-compute/foundation/errors'
import type { SearchBox } from './common'
import {
  bayesianOptimisationSteps,
  type AcquisitionName,
  type BayesianOptimisationOptions,
  type BayesianOptimisationState,
} from './loop'
import { resolveObjective, type NamedObjective } from './named'

/** One arm of a study: loop options, with a label for the legend. */
export type StudyArm = BayesianOptimisationOptions & {
  /** The arm's label (default the acquisition, with $\beta$ for `'ucb'`). */
  label?: string
}

/** The options of `bayesianOptimisationStudy`. */
export interface BayesianOptimisationStudyOptions extends BayesianOptimisationOptions {
  /** The registered objective, by name or with its parameters (`{ name: 'svm-tuning-surface', noise: 0.02 }`). */
  objective: NamedObjective
  /** The search box (default the objective's own). */
  box?: SearchBox
  /** The seeds: a count $S$ (seeds 0 to $S - 1$, default 5) or the seeds themselves; run $s$ uses `stream(seed)`. */
  seeds?: number | readonly number[]
  /** The steps of every run (default 20), each one evaluation (or one batch). */
  iterations?: number
  /**
   * The arms, each a set of options over the shared ones; left out, one arm per entry of `acquisitions` and of `betas`
   * (in that order), or a single arm of the shared options when neither is given.
   */
  arms?: readonly StudyArm[]
  /** One arm per acquisition. */
  acquisitions?: readonly AcquisitionName[]
  /** One `'ucb'` arm per exploration weight $\beta$. */
  betas?: readonly number[]
}

/** The curves of a study, plain tensors: arms by seeds by evaluations. */
export interface BayesianOptimisationStudy {
  /** The arms' labels, $A$ of them. */
  labels: string[]
  /** The seeds, $S$ of them. */
  seeds: number[]
  /** The objective's registry key. */
  objective: string
  /** Its known minimum value $f^*$. */
  minimum: number
  /** The evaluation counts $1, \dots, N$ the curves are indexed by, `[N]`. */
  evaluations: Tensor
  /** The size $n_0$ of every run's initial design (the largest over the arms). */
  initial: number
  /** The points evaluated, $A \times S \times N \times d$ (NaN past a shorter run's end). */
  x: Tensor
  /** The values observed (with any evaluation noise), $A \times S \times N$. */
  observed: Tensor
  /** The noise-free values $f(\xvec_i)$, $A \times S \times N$. */
  values: Tensor
  /** The best noise-free value so far, $\min_{i \le n} f(\xvec_i)$, $A \times S \times N$. */
  bestSoFar: Tensor
  /** The simple regret, `bestSoFar` less $f^*$, $A \times S \times N$. */
  simpleRegret: Tensor
  /** The cumulative regret $R_n = \sum_{i \le n} (f(\xvec_i) - f^*)$, $A \times S \times N$. */
  cumulativeRegret: Tensor
  /** The simple regret averaged over the seeds, $A \times N$. */
  meanSimpleRegret: Tensor
  /** The cumulative regret averaged over the seeds, $A \times N$. */
  meanCumulativeRegret: Tensor
}

/**
 * Run Bayesian optimisation on a registered objective from several seeds and for several arms, and return the
 * best-so-far, simple-regret and cumulative-regret curves as plain tensors (`BayesianOptimisationStudy`). Arm $a$ from
 * seed $s$ is `bayesianOptimisationSteps(objective, box, { ...shared, ...arms[a] })` run for `iterations` steps from
 * `stream(seeds[s])`, so every arm starts from the same initial design for a seed (common random numbers). Regret is
 * measured with the noise-free values against the known minimum, and the study only minimises. The arguments and the
 * result are plain data, so it is meant to run in a worker:
 * `call('optimisation/bayesian/bayesianOptimisationStudy', { objective: 'forrester', seeds: 5, betas: [1, 4] })`.
 * Throws `DomainError` for `minimise: false`, an unknown objective, or fewer than one seed or iteration.
 *
 * @param options The objective, box, seeds, iterations and arms, and the loop options all arms share, as
 *   `BayesianOptimisationStudyOptions`.
 * @returns The curves, arms by seeds by evaluations, with their means over the seeds.
 *
 * @example Expected improvement against the confidence bound on Forrester, over three seeds
 * const study = bayesianOptimisationStudy({
 *   objective: 'forrester', seeds: 3, iterations: 4, grid: 51, refit: false, acquisitions: ['ei'], betas: [4],
 * })
 * print('arms:', study.labels, ' evaluations:', study.evaluations)
 * print('mean simple regret:', study.meanSimpleRegret)
 */
export function bayesianOptimisationStudy(options: BayesianOptimisationStudyOptions): BayesianOptimisationStudy {
  const where = 'bayesianOptimisationStudy'
  const { objective, box, seeds = 5, iterations = 20, arms, acquisitions, betas, ...shared } = options
  if (shared.minimise === false) throw new DomainError(where, `${where}: a study minimises`)
  const seedList = typeof seeds === 'number' ? Array.from({ length: seeds }, (_, i) => i) : [...seeds]
  if (seedList.length < 1) throw new DomainError(where, `${where}: needs at least one seed`)
  if (!(Number.isInteger(iterations) && iterations >= 1))
    throw new DomainError(where, `${where}: iterations must be a positive integer`)
  const resolved = resolveObjective(objective, where)
  const armList: StudyArm[] = arms
    ? [...arms]
    : [
        ...(acquisitions ?? []).map((a) => ({ acquisition: a })),
        ...(betas ?? []).map((b) => ({ acquisition: 'ucb' as const, beta: b })),
      ]
  if (armList.length === 0) armList.push({})
  const labels = armList.map((a) => {
    if (a.label) return a.label
    const acq = a.acquisition ?? shared.acquisition ?? 'ei'
    const beta = a.beta ?? shared.beta ?? 4
    return acq === 'ucb' ? `ucb (beta ${beta})` : acq
  })
  const A = armList.length
  const S = seedList.length
  const finals: BayesianOptimisationState[] = []
  let initial = 0
  for (const arm of armList) {
    const armOptions: BayesianOptimisationOptions = { ...shared, ...arm }
    delete (armOptions as StudyArm).label
    const alg = bayesianOptimisationSteps(objective, box ?? resolved.fn, armOptions)
    const n0 = armOptions.init ?? 2 * alg.box.lo.shape[0] + 1
    initial = Math.max(initial, typeof n0 === 'number' ? n0 : n0.shape[0])
    for (const seed of seedList) finals.push(run(alg, undefined, iterations, { stream: stream(seed) }))
  }
  const N = Math.max(...finals.map((f) => f.evaluations))
  const d = finals[0].x.shape[1]
  const fStar = resolved.fn.minimumValue
  const x = new Float64Array(A * S * N * d).fill(NaN)
  const observed = new Float64Array(A * S * N).fill(NaN)
  const values = new Float64Array(A * S * N).fill(NaN)
  const bestSoFar = new Float64Array(A * S * N).fill(NaN)
  const simple = new Float64Array(A * S * N).fill(NaN)
  const cumulative = new Float64Array(A * S * N).fill(NaN)
  const meanSimple = new Float64Array(A * N)
  const meanCumulative = new Float64Array(A * N)
  finals.forEach((f, r) => {
    const xs = toFlat(f.x)
    const ys = toFlat(f.y)
    let best = Infinity
    let total = 0
    for (let i = 0; i < f.evaluations; i++) {
      const o = r * N + i
      const point = fromData(Float64Array.from(xs.slice(i * d, i * d + d)), [d])
      const v = resolved.clean(point)
      x.set(xs.slice(i * d, i * d + d), o * d)
      observed[o] = ys[i]
      values[o] = v
      best = Math.min(best, v)
      total += v - fStar
      bestSoFar[o] = best
      simple[o] = best - fStar
      cumulative[o] = total
    }
  })
  for (let a = 0; a < A; a++)
    for (let i = 0; i < N; i++) {
      for (let s = 0; s < S; s++) {
        meanSimple[a * N + i] += simple[(a * S + s) * N + i] / S
        meanCumulative[a * N + i] += cumulative[(a * S + s) * N + i] / S
      }
    }
  return {
    labels,
    seeds: seedList,
    objective: resolved.key,
    minimum: fStar,
    evaluations: fromData(
      Float64Array.from({ length: N }, (_, i) => i + 1),
      [N],
    ),
    initial,
    x: fromData(x, [A, S, N, d]),
    observed: fromData(observed, [A, S, N]),
    values: fromData(values, [A, S, N]),
    bestSoFar: fromData(bestSoFar, [A, S, N]),
    simpleRegret: fromData(simple, [A, S, N]),
    cumulativeRegret: fromData(cumulative, [A, S, N]),
    meanSimpleRegret: fromData(meanSimple, [A, N]),
    meanCumulativeRegret: fromData(meanCumulative, [A, N]),
  }
}
