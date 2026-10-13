/**
 * A multi-fidelity schedule run over given learning curves, as a step-through algorithm: successive halving and
 * Hyperband rung by rung, or asynchronously, one finished evaluation at a time, as ASHA (Li et al., 2020).
 *
 * The curves stand in for training: the loss of configuration $i$ after budget $r$ is read from them, so a run costs
 * nothing and the true answer is known. Lower losses are better. Synchronous successive halving evaluates every
 * configuration of a rung, then promotes the best $\lfloor n_k / \eta \rfloor$ (Jamieson and Talwalkar, 2016). ASHA
 * never waits for a rung to fill: whenever a worker is free it promotes a configuration in the top
 * $\lfloor \lvert \text{rung } k \rvert / \eta \rfloor$ of some rung $k$ that has not been promoted yet (trying the
 * highest rung first), and otherwise starts a new configuration at the bottom rung (Li et al., 2020, Algorithm 2).
 * Time is simulated: an evaluation takes as long as the budget it costs, on one of a number of workers.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { isTensor, tensor, toFlat, type NestedArray, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Status } from 'aifn-compute/foundation/trace'
import type { MultiFidelitySchedule } from './schedules'

/**
 * Learning curves as a table: the loss of each configuration at each of a list of budgets, read between the budgets
 * by linear interpolation. The output of `learningCurves` (`aifn-methods/data/synthetic`) is one, with its noise-free
 * `clean` table.
 */
export interface CurveTable {
  /** The budgets $t_1 < \dots < t_B$ the table holds. */
  budgets: Tensor | ArrayLike<number>
  /** The losses the search observes, $n \times B$: row $i$ is configuration $i$'s curve. */
  values: Tensor | NestedArray
  /**
   * The noise-free losses, $n \times B$ like `values`, when they are known: the true best configuration and the
   * incumbent's final loss are then read from them rather than from the observed `values`.
   */
  clean?: Tensor | NestedArray
}

/** Learning curves as a function: the loss of configuration `config` (from 0 to `count - 1`) after `budget`. */
export interface CurveFunction {
  /** The number of configurations. */
  count: number
  /** The loss of a configuration after a budget, as the search observes it. */
  at: (config: number, budget: number) => number
  /** The noise-free loss, when it is known; the true best and the incumbent's final loss are then read from it. */
  clean?: (config: number, budget: number) => number
}

/** One evaluation of a configuration at a budget. */
export interface MultiFidelityEvaluation {
  /** The configuration: a row of the curves. */
  config: number
  /** The bracket it belongs to. */
  bracket: number
  /** The rung it was evaluated at. */
  rung: number
  /** The budget it was evaluated at. */
  budget: number
  /** Its loss at that budget. */
  loss: number
  /** The budget the evaluation cost: `budget`, or the increase over its last budget when training resumes. */
  cost: number
  /** The worker that ran it. */
  worker: number
  /** The simulated time it started. */
  start: number
  /** The simulated time it finished. */
  end: number
}

/** One promotion of a configuration to the next rung. */
export interface MultiFidelityPromotion {
  /** The configuration. */
  config: number
  /** Its bracket. */
  bracket: number
  /** The rung it leaves. */
  from: number
  /** The rung it is promoted to. */
  to: number
  /** The simulated time of the decision. */
  time: number
}

/** The best evaluation so far. */
export interface MultiFidelityIncumbent {
  /** The configuration. */
  config: number
  /** Its loss: the lowest of any evaluation so far. */
  loss: number
  /** The budget it reached that loss at. */
  budget: number
  /** Its true loss at the schedule's maximum budget: from the noise-free curves when given, else the observed ones. */
  final: number
}

/** An evaluation in progress (asynchronous runs). */
export interface MultiFidelityJob {
  /** The bracket of the configuration. */
  bracket: number
  /** The worker running it. */
  worker: number
  /** The configuration. */
  config: number
  /** The rung it will join. */
  rung: number
  /** When it started. */
  start: number
  /** When it will finish. */
  end: number
}

/** One state of `multiFidelitySteps`. */
export interface MultiFidelityState extends Status {
  /** Steps taken: rungs completed (synchronous) or evaluations finished (asynchronous). */
  t: number
  /** The bracket of this step's evaluations (0 at the start). */
  bracket: number
  /** The rung of this step's evaluations ($-1$ at the start). */
  rung: number
  /** The budget of this step's evaluations (0 at the start). */
  budget: number
  /** The configurations evaluated at this step. */
  evaluated: number[]
  /** Their losses, in the same order. */
  losses: number[]
  /** The configurations promoted at this step (out of `rung`, when synchronous). */
  promoted: number[]
  /** Every evaluation so far, in the order they finished. */
  evaluations: MultiFidelityEvaluation[]
  /** Every promotion so far. */
  promotions: MultiFidelityPromotion[]
  /** The budget spent so far, the sum of the evaluations' costs. */
  spent: number
  /** The simulated time: the end of the last finished evaluation. */
  time: number
  /** The best evaluation so far (`null` at the start). */
  incumbent: MultiFidelityIncumbent | null
  /**
   * The configuration of this step's bracket with the lowest true loss at the maximum budget $R$: by the noise-free
   * curves when they are given (`clean`, as `learningCurves` provides, whose `best` it then agrees with), else by the
   * observed ones.
   */
  trueBest: number
  /**
   * The configuration of this step's bracket with the lowest observed loss at $R$ (the noisy value a full-budget
   * evaluation would see). It equals `trueBest` for noise-free curves and can differ when there is noise.
   */
  observedBest: number
  /** The highest rung `trueBest` has been evaluated at or promoted to ($-1$ before it starts). */
  bestRung: number
  /**
   * Whether `trueBest` is still in the race: it has not started yet, or it has reached the highest rung any
   * configuration of its bracket has reached. Synchronously, false once it has been left out of a promotion.
   */
  survived: boolean
  /** Synchronous runs: the rung to evaluate next (`null` when the schedule is done). */
  queue: { bracket: number; rung: number; configurations: number[] } | null
  /** Asynchronous runs: the evaluations in progress. */
  running: MultiFidelityJob[]
  /** Asynchronous runs: the configurations of the current bracket started so far. */
  started: number
}

/** Options of `multiFidelitySteps`. */
export interface MultiFidelityOptions {
  /** Run as ASHA, one finished evaluation per step (default false: synchronous, one rung per step). */
  asynchronous?: boolean
  /** The number of workers evaluating in parallel (default 1). */
  workers?: number
  /**
   * Whether a promoted configuration resumes its training (costing the increase in budget) rather than starting
   * again (costing the whole budget); default false.
   */
  resume?: boolean
}

/**
 * Reads curves into a loss function and a count, checking that every budget of the schedule lies within a table's
 * budgets.
 *
 * @param curves The curves.
 * @param schedule The schedule, whose budgets are checked.
 * @returns The configuration count, the observed loss function, and the noise-free one (`null` when not given).
 */
function readCurves(
  curves: CurveTable | CurveFunction,
  schedule: MultiFidelitySchedule,
): {
  count: number
  at: (config: number, budget: number) => number
  clean: ((config: number, budget: number) => number) | null
} {
  const where = 'multiFidelitySteps'
  if ('at' in curves) return { count: curves.count, at: curves.at, clean: curves.clean ?? null }
  const budgets = Float64Array.from(
    (isTensor(curves.budgets) ? toFlat(curves.budgets) : curves.budgets) as ArrayLike<number>,
  )
  const table = isTensor(curves.values) ? curves.values : tensor(curves.values)
  if (table.shape.length !== 2 || table.shape[1] !== budgets.length)
    throw new ShapeError(where, `${where}: the values must be n × ${budgets.length}, one column per budget`)
  for (let j = 1; j < budgets.length; j++)
    if (!(budgets[j] > budgets[j - 1])) throw new DomainError(where, `${where}: the budgets must increase`)
  const [count, B] = table.shape
  const cleanTable = curves.clean === undefined ? null : isTensor(curves.clean) ? curves.clean : tensor(curves.clean)
  if (
    cleanTable !== null &&
    (cleanTable.shape.length !== 2 || cleanTable.shape[0] !== count || cleanTable.shape[1] !== B)
  )
    throw new ShapeError(where, `${where}: the clean values must be ${count} × ${B}, as the values are`)
  for (const b of schedule.brackets)
    for (const g of b.rungs)
      if (!(g.budget >= budgets[0] * (1 - 1e-12) && g.budget <= budgets[B - 1] * (1 + 1e-12)))
        throw new DomainError(
          where,
          `${where}: budget ${g.budget} lies outside the curves' [${budgets[0]}, ${budgets[B - 1]}]`,
        )
  const reader = (values: ArrayLike<number>) => (config: number, budget: number) => {
    let j = 0
    while (j < B - 2 && budgets[j + 1] < budget) j++
    if (B === 1) return values[config]
    const [t0, t1] = [budgets[j], budgets[j + 1]]
    const w = Math.min(1, Math.max(0, (budget - t0) / (t1 - t0)))
    return (1 - w) * values[config * B + j] + w * values[config * B + j + 1]
  }
  return {
    count,
    at: reader(toFlat(table) as ArrayLike<number>),
    clean: cleanTable === null ? null : reader(toFlat(cleanTable) as ArrayLike<number>),
  }
}

/**
 * Runs a schedule over given learning curves as a step-through algorithm (see the file comment). Bracket $b$ takes
 * the next $n_0$ configurations of the curves in order, so the curves need at least `schedule.configurations` rows.
 * Synchronously each step evaluates one whole rung (the jobs spread over the workers) and promotes the best
 * $\lfloor n_k / \eta \rfloor$; brackets run one after another. Asynchronously (ASHA) each step finishes one
 * evaluation and hands the free workers their next jobs; a bracket ends when no job is left in it, and the next starts.
 * The state tracks the budget spent, the simulated time, the incumbent (the lowest loss seen, as Hyperband returns),
 * and whether the bracket's true best configuration, by its noise-free loss at the maximum budget when the curves give
 * it (`clean`) and by its observed loss otherwise, is still in the race; `observedBest` is the best by the observed
 * loss at the maximum budget, which noise can make differ. The run is deterministic: ties go to the lower
 * configuration index. Throws `DomainError` for too few curves, a schedule budget outside a table's budgets or a
 * worker count that is not a positive integer, and `ShapeError` for a table (or its clean table) of the wrong shape.
 *
 * @param curves The learning curves: a `CurveTable` (as `learningCurves` returns) or a `CurveFunction`.
 * @param schedule The schedule, from `successiveHalvingSchedule` or `hyperbandSchedule`.
 * @param options Asynchronous or not, the workers and whether training resumes, as `MultiFidelityOptions`.
 * @returns The algorithm, to run with `run` or `trace`; it takes no start (`undefined`) and draws nothing.
 *
 * @example Successive halving over nine curves: which configurations each rung keeps
 * // Configuration i ends at loss 0.1 * (9 - i) / 9 but starts badly when i is large: the curves cross.
 * const at = (i, r) => 0.1 * (9 - i) / 9 + (0.02 * i) / r
 * const schedule = successiveHalvingSchedule({ n: 9, minBudget: 1, maxBudget: 9, eta: 3 })
 * const tr = trace(multiFidelitySteps({ count: 9, at }, schedule), undefined, 10, { keep: 'all' })
 * for (const s of tr.steps.slice(1)) print('rung', s.rung, 'budget', s.budget, 'kept', s.promoted, 'spent', s.spent)
 * print('true best', tr.final.trueBest, 'survived?', tr.final.survived, ' incumbent', tr.final.incumbent.config)
 *
 * @example ASHA on four workers: each step is one finished evaluation
 * const at = (i, r) => 0.1 * ((i * 7) % 9) / 9 + 0.05 / r
 * const schedule = successiveHalvingSchedule({ n: 9, minBudget: 1, maxBudget: 9, eta: 3 })
 * const s = run(multiFidelitySteps({ count: 9, at }, schedule, { asynchronous: true, workers: 4 }), undefined, 100)
 * print('evaluations:', s.evaluations.map((e) => `${e.config}@${e.budget}`).join(' '))
 * print('time:', s.time, ' spent:', s.spent, ' incumbent:', s.incumbent.config)
 */
export function multiFidelitySteps(
  curves: CurveTable | CurveFunction,
  schedule: MultiFidelitySchedule,
  options: MultiFidelityOptions = {},
): Algorithm<undefined, MultiFidelityState> {
  const where = 'multiFidelitySteps'
  const { asynchronous = false, workers = 1, resume = false } = options
  if (!(Number.isInteger(workers) && workers >= 1))
    throw new DomainError(where, `${where}: workers must be a positive integer`)
  const { count, at, clean } = readCurves(curves, schedule)
  if (count < schedule.configurations)
    throw new DomainError(
      where,
      `${where}: the schedule starts ${schedule.configurations} configurations, the curves hold ${count}`,
    )
  const { eta, maxBudget, brackets } = schedule
  // The configurations of each bracket: consecutive rows of the curves.
  const offsets = brackets.map((_, b) => brackets.slice(0, b).reduce((a, x) => a + x.rungs[0].configurations, 0))
  const configsOf = (b: number) => Array.from({ length: brackets[b].rungs[0].configurations }, (_, i) => offsets[b] + i)
  // The true final loss (noise-free when known) and the observed one, at R.
  const finals = new Map<number, number>()
  const final = (config: number) => {
    if (!finals.has(config)) finals.set(config, (clean ?? at)(config, maxBudget))
    return finals.get(config) as number
  }
  const observed = (config: number) => at(config, maxBudget)
  const argmin = (configs: number[], loss: (c: number) => number) =>
    configs.reduce((best, c) => (loss(c) < loss(best) || (loss(c) === loss(best) && c < best) ? c : best))
  const trueBestOf = (b: number) => argmin(configsOf(b), final)
  const byLoss = (a: { config: number; loss: number }, b: { config: number; loss: number }) =>
    a.loss - b.loss || a.config - b.config

  // The cost of evaluating a configuration at a rung's budget, given its earlier evaluations.
  const costOf = (evaluations: readonly MultiFidelityEvaluation[], config: number, budget: number): number => {
    if (!resume) return budget
    let last = 0
    for (const e of evaluations) if (e.config === config && e.budget > last) last = e.budget
    return Math.max(0, budget - last)
  }

  const incumbentAfter = (
    incumbent: MultiFidelityIncumbent | null,
    added: readonly MultiFidelityEvaluation[],
  ): MultiFidelityIncumbent | null => {
    let best = incumbent
    for (const e of added)
      if (best === null || e.loss < best.loss || (e.loss === best.loss && e.budget > best.budget))
        best = { config: e.config, loss: e.loss, budget: e.budget, final: final(e.config) }
    return best
  }

  // Whether the bracket's true best is still in the race, and the highest rung it has reached.
  const race = (
    b: number,
    evaluations: readonly MultiFidelityEvaluation[],
    promotions: readonly MultiFidelityPromotion[],
    running: readonly MultiFidelityJob[],
  ) => {
    const best = trueBestOf(b)
    const reached = new Map<number, number>()
    const reach = (c: number, k: number) => reached.set(c, Math.max(reached.get(c) ?? -1, k))
    for (const e of evaluations) if (e.bracket === b) reach(e.config, e.rung)
    for (const p of promotions) if (p.bracket === b) reach(p.config, p.to)
    for (const j of running)
      if (offsets[b] <= j.config && j.config < offsets[b] + configsOf(b).length) reach(j.config, j.rung)
    const top = Math.max(-1, ...reached.values())
    const bestRung = reached.get(best) ?? -1
    return {
      trueBest: best,
      observedBest: argmin(configsOf(b), observed),
      bestRung,
      survived: bestRung === -1 || bestRung === top,
    }
  }

  const start = (): MultiFidelityState => ({
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
    trueBest: trueBestOf(0),
    observedBest: argmin(configsOf(0), observed),
    bestRung: -1,
    survived: true,
    queue: null,
    running: [],
    started: 0,
  })

  if (!asynchronous) {
    return {
      name: 'multi-fidelity',
      init: () => ({ ...start(), queue: { bracket: 0, rung: 0, configurations: configsOf(0) } }),
      step: (state) => {
        if (state.queue === null) return { ...state, t: state.t + 1, terminated: true }
        const { bracket: b, rung: k, configurations } = state.queue
        const g = brackets[b].rungs[k]
        // List scheduling: each job goes to the worker free soonest.
        const free = new Float64Array(workers).fill(state.time)
        const added: MultiFidelityEvaluation[] = []
        for (const config of configurations) {
          let w = 0
          for (let v = 1; v < workers; v++) if (free[v] < free[w]) w = v
          const cost = costOf(state.evaluations, config, g.budget)
          added.push({
            config,
            bracket: b,
            rung: k,
            budget: g.budget,
            loss: at(config, g.budget),
            cost,
            worker: w,
            start: free[w],
            end: free[w] + cost,
          })
          free[w] += cost
        }
        const ranked = [...added].sort(byLoss)
        const promoted = ranked.slice(0, g.keep).map((e) => e.config)
        const time = Math.max(state.time, ...free)
        const promotions = [
          ...state.promotions,
          ...promoted.map((config) => ({ config, bracket: b, from: k, to: k + 1, time })),
        ]
        const evaluations = [...state.evaluations, ...added]
        const queue =
          g.keep > 0
            ? { bracket: b, rung: k + 1, configurations: promoted }
            : b + 1 < brackets.length
              ? { bracket: b + 1, rung: 0, configurations: configsOf(b + 1) }
              : null
        return {
          ...state,
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
          incumbent: incumbentAfter(state.incumbent, added),
          ...race(b, evaluations, promotions, []),
          queue,
          terminated: queue === null,
        }
      },
    }
  }

  // ASHA: the next job of bracket b given its finished evaluations and promotions, or null when there is none.
  const nextJob = (
    b: number,
    evaluations: readonly MultiFidelityEvaluation[],
    promotions: readonly MultiFidelityPromotion[],
    started: number,
  ): { config: number; rung: number; from?: number } | null => {
    const rungs = brackets[b].rungs
    for (let k = rungs.length - 2; k >= 0; k--) {
      const done = evaluations.filter((e) => e.bracket === b && e.rung === k).sort(byLoss)
      const top = done.slice(0, Math.floor(done.length / eta))
      const gone = new Set(promotions.filter((p) => p.bracket === b && p.from === k).map((p) => p.config))
      const pick = top.find((e) => !gone.has(e.config))
      if (pick) return { config: pick.config, rung: k + 1, from: k }
    }
    if (started < rungs[0].configurations) return { config: offsets[b] + started, rung: 0 }
    return null
  }

  // Hands every idle worker a job of bracket b at `time`, moving on to the next bracket when b has no job left and
  // nothing running.
  const dispatch = (
    b: number,
    time: number,
    evaluations: readonly MultiFidelityEvaluation[],
    promotions: MultiFidelityPromotion[],
    running: MultiFidelityJob[],
    started: number,
  ): { running: MultiFidelityJob[]; started: number; promoted: number[] } => {
    const promoted: number[] = []
    for (;;) {
      for (let w = 0; w < workers; w++) {
        if (running.some((j) => j.worker === w)) continue
        const job = nextJob(b, evaluations, promotions, started)
        if (job === null) break
        if (job.from === undefined) started++
        else {
          promotions.push({ config: job.config, bracket: b, from: job.from, to: job.rung, time })
          promoted.push(job.config)
        }
        const budget = brackets[b].rungs[job.rung].budget
        const cost = costOf(evaluations, job.config, budget)
        running.push({ bracket: b, worker: w, config: job.config, rung: job.rung, start: time, end: time + cost })
      }
      if (running.length > 0 || b + 1 >= brackets.length) return { running, started, promoted }
      b++
      started = 0
    }
  }

  return {
    name: 'asha',
    init: () => {
      const promotions: MultiFidelityPromotion[] = []
      const d = dispatch(0, 0, [], promotions, [], 0)
      return { ...start(), running: d.running, started: d.started, promotions }
    },
    step: (state) => {
      if (state.running.length === 0) return { ...state, t: state.t + 1, terminated: true }
      let first = 0
      state.running.forEach((j, i) => {
        const f = state.running[first]
        if (j.end < f.end || (j.end === f.end && j.worker < f.worker)) first = i
      })
      const job = state.running[first]
      // Every running job is of the bracket being dispatched, since a bracket ends only when nothing runs.
      const b = job.bracket
      const budget = brackets[b].rungs[job.rung].budget
      const done: MultiFidelityEvaluation = {
        config: job.config,
        bracket: b,
        rung: job.rung,
        budget,
        loss: at(job.config, budget),
        cost: job.end - job.start,
        worker: job.worker,
        start: job.start,
        end: job.end,
      }
      const evaluations = [...state.evaluations, done]
      const promotions = [...state.promotions]
      const running = state.running.filter((_, i) => i !== first)
      const d = dispatch(b, job.end, evaluations, promotions, running, state.started)
      return {
        ...state,
        t: state.t + 1,
        bracket: b,
        rung: job.rung,
        budget,
        evaluated: [job.config],
        losses: [done.loss],
        promoted: d.promoted,
        evaluations,
        promotions,
        spent: state.spent + done.cost,
        time: job.end,
        incumbent: incumbentAfter(state.incumbent, [done]),
        ...race(b, evaluations, promotions, d.running),
        queue: null,
        running: d.running,
        started: d.started,
        terminated: d.running.length === 0,
      }
    },
  }
}
