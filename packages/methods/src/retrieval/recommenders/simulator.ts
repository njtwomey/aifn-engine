/**
 * A feedback-loop simulator (Chaney, Stewart and Engelhardt, 2018; Mansoury et al., 2020): recommenders retrained each
 * round on the clicks their own recommendations produced. Users have fixed true click probabilities; each round every
 * user is shown a slate of K items they have not clicked, clicks each with its true probability, and the clicks join
 * the log the next round's model is trained on. Policies that rank by what was clicked before concentrate exposure on
 * the items that were shown early, which the exposure Gini coefficient measures round by round.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, stream, units, type Stream } from 'aifn-compute/foundation/random'
import { giniCoefficient } from 'aifn-methods/evaluation/beyond-accuracy'
import { alsFactors, factorScorer, implicitAls } from './factorisation'
import { popularity } from './neighbourhood'
import { topK, type Interactions, type Scorer } from './interactions'

/** The simulated world: every user's true click probability for every item (users × items, row-major). */
export type SimulatedWorld = { users: Size; items: Size; clickProbability: Float64Array }

/** How a policy ranks: by training popularity, by implicit-ALS factors, at random, or by the true probabilities. */
export type FeedbackPolicy = 'popularity' | 'matrix-factorisation' | 'random' | 'oracle'

/** Options of `feedbackLoop`. */
export type FeedbackLoopOptions = {
  world: SimulatedWorld
  /** Policies simulated side by side, each with its own log (default all four). */
  policies?: readonly FeedbackPolicy[]
  /** Rounds after the bootstrap round (default 30). */
  rounds?: Size
  /** Slate size K (default 5). */
  slate?: Size
  /** Share of each slate filled with random unclicked items (ε-exploration; default 0). */
  exploration?: number
  /** Implicit-ALS factors and sweeps per round for `matrix-factorisation` (default 8 and 3). */
  factors?: Size
  sweeps?: Size
  /** Root seed (default 'feedback'). The bootstrap round shows every policy the same random slates. */
  seed?: string | number
}

/** The curves of one policy over the rounds (index 0 is the bootstrap round). */
export type PolicyHistory = {
  /** Gini coefficient of cumulative exposure across items. */
  gini: number[]
  /** Share of cumulative exposure taken by the most-exposed 10% of items. */
  topShare: number[]
  /** Share of the catalogue shown to someone in the round. */
  coverage: number[]
  /** Mean true click probability of the shown items (the round's expected click-through rate). */
  ctr: number[]
  /** Cumulative clicks per user. */
  clicks: number[]
  /** Cumulative exposure of each item after the round. */
  exposure: Float64Array
}

/** A snapshot of `feedbackLoop` after a round. */
export type FeedbackSnapshot = {
  round: Size
  rounds: Size
  done: boolean
  policies: Record<string, PolicyHistory>
}

function scorerOf(
  policy: FeedbackPolicy,
  log: Interactions,
  world: SimulatedWorld,
  s: Stream,
  warm: { P: Float64Array; Q: Float64Array } | null,
  factors: Size,
  sweeps: Size,
): { score: Scorer; warm: { P: Float64Array; Q: Float64Array } | null } {
  const { items } = world
  if (policy === 'popularity') return { score: popularity(log), warm }
  if (policy === 'oracle')
    return {
      score: (us) => {
        const out = new Float64Array(us.length * items)
        us.forEach((u, r) => out.set(world.clickProbability.subarray(u * items, (u + 1) * items), r * items))
        return out
      },
      warm,
    }
  if (policy === 'random') return { score: (us) => Float64Array.from(units(s, us.length * items)), warm }
  // Implicit ALS warm-started from the previous round's factors.
  const alg = implicitAls(log, { factors, alpha: 5, regularisation: 0.5, init: warm ?? undefined })
  let state = alg.init(undefined, child(s, 'init'))
  for (let t = 0; t < sweeps; t++) state = alg.step(state, { t, stream: child(s, 'step', t) })
  const f = alsFactors(state)
  return { score: factorScorer(f), warm: { P: f.P, Q: f.Q } }
}

/**
 * Simulate the feedback loop round by round, yielding a snapshot after each: a generator, so a worker can stream it. In
 * round r every policy ranks the items each user has not clicked, shows the top K (the last ⌈εK⌉ slots replaced by
 * random items under exploration), and logs the clicks drawn from the true probabilities with `child(seed, r, u)`, the
 * same draws for every policy, so differences come from the slates alone.
 */
export function* feedbackLoop(options: FeedbackLoopOptions): Generator<FeedbackSnapshot> {
  const {
    world,
    policies = ['popularity', 'matrix-factorisation', 'random', 'oracle'],
    rounds = 30,
    slate: K = 5,
    exploration = 0,
    factors = 8,
    sweeps = 3,
    seed = 'feedback',
  } = options
  const { users, items, clickProbability } = world
  const root = stream(seed)
  type Run = {
    user: number[]
    item: number[]
    clicked: Set<number>[]
    exposure: Float64Array
    history: PolicyHistory
    warm: { P: Float64Array; Q: Float64Array } | null
  }
  const runs: Record<string, Run> = {}
  for (const p of policies)
    runs[p] = {
      user: [],
      item: [],
      clicked: Array.from({ length: users }, () => new Set<number>()),
      exposure: new Float64Array(items),
      history: { gini: [], topShare: [], coverage: [], ctr: [], clicks: [], exposure: new Float64Array(items) },
      warm: null,
    }
  const record = (run: Run, shown: Set<number>, ctr: number) => {
    const h = run.history
    h.gini.push(giniCoefficient(run.exposure))
    const sorted = Float64Array.from(run.exposure).sort().reverse()
    const top = Math.max(1, Math.round(items / 10))
    const total = sorted.reduce((a, b) => a + b, 0)
    h.topShare.push(sorted.slice(0, top).reduce((a, b) => a + b, 0) / Math.max(total, 1))
    h.coverage.push(shown.size / items)
    h.ctr.push(ctr)
    h.clicks.push(run.user.length / users)
    h.exposure = Float64Array.from(run.exposure)
  }
  /** Show slates and log clicks for one policy in one round. */
  const play = (run: Run, slates: number[][], round: number) => {
    const shown = new Set<number>()
    let p = 0
    slates.forEach((list, u) => {
      const draws = units(child(root, 'click', round, u), items)
      for (const i of list) {
        shown.add(i)
        run.exposure[i]++
        const q = clickProbability[u * items + i]
        p += q
        // The click of (u, i) in this round depends only on the round, the user and the item.
        if (draws[i] < q && !run.clicked[u].has(i)) {
          run.clicked[u].add(i)
          run.user.push(u)
          run.item.push(i)
        }
      }
    })
    record(
      run,
      shown,
      p /
        Math.max(
          1,
          slates.reduce((a, l) => a + l.length, 0),
        ),
    )
  }
  // Bootstrap: the same random slates for every policy.
  const bootstrap = Array.from({ length: users }, (_, u) => topK(units(child(root, 'bootstrap', u), items), K))
  for (const p of policies) play(runs[p], bootstrap, 0)
  const snapshot = (round: number): FeedbackSnapshot => ({
    round,
    rounds,
    done: round === rounds,
    policies: Object.fromEntries(
      policies.map((p) => {
        const h = runs[p].history
        return [
          p,
          {
            gini: [...h.gini],
            topShare: [...h.topShare],
            coverage: [...h.coverage],
            ctr: [...h.ctr],
            clicks: [...h.clicks],
            exposure: Float64Array.from(h.exposure),
          },
        ]
      }),
    ),
  })
  yield snapshot(0)
  for (let r = 1; r <= rounds; r++) {
    for (const p of policies) {
      const run = runs[p]
      const log: Interactions = { users, items, user: Int32Array.from(run.user), item: Int32Array.from(run.item) }
      const s = child(root, 'policy', p, r)
      const fitted = scorerOf(p, log, world, s, run.warm, factors, sweeps)
      run.warm = fitted.warm
      const all = Array.from({ length: users }, (_, u) => u)
      const S = fitted.score(all)
      const explore = Math.round(exploration * K)
      const slates = all.map((u) => {
        const ranked = topK(S.subarray(u * items, (u + 1) * items), K - explore, run.clicked[u])
        if (explore === 0) return ranked
        const taken = new Set([...ranked, ...run.clicked[u]])
        const extra = topK(units(child(s, 'explore', u), items), explore, taken)
        return [...ranked, ...extra]
      })
      play(run, slates, r)
    }
    yield snapshot(r)
  }
}

/** A world from user and item factors: click probability σ(scale · w_uᵀv_i + offset). */
export function worldFromFactors(
  userFactors: ArrayLike<number>,
  itemFactors: ArrayLike<number>,
  users: Size,
  items: Size,
  { scale = 2, offset = -2 }: { scale?: number; offset?: number } = {},
): SimulatedWorld {
  const k = userFactors.length / users
  const p = new Float64Array(users * items)
  for (let u = 0; u < users; u++)
    for (let i = 0; i < items; i++) {
      let s = 0
      for (let c = 0; c < k; c++) s += userFactors[u * k + c] * itemFactors[i * k + c]
      p[u * items + i] = 1 / (1 + Math.exp(-(scale * s + offset)))
    }
  return { users, items, clickProbability: p }
}
