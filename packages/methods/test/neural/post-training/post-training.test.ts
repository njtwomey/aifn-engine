/**
 * Post-training: the advantage variants against their formulas; pass@k against brute-force counting; the KL
 * estimators' expectations; GRPO's first step against the policy-gradient formula, its determinism, clipping and KL
 * penalty; the toy policy under SFT, DPO, IPO and GRPO; the coverage problem's sharpening.
 */
import { describe, expect, it } from 'vitest'
import { categorical, child, stream } from 'aifn-compute/foundation/random'
import {
  batchAdvantages,
  coverageProblem,
  groupAdvantages,
  grpoTrace,
  klEstimators,
  passAtK,
  passAtKExact,
  postTrainingTrace,
  solveProbabilities,
  tokenWeights,
  type GroupPolicyProblem,
} from 'aifn-methods/neural/post-training'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, digits = 12) =>
  Array.from(a).forEach((v, i) => expect(v).toBeCloseTo(b[i], digits))

describe('advantages', () => {
  const r = [1, 0, 0, 1, 1]
  const mean = 0.6
  const std = Math.sqrt(0.24)

  it('GRPO: (r − mean) / (std + eps)', () => {
    close(
      groupAdvantages(r),
      r.map((v) => (v - mean) / (std + 1e-4)),
    )
    close(
      groupAdvantages(r, { ddof: 1, eps: 0 }),
      r.map((v) => (v - mean) / Math.sqrt(0.3)),
    )
  })
  it('RLOO: the leave-one-out baseline is G/(G − 1) times the centred reward', () => {
    close(
      groupAdvantages(r, { baseline: 'leave-one-out', scale: 'none' }),
      r.map((v) => (5 / 4) * (v - mean)),
    )
  })
  it('Dr. GRPO keeps the centred reward; a group of equal rewards is zero under every option', () => {
    close(
      groupAdvantages(r, { scale: 'none' }),
      r.map((v) => v - mean),
    )
    for (const o of [{}, { scale: 'none' as const }, { baseline: 'leave-one-out' as const }])
      expect(Array.from(groupAdvantages([2, 2, 2], o))).toEqual([0, 0, 0])
  })
  it('batch scale: centred within groups, divided by the spread of every centred reward', () => {
    const groups = [
      [1, 0, 0, 0],
      [5, 0, 5, 0],
    ]
    const centred = [
      [0.75, -0.25, -0.25, -0.25],
      [2.5, -2.5, 2.5, -2.5],
    ]
    const all = centred.flat()
    const s = Math.sqrt(
      all.reduce((a, v) => a + v * v, 0) / all.length - (all.reduce((a, v) => a + v, 0) / all.length) ** 2,
    )
    batchAdvantages(groups).forEach((g, i) =>
      close(
        g,
        centred[i].map((v) => v / (s + 1e-4)),
      ),
    )
    expect(() => groupAdvantages(r, { scale: 'batch-std' })).toThrow(/batch/)
  })
  it('token weights: per-response mean, token mean and constant', () => {
    close(tokenWeights([1, -1], [10, 40], 'sequence-mean'), [1 / 10 / 2, -1 / 40 / 2])
    close(tokenWeights([1, -1], [10, 40], 'token-mean'), [1 / 50, -1 / 50])
    close(tokenWeights([1, -1], [10, 40], 'constant', 100), [0.01, -0.01])
  })
})

describe('KL estimators', () => {
  it('k1 and k3 are unbiased for KL(π ‖ π_ref); k3 is never negative', () => {
    const p = [0.5, 0.3, 0.15, 0.05]
    const q = [0.25, 0.25, 0.25, 0.25]
    const exact = p.reduce((a, pi, i) => a + pi * Math.log(pi / q[i]), 0)
    const mean = (k: 'k1' | 'k2' | 'k3') =>
      p.reduce((a, pi, i) => a + pi * klEstimators(Math.log(pi), Math.log(q[i]))[k], 0)
    expect(mean('k1')).toBeCloseTo(exact, 12)
    expect(mean('k3')).toBeCloseTo(exact, 12)
    for (const [a, b] of [
      [-0.1, -3],
      [-3, -0.1],
    ])
      expect(klEstimators(a, b).k3).toBeGreaterThanOrEqual(0)
  })
})

describe('pass@k', () => {
  it('matches counting the k-subsets that hold a correct sample', () => {
    const choose = (n: number, k: number): number => (k === 0 ? 1 : (n * choose(n - 1, k - 1)) / k)
    for (const [n, c, k] of [
      [6, 2, 3],
      [10, 1, 1],
      [10, 0, 4],
      [8, 5, 4],
      [20, 3, 7],
    ])
      expect(passAtK(n, c, k)).toBeCloseTo(1 - choose(n - c, k) / choose(n, k), 12)
    expect(passAtK(10, 3, 1)).toBeCloseTo(0.3, 12)
    expect(() => passAtK(5, 1, 6)).toThrow()
  })
  it('the exact form is 1 − (1 − p)^k, accurate for small p', () => {
    expect(passAtKExact(0.2, 3)).toBeCloseTo(1 - 0.8 ** 3, 14)
    expect(passAtKExact(1e-12, 10)).toBeCloseTo(1e-11, 20)
    expect(passAtKExact(1, 1)).toBe(1)
    expect(passAtKExact(0.5, 0)).toBe(0)
  })
})

describe('grpoTrace', () => {
  const problem = coverageProblem({ prompts: 12, strategies: 5, density: 0.4, seed: 3 })

  it('is deterministic from its seed, and another seed draws other groups', () => {
    const run = (seed: number) => [
      ...grpoTrace(problem, new Float64Array(5), { groupSize: 6, learningRate: 0.1, seed }, 5),
    ]
    expect(run(1).map((s) => Array.from(s.theta))).toEqual(run(1).map((s) => Array.from(s.theta)))
    expect(run(2)[5].theta).not.toEqual(run(1)[5].theta)
  })

  it('takes the policy-gradient step on its first update: SGD moves θ_a by η/N times the advantages of a’s samples', () => {
    // One prompt with logits = θ, so ∇ log π(a) = e_a − π; with a group-mean baseline the advantages sum to zero and
    // the step is Δθ_a = (η/N) Σ_{i: a_i = a} A_i. The group is redrawn from the same stream the step draws from.
    const rewards = [1, 0, 0.5, 0]
    const single: GroupPolicyProblem = { prompts: 1, actions: 4, logits: (t) => t, reward: (_x, a) => rewards[a] }
    const theta0 = [0.2, -0.1, 0, 0.3]
    const [s0, s1] = [...grpoTrace(single, theta0, { groupSize: 8, learningRate: 0.5, seed: 4, optimiser: 'sgd' }, 1)]
    const g = child(child(stream(4), 'step', 1), 'group', 0)
    const actions = Array.from({ length: 8 }, () => categorical(g, s0.probs[0]))
    const adv = groupAdvantages(actions.map((a) => rewards[a]))
    const expected = theta0.map(
      (t, a) => t + (0.5 / 8) * actions.reduce((acc, b, i) => acc + (b === a ? adv[i] : 0), 0),
    )
    close(s1.theta, expected, 12)
    // At the old policy every ratio is 1, so nothing is clipped on the first update.
    expect(s1.clipFraction).toBe(0)
  })

  it('raises the expected reward, and the KL penalty holds the policy nearer the start', () => {
    const free = [...grpoTrace(problem, new Float64Array(5), { groupSize: 8, learningRate: 0.1, seed: 1 }, 40)]
    const held = [...grpoTrace(problem, new Float64Array(5), { groupSize: 8, learningRate: 0.1, seed: 1, beta: 1 }, 40)]
    expect(free[40].meanReward).toBeGreaterThan(free[0].meanReward)
    expect(held[40].kl).toBeLessThan(free[40].kl)
    expect(free[0].kl).toBe(0)
  })

  it('clips only when it takes several updates per batch', () => {
    const many = [
      ...grpoTrace(problem, new Float64Array(5), { groupSize: 8, learningRate: 0.3, seed: 1, updatesPerBatch: 4 }, 10),
    ]
    expect(Math.max(...many.map((s) => s.clipFraction))).toBeGreaterThan(0)
    const one = [...grpoTrace(problem, new Float64Array(5), { groupSize: 8, learningRate: 0.3, seed: 1 }, 10)]
    expect(Math.max(...one.map((s) => s.clipFraction))).toBe(0)
  })

  it('reports the groups with equal rewards; dynamic sampling skips them', () => {
    const trace = [...grpoTrace(problem, new Float64Array(5), { groupSize: 4, learningRate: 0.1, seed: 1 }, 5)]
    expect(trace.slice(1).every((s) => s.fracZeroStd > 0 && s.fracZeroStd < 1)).toBe(true)
    const all1: GroupPolicyProblem = { prompts: 2, actions: 3, logits: (t) => t, reward: () => 1 }
    const still = [
      ...grpoTrace(all1, [0, 0, 0], { groupSize: 4, learningRate: 0.1, seed: 1, dynamicSampling: true }, 3),
    ]
    expect(still[3].theta).toEqual(still[0].theta)
    expect(still[3].fracZeroStd).toBe(1)
  })
})

describe('coverage: RLVR sharpens', () => {
  it('raises pass@1 and lowers pass@256 averaged over prompts', () => {
    const problem = coverageProblem({ prompts: 40, strategies: 8, density: 0.25, seed: 1 })
    const trace = [...grpoTrace(problem, new Float64Array(8), { groupSize: 8, learningRate: 0.1, seed: 1 }, 80)]
    const passAt = (probs: Float64Array[], k: number) =>
      solveProbabilities(problem, probs).reduce((a, p) => a + passAtKExact(p, k), 0) / 40
    expect(passAt(trace[80].probs, 1)).toBeGreaterThan(passAt(trace[0].probs, 1))
    expect(passAt(trace[80].probs, 256)).toBeLessThan(passAt(trace[0].probs, 256))
    // A prompt no strategy solves is never solved; the solves matrix has the asked density, roughly.
    const ones = problem.solves.reduce((a, b) => a + b, 0) / problem.solves.length
    expect(ones).toBeGreaterThan(0.15)
    expect(ones).toBeLessThan(0.35)
  })
})

describe('postTrainingTrace', () => {
  const policy = {
    features: [
      [1, 0],
      [0, 1],
      [1, 1],
      [-1, 0],
    ],
    theta0: [0, 0],
  }

  it('SFT drives the target response towards probability 1, and the KL from 0 up', () => {
    const trace = postTrainingTrace(policy, { method: 'sft', targets: [2], learningRate: 0.1, steps: 300, seed: 1 })
    expect(trace).toHaveLength(301)
    expect(trace[0].loss).toBeCloseTo(Math.log(4), 12)
    expect(trace[0].kl).toBe(0)
    expect(trace[300].probs[2]).toBeGreaterThan(0.95)
    expect(trace[300].kl).toBeGreaterThan(1)
  })

  it('DPO: loss log 2 at the start, then accuracy 1 and a growing margin', () => {
    const trace = postTrainingTrace(policy, {
      method: 'dpo',
      pairs: [
        { chosen: 2, rejected: 3 },
        { chosen: 1, rejected: 0 },
      ],
      beta: 0.5,
      learningRate: 0.05,
      steps: 150,
      seed: 1,
    })
    expect(trace[0].loss).toBeCloseTo(Math.LN2, 12)
    expect(trace[0].margin).toBe(0)
    expect(trace[150].accuracy).toBe(1)
    expect(trace[150].margin!).toBeGreaterThan(trace[50].margin!)
    expect(trace[150].logpChosen!).toBeGreaterThan(trace[0].logpChosen!)
  })

  it('IPO settles the log-ratio margin at 1/(2τ) instead of growing it', () => {
    const tau = 0.5
    const trace = postTrainingTrace(policy, {
      method: 'dpo',
      loss: 'ipo',
      pairs: [{ chosen: 2, rejected: 3 }],
      beta: tau,
      learningRate: 0.05,
      steps: 1500,
      seed: 1,
    })
    // margin = τ · h, so h = margin / τ → 1/(2τ).
    expect(trace[1500].margin! / tau).toBeCloseTo(1 / (2 * tau), 3)
  })

  it('GRPO raises the rewarded responses, deterministically from the seed', () => {
    const opts = {
      method: 'grpo' as const,
      rewards: [0, 1, 1, 0],
      groupSize: 8,
      learningRate: 0.05,
      steps: 100,
      seed: 1,
    }
    const a = postTrainingTrace(policy, opts)
    const b = postTrainingTrace(policy, opts)
    expect(a.map((s) => Array.from(s.probs))).toEqual(b.map((s) => Array.from(s.probs)))
    expect(a[100].probs[1] + a[100].probs[2]).toBeGreaterThan(0.9)
    expect(Number.isNaN(a[0].rewardMean!)).toBe(true)
    expect(a[1].rewardStd!).toBeGreaterThanOrEqual(0)
  })

  it('rejects responses that are not the policy’s', () => {
    expect(() =>
      postTrainingTrace(policy, { method: 'sft', targets: [4], learningRate: 0.1, steps: 1, seed: 1 }),
    ).toThrow(/response/)
  })
})
