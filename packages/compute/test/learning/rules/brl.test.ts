/**
 * Bayesian rule lists on the Titanic table: the empty list's marginal likelihood
 * lgamma(712) + lgamma(1491) - lgamma(2203) and prior -log sum 3^k/k!, the Letham-style list's captured counts and
 * rule probabilities (0.203, 0.423, 0.967, 0.880), and the sampler's states (each scored as `ruleListLogPosterior`
 * scores it, the best one tracked, reproducible from the seed).
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { logFactorial, logGamma } from 'aifn-compute/numerics/special'
import {
  bayesianRuleListTrace,
  capturedCounts,
  mineAntecedents,
  nominalConditions,
  ruleListLogPosterior,
  ruleProbabilities,
  ruleText,
} from 'aifn-compute/learning/rules'
import { titanicRows } from './titanic'

const t = titanicRows()
const ants = mineAntecedents(t.rows, nominalConditions([4, 2, 2]), { minSupport: 0.01, maxLength: 2 })
const index = (text: string) => ants.findIndex((a) => ruleText(a.conditions, t) === text)

describe('ruleListLogPosterior', () => {
  it('of the empty list: the Dirichlet-multinomial term and the truncated Poisson prior', () => {
    expect(ants).toHaveLength(26)
    const empty = ruleListLogPosterior([], ants, t.y, 2, { lambda: 3, eta: 1, alpha: 1 })
    expect(empty.logLikelihood).toBeCloseTo(logGamma(712) + logGamma(1491) - logGamma(2203), 6)
    let norm = 0
    for (let k = 0; k <= ants.length; k++) norm += Math.exp(k * Math.log(3) - logFactorial(k))
    expect(empty.logPrior).toBeCloseTo(-Math.log(norm), 10)
    expect(empty.counts).toEqual([[1490, 711]])
  })

  it('of the Letham-style list: counts and rule probabilities 0.203, 0.423, 0.967, 0.880', () => {
    const list = [index('sex = male ∧ age = adult'), index('class = 3rd'), index('class = 1st')]
    expect(list.every((j) => j >= 0)).toBe(true)
    const s = ruleListLogPosterior(list, ants, t.y, 2)
    expect(s.counts).toEqual([
      [1329, 338],
      [141, 103],
      [4, 146],
      [16, 124],
    ])
    expect(capturedCounts(list, ants, t.y, 2)).toEqual(s.counts)
    expect(ruleProbabilities(s.counts).map((p) => +p[1].toFixed(3))).toEqual([0.203, 0.423, 0.967, 0.88])
    expect(s.logPrior).toBeCloseTo(-10.321189, 5)
    expect(s.logLikelihood).toBeCloseTo(-1087.19912, 4)
    expect(s.logPosterior).toBeCloseTo(s.logPrior + s.logLikelihood, 12)
  })

  it('charges a one-rule list the length, cardinality and choice terms', () => {
    const empty = ruleListLogPosterior([], ants, t.y, 2)
    const one = ruleListLogPosterior([index('class = 3rd')], ants, t.y, 2)
    // log(3/1!) for the length, log(1 / (1 + 1/2)) for the cardinality (sizes 1 and 2 left), log(1/8) for the choice.
    expect(one.logPrior - empty.logPrior).toBeCloseTo(Math.log(3) + Math.log(1 / 1.5) + Math.log(1 / 8), 10)
  })
})

describe('bayesianRuleListTrace', () => {
  const states = bayesianRuleListTrace(ants, t.y, 2, stream(11), { steps: 300 })

  it('has one state per step, each scored by the posterior', () => {
    expect(states).toHaveLength(301)
    expect(states[0]).toMatchObject({ t: 0, proposal: 'start', accepted: true, list: [] })
    for (const s of states.filter((_, k) => k % 37 === 0))
      expect(s.logPosterior).toBeCloseTo(ruleListLogPosterior(s.list, ants, t.y, 2).logPosterior, 9)
  })

  it('tracks the best list and moves only on acceptance', () => {
    let best = -Infinity
    for (let k = 1; k < states.length; k++) {
      const s = states[k]
      if (!s.accepted) expect(s.list).toEqual(states[k - 1].list)
      else expect(s.list).toEqual(s.proposed)
      best = Math.max(best, s.logPosterior)
      expect(s.bestLogPosterior).toBeCloseTo(Math.max(best, states[0].logPosterior), 12)
    }
  })

  it('is reproducible from the seed', () => {
    const again = bayesianRuleListTrace(ants, t.y, 2, stream(11), { steps: 300 })
    expect(again.map((s) => s.logPosterior)).toEqual(states.map((s) => s.logPosterior))
    const last = states[states.length - 1]
    expect(states.filter((s) => s.accepted)).toHaveLength(77)
    expect(last.bestLogPosterior).toBeCloseTo(-1108.180906, 5)
    expect(last.best).toEqual([12, 20, 14, 1, 8, 4, 0])
  })
})
