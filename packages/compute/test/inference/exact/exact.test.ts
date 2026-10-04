import { describe, expect, it } from 'vitest'
import {
  chainForwardBackward,
  chainSumProduct,
  chainViterbi,
  enumerate,
  enumerationSteps,
  factorChain,
  forwardBackward,
  forwardBackwardSteps,
  jointDistribution,
  sampleHiddenPath,
  variableElimination,
  variableEliminationSteps,
  viterbi,
  viterbiSteps,
} from 'aifn-compute/inference/exact'
import { discreteFactor, discreteFactorGraph, toDiscreteFactorGraph } from 'aifn-compute/inference/model'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, log, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'
import { casino, casinoBrute, casinoChain, isingGrid, randomTree, sprinkler, sprinklerBrute } from '../graphs'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

describe('enumeration and variable elimination', () => {
  const bindings = { data: { wet: 1 } }
  const brute = sprinklerBrute()
  it('enumeration of the sprinkler factor graph gives the brute-force posterior and evidence', () => {
    const { graph, keys, logConstant } = toDiscreteFactorGraph(sprinkler, bindings)
    expect(keys).toEqual(['cloudy', 'sprinkler', 'rain'])
    const r = enumerate(graph)
    expect(r.marginals[2].data[1]).toBeCloseTo(brute.rain, 12)
    expect(r.marginals[0].data[1]).toBeCloseTo(brute.cloudy, 12)
    expect(r.logZ + logConstant).toBeCloseTo(Math.log(brute.evidence), 12)
    expect(toFlat(jointDistribution(graph)).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('variable elimination agrees with enumeration for every order heuristic', () => {
    const { graph } = toDiscreteFactorGraph(sprinkler, bindings)
    for (const order of ['min-fill', 'min-degree', [0, 1]] as const) {
      const r = variableElimination(graph, [2], { order })
      expect(r.marginal.data[1]).toBeCloseTo(brute.rain, 12)
      expect(r.logZ).toBeCloseTo(Math.log(brute.evidence), 12)
    }
  })

  it('evidence on a random tree: variable elimination equals enumeration of the reduced graph', () => {
    const g = randomTree(11)
    const ve = variableElimination(g, [0], { evidence: { 3: 1 } })
    const clamped = discreteFactorGraph(g.cardinalities, [...g.factors, discreteFactor([3], g.cardinalities, [0, 1])])
    const e = enumerate(clamped)
    close(ve.marginal.data, e.marginals[0].data, 1e-12)
    expect(ve.logZ).toBeCloseTo(e.logZ, 10)
  })

  it('satisfy the Algorithm protocol', () => {
    const g = isingGrid(2, 2, 0.4, 0.1)
    checkProtocol(enumerationSteps(g), undefined, { steps: 6, record: { logZ: (s) => s.logZ } })
    checkProtocol(enumerationSteps(g, { chunk: 3 }), undefined, { steps: 6 })
    checkProtocol(variableEliminationSteps(g), undefined, { steps: 6, record: { s: (st) => st.logScale } })
    expect(run(variableEliminationSteps(g), undefined, 100).done).toBe(true)
  })
})

describe('chains: the dishonest casino against enumeration', () => {
  const obs = [5, 5, 0, 5, 2, 5, 5]
  const brute = casinoBrute(obs)
  const chain = casinoChain(obs)
  const fb = forwardBackward(chain)

  it('forward–backward marginals, pairwise marginals and log-likelihood', () => {
    toRows(fb.marginals).forEach((r, n) => close(r, brute.marginals[n], 1e-12))
    const K = 2
    for (let n = 0; n < obs.length - 1; n++)
      close(Array.from(toFlat(fb.pairwise).slice(n * K * K, (n + 1) * K * K)), brute.pairwise[n].flat(), 1e-12)
    expect(fb.logLikelihood).toBeCloseTo(brute.logLikelihood, 12)
  })

  it('Viterbi finds the best path, in probability and log space', () => {
    const v = viterbi(chain)
    expect(toFlat(v.path)).toEqual(brute.path)
    expect(v.logProbability).toBeCloseTo(brute.logBest, 12)
    expect(toFlat(chainViterbi(log(chain.nodePotentials), log(chain.transition)).path)).toEqual(brute.path)
  })

  it('log-space chain forward–backward agrees with the scaled version', () => {
    const c = chainForwardBackward(log(chain.nodePotentials), log(chain.transition))
    close(toFlat(c.marginals), toFlat(fb.marginals), 1e-12)
    expect(c.logZ).toBeCloseTo(fb.logLikelihood, 12)
  })

  it('a chain-shaped factor graph is recognised and solved by chainSumProduct', () => {
    const cards = new Array<number>(obs.length).fill(2)
    const factors = [
      ...toRows(chain.nodePotentials).map((row, n) => discreteFactor([n], cards, row)),
      ...obs.slice(1).map((_, n) => discreteFactor([n, n + 1], cards, casino.transition.flat())),
    ]
    const g = discreteFactorGraph(cards, factors)
    expect(factorChain(g)?.order).toEqual(obs.map((_, n) => n))
    expect(factorChain(isingGrid(2, 2, 0.1, 0))).toBeNull()
    const s = run(chainSumProduct(g), undefined, 1000)
    expect(s.done).toBe(true)
    s.marginals.forEach((m, n) => close(toFlat(m), brute.marginals[n], 1e-12))
    expect(s.logZ).toBeCloseTo(brute.logLikelihood, 12)
    expect(toFlat(s.map!)).toEqual(brute.path)
    checkProtocol(chainSumProduct(g), undefined, { steps: 20 })
  })

  it('FFBS draws valid paths, reproducibly; its path frequencies follow the posterior', () => {
    const a = sampleHiddenPath(stream(5), chain)
    expect(toFlat(a)).toEqual(toFlat(sampleHiddenPath(stream(5), chain)))
    expect(a.shape).toEqual([obs.length])
    let loaded = 0
    const R = 4000
    for (let r = 0; r < R; r++) loaded += toFlat(sampleHiddenPath(stream(r), chain))[2]
    expect(loaded / R).toBeCloseTo(brute.marginals[2][1], 1)
  })

  it('stepping reveals the same result, and the steps satisfy the Algorithm protocol', () => {
    const s = run(forwardBackwardSteps(chain), undefined, 100)
    close(toFlat(s.marginals), toFlat(fb.marginals), 0)
    const v = run(viterbiSteps(chain), undefined, 100)
    expect(toFlat(v.path)).toEqual(brute.path)
    expect(trace(forwardBackwardSteps(chain), undefined, 100).meta.stopped).toBe('done')
    checkProtocol(forwardBackwardSteps(chain), undefined, { steps: 20, record: { ll: (st) => st.logLikelihood } })
    checkProtocol(viterbiSteps(chain), undefined, { steps: 20, record: { p: (st) => st.position } })
  })

  it('rejects potentials of the wrong shape', () => {
    expect(() =>
      forwardBackward({ nodePotentials: fromData(new Float64Array(6), [3, 2]), transition: chain.nodePotentials }),
    ).toThrow()
  })
})
