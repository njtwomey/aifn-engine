import { describe, expect, it } from 'vitest'
import {
  discreteFactor,
  dist,
  expandModel,
  factorMarginalise,
  factorProduct,
  factorsOf,
  logJoint,
  model,
  modelMarkovBlanket,
  sampleModel,
  toDiscreteFactorGraph,
  toFactorDiagram,
  toFactorGraph,
  toPlateDiagram,
} from 'aifn-compute/inference/model'
import { shape } from 'aifn-compute/graph/structured'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { casino, sprinkler } from '../graphs'

describe('factor algebra', () => {
  it('multiplies and marginalises tables', () => {
    const cards = [2, 3]
    const a = discreteFactor([0], cards, [1, 2])
    const b = discreteFactor([0, 1], cards, [1, 2, 3, 4, 5, 6])
    const ab = factorProduct(a, b, cards)
    expect(ab.scope).toEqual([0, 1])
    expect(toFlat(ab.table)).toEqual([1, 2, 3, 8, 10, 12])
    expect(toFlat(factorMarginalise(ab, [0]).table)).toEqual([9, 12, 15])
    expect(toFlat(factorMarginalise(ab, [1], 'max').table)).toEqual([3, 12])
  })
})

/** A plated model: topics φ_k, documents θ_d and their words, as LDA is written (Blei, Ng & Jordan, 2003). */
const topicModel = model('topics', (m) => {
  const K = m.size('K')
  const V = m.size('V')
  const topics = m.plate('topics', K, { label: 'K' })
  const docs = m.plate('documents', 'D', { label: 'D' })
  const words = docs.plate('words', 'N', { label: 'N_d' })
  const alpha = m.constant('α')
  const beta = m.constant('β')
  const phi = topics.variable('φ', dist.Dirichlet(beta, V))
  const theta = docs.variable('θ', dist.Dirichlet(alpha, K))
  const z = words.variable('z', dist.Categorical(theta))
  words.observed('w', dist.Categorical(phi.at(z)))
})
const topicBindings = {
  sizes: { K: 2, V: 4 },
  constants: { α: 0.5, β: 0.1 },
  data: {
    w: [
      [0, 1, 0, 1, 0],
      [2, 3, 3, 2],
    ],
  },
}

describe('the model language', () => {
  it('Markov blanket of sprinkler: parents, children and co-parents', () => {
    const mb = modelMarkovBlanket(sprinkler, 'sprinkler', { data: { wet: 1 } })
    expect(mb.parents).toEqual(['cloudy'])
    expect(mb.children).toEqual(['wet'])
    expect(mb.coParents).toEqual(['rain'])
  })

  it('a plated description expands, samples, and has the expected Markov blankets', () => {
    const em = expandModel(topicModel, topicBindings)
    expect(em.byNode.get('z')!.length).toBe(9)
    const mb = modelMarkovBlanket(topicModel, 'z[0,0]', topicBindings)
    expect(mb.parents).toEqual(['θ[0]'])
    expect(mb.children).toEqual(['w[0,0]'])
    expect(mb.coParents.sort()).toEqual(['φ[0]', 'φ[1]'])
    const draw = sampleModel(stream(1), topicModel, topicBindings)
    expect(Number.isFinite(logJoint(em, draw))).toBe(true)
  })

  it('the joint density is the sum of the conditionals', () => {
    const em = expandModel(sprinkler, { data: { wet: 1 } })
    const values = { cloudy: 1, sprinkler: 0, rain: 1 }
    const pW = 0.9
    const expected = Math.log(0.5) + Math.log(1 - 0.1) + Math.log(0.8) + Math.log(pW)
    expect(logJoint(em, values)).toBeCloseTo(expected, 12)
  })

  it('a chain declared in the language is chain-shaped once tabulated', () => {
    const hmm = model('casino', (m) => {
      const pi = m.constant('π', casino.initial)
      const A = m.constant('A', casino.transition)
      const B = m.constant('B', casino.emission)
      const time = m.chain('time', 'T')
      const z = time.variable('z', dist.Categorical(pi), { next: (previous) => dist.Categorical(A.at(previous)) })
      time.observed('x', dist.Categorical(B.at(z)))
    })
    const d = toDiscreteFactorGraph(hmm, { sizes: { T: 4 }, data: { x: [5, 5, 0, 5] } })
    expect(d.keys).toEqual(['z[0]', 'z[1]', 'z[2]', 'z[3]'])
    expect(shape(toFactorGraph(hmm, { sizes: { T: 4 }, data: { x: [5, 5, 0, 5] } }) as never)).toBeDefined()
  })

  it('diagrams: plates around their nodes, a factor graph with a highlighted blanket', () => {
    const plate = toPlateDiagram(topicModel)
    expect(plate.nodes.map((n) => n.id)).toEqual(['α', 'β', 'φ', 'θ', 'z', 'w'])
    expect(plate.nodes.find((n) => n.id === 'w')!.filled).toBe(true)
    const around = (name: string) => plate.groups.find((g) => g.id === `group ${name}`)!.around
    expect(around('words')).toEqual(['z', 'w'])
    expect([...around('documents')].sort()).toEqual(['w', 'z', 'θ'].sort())
    const small = { ...topicBindings, data: { w: [[0, 1]] } }
    const fg = toFactorGraph(topicModel, small)
    expect(factorsOf(fg).length).toBe(2 + 1 + 2 + 2)
    const diagram = toFactorDiagram(topicModel, { bindings: small, highlight: 'z[0,0]' })
    const states = diagram.nodes.filter((n) => n.state !== undefined).map((n) => n.state)
    expect(states).toContain('active')
  })
})

describe('factor-graph labels', () => {
  const labelled = model('labelled', (m) => {
    const docs = m.plate('documents', 'D', { label: 'D' })
    const alpha = m.constant('α')
    docs.variable('θ', dist.Dirichlet(alpha, 2), { label: '\\theta_d' })
    docs.variable('μ', dist.Dirichlet(alpha, 2), { label: '$\\mu$' })
  })
  it('an instance subscript groups a label that already has one (no double subscript)', () => {
    const fg = toFactorGraph(labelled, { sizes: { D: 2 }, constants: { α: 1 } })
    const labels = fg.attributes.slice(0, fg.variables).map((n) => n.label)
    expect(labels).toContain('{\\theta_d}_{0}')
    expect(labels).toContain('$\\mu_{1}$')
    expect(labels.some((l) => /_[^{]*_\{/.test(l ?? '') && !l!.startsWith('{'))).toBe(false)
  })
})
