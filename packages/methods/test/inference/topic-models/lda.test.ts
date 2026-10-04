import { describe, expect, it } from 'vitest'
import { ldaEngines, ldaModel, type LdaState } from 'aifn-methods/inference/topic-models'
import { infer } from 'aifn-compute/inference/engines'
import {
  expandModel,
  factorsOf,
  logJoint,
  modelMarkovBlanket,
  sampleModel,
  toFactorDiagram,
  toFactorGraph,
  toPlateDiagram,
} from 'aifn-compute/inference/model'
import { modelGibbs } from 'aifn-compute/inference/stochastic'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace, type Algorithm } from 'aifn-compute/foundation/trace'

describe('LDA', () => {
  const lda = ldaModel()
  const bindings = {
    sizes: { K: 2, V: 4 },
    constants: { α: 0.5, β: 0.1 },
    data: {
      w: [
        [0, 1, 0, 1, 0],
        [2, 3, 3, 2],
        [0, 0, 1, 1, 1],
        [3, 2, 2, 3, 3],
      ],
    },
  }
  it('the description expands, samples and has the expected Markov blanket', () => {
    const em = expandModel(lda, bindings)
    expect(em.byNode.get('z')!.length).toBe(19)
    const mb = modelMarkovBlanket(lda, 'z[0,0]', bindings)
    expect(mb.parents).toEqual(['θ[0]'])
    expect(mb.children).toEqual(['w[0,0]'])
    expect(mb.coParents.sort()).toEqual(['φ[0]', 'φ[1]'])
    const draw = sampleModel(stream(1), lda, bindings)
    expect(Number.isFinite(logJoint(em, draw))).toBe(true)
  })
  it('infer picks the registered collapsed Gibbs engine; the collapsed likelihood improves', () => {
    expect(infer(lda, bindings).engine).toBe('gibbs')
    const inf = infer(lda, bindings, { engines: ldaEngines })
    expect(inf.engine).toBe('lda-collapsed-gibbs')
    const t = trace(inf.algorithm as Algorithm<void, LdaState>, undefined, 50, {
      stream: stream(8),
      record: { ll: (s) => s.logLikelihood },
    })
    const ll = toFlat(t.series.ll)
    expect(ll[ll.length - 1]).toBeGreaterThan(ll[0])
  })
  it('uncollapsed Gibbs runs from conjugate Dirichlet conditionals', () => {
    const s = run(modelGibbs(lda, bindings), undefined, 5, { stream: stream(3) })
    expect(s.kinds['φ[0]']).toBe('dirichlet')
    expect(s.kinds['θ[1]']).toBe('dirichlet')
  })
  it('diagrams: plates around their nodes, a factor graph with a highlighted blanket', () => {
    const plate = toPlateDiagram(lda)
    expect(plate.nodes.map((n) => n.id)).toEqual(['α', 'β', 'φ', 'θ', 'z', 'w'])
    expect(plate.nodes.find((n) => n.id === 'w')!.filled).toBe(true)
    expect(plate.groups.find((g) => g.id === 'group words')!.around).toEqual(['z', 'w'])
    expect([...plate.groups.find((g) => g.id === 'group documents')!.around].sort()).toEqual(['w', 'z', 'θ'].sort())
    const small = { ...bindings, data: { w: [[0, 1]] } }
    const fg = toFactorDiagram(lda, { bindings: small, highlight: 'z[0,0]' })
    const state = (id: string) => fg.nodes.find((n) => n.id === id)?.state
    expect(state('z[0,0]')).toBe('active')
    expect(state('φ[1]')).toBe('done')
    expect(state('z[0,1]')).toBe('idle')
    expect(factorsOf(toFactorGraph(lda, small)).length).toBe(2 + 1 + 2 + 2)
  })
})
