import { describe, expect, it } from 'vitest'
import { builtInEngines, infer, withEngines, type EngineRegistration } from 'aifn-compute/inference/engines'
import { dist, model } from 'aifn-compute/inference/model'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import type { Status } from 'aifn-compute/foundation/contracts'
import { betheLogZ, type BeliefPropagationState } from 'aifn-compute/inference/message-passing'
import type { EnumerationState } from 'aifn-compute/inference/exact'
import { casino, casinoBrute, sprinkler, sprinklerBrute } from '../graphs'
import { checkProtocol } from '../../protocol'

const casinoModel = model('casino', (m) => {
  const pi = m.constant('π', casino.initial)
  const A = m.constant('A', casino.transition)
  const B = m.constant('B', casino.emission)
  const time = m.chain('time', 'T')
  const z = time.variable('z', dist.Categorical(pi), { next: (previous) => dist.Categorical(A.at(previous)) })
  time.observed('x', dist.Categorical(B.at(z)))
})

/** A tree: a root with three children, each observed through a noisy copy. */
const star = model('star', (m) => {
  const r = m.variable('root', dist.Bernoulli(0.3))
  const p = m.constant('p', [0.2, 0.7])
  const q = m.constant('q', [0.1, 0.9])
  for (const k of [0, 1, 2]) {
    const c = m.variable(`c${k}`, dist.Bernoulli(p.at(r)))
    m.observed(`y${k}`, dist.Bernoulli(q.at(c)))
  }
})

/** A loop: a – b – c – a through pairwise observed agreements. */
const loop = model('loop', (m) => {
  const a = m.variable('a', dist.Bernoulli(0.5))
  const b = m.variable('b', dist.Bernoulli(0.5))
  const c = m.variable('c', dist.Bernoulli(0.5))
  const agree = m.constant('agree', [
    [0.8, 0.2],
    [0.2, 0.8],
  ])
  m.observed('ab', dist.Bernoulli(m.deterministic('pab', 'index', [agree, a, b])))
  m.observed('bc', dist.Bernoulli(m.deterministic('pbc', 'index', [agree, b, c])))
  m.observed('ca', dist.Bernoulli(m.deterministic('pca', 'index', [agree, c, a])))
})

const continuous = model('normal mean', (m) => {
  const mu = m.variable('μ', dist.Normal(0, 10))
  m.plate('points', 'n').observed('x', dist.Normal(mu, 1))
})

describe('infer chooses the engine by the shape of the model', () => {
  it('a chain: forward–backward, exact', () => {
    const obs = [5, 5, 0, 5, 2]
    const inf = infer(casinoModel, { sizes: { T: obs.length }, data: { x: obs } })
    expect(inf.engine).toBe('forward-backward')
    expect(inf.shape).toBe('chain')
    const s = run(inf.algorithm, undefined, 100) as Status & { marginals: Tensor[]; logZ: number }
    const brute = casinoBrute(obs)
    s.marginals.forEach((m, n) => expect(toFlat(m)[1]).toBeCloseTo(brute.marginals[n][1], 12))
    expect(s.logZ + inf.discrete!.logConstant).toBeCloseTo(brute.logLikelihood, 10)
  })

  it('a tree: belief propagation, exact', () => {
    const inf = infer(star, { data: { y0: 1, y1: 1, y2: 0 } })
    expect(inf.engine).toBe('belief-propagation')
    expect(inf.shape).toBe('tree')
    const byName = infer(star, { data: { y0: 1, y1: 1, y2: 0 } }, { engine: 'enumeration' })
    expect(byName.engine).toBe('enumeration')
    const bp = run(inf.algorithm, undefined, 1000) as BeliefPropagationState
    const en = run(byName.algorithm, undefined, 1000) as EnumerationState
    expect(bp.converged).toBe(true)
    expect(betheLogZ(bp)).toBeCloseTo(en.logZ, 10)
    bp.beliefs.forEach((b, v) => {
      const m = toFlat(en.logMarginals[v]).map((l) => Math.exp(l - en.logZ))
      toFlat(b).forEach((p, k) => expect(p).toBeCloseTo(m[k], 10))
    })
  })

  it('a loop: loopy belief propagation; the sprinkler (a loop through its v-structure) too', () => {
    expect(infer(loop, { data: { ab: 1, bc: 1, ca: 0 } }).engine).toBe('belief-propagation')
    const inf = infer(sprinkler, { data: { wet: 1 } })
    expect(inf.engine).toBe('belief-propagation')
    expect(['dag', 'general', 'tree', 'lattice']).toContain(inf.shape)
  })

  it('a linear-Gaussian model: expectation propagation, exact here; Gibbs by name', () => {
    const ep = infer(continuous, { data: { x: [1.2, 0.8, 1.1] } })
    expect(ep.engine).toBe('expectation-propagation')
    const q = run(ep.algorithm, undefined, 20) as Status & { means: Tensor; variances: Tensor }
    const precision = 1 / 100 + 3
    expect(q.means.data[0]).toBeCloseTo(3.1 / precision, 12)
    expect(q.variances.data[0]).toBeCloseTo(1 / precision, 12)
    const inf = infer(continuous, { data: { x: [1.2, 0.8, 1.1] } }, { engine: 'gibbs' })
    expect(inf.engine).toBe('gibbs')
    expect(inf.discrete).toBeNull()
    const s = run(inf.algorithm, undefined, 50, { stream: stream(1) }) as Status & { values: Record<string, number> }
    expect(Math.abs(s.values['μ'] - 1)).toBeLessThan(2)
  })

  it('an engine can be named, and extra engines are tried first', () => {
    const bindings = { data: { wet: 1 } }
    const ve = infer(sprinkler, bindings, { engine: 'variable-elimination' })
    expect(ve.engine).toBe('variable-elimination')
    const brute = sprinklerBrute()
    const s = run(ve.algorithm, undefined, 1000) as Status & { logZ?: number; logScale?: number }
    expect(s).toBeDefined()
    expect(brute.evidence).toBeGreaterThan(0)
    const mine: EngineRegistration = {
      name: 'mine',
      matches: (c) => c.model.name === 'Sprinkler',
      create: () => ({ name: 'nothing', init: () => ({ t: 0 }), step: (st) => ({ t: st.t + 1 }) }),
    }
    const table = withEngines(builtInEngines, mine)
    expect(infer(sprinkler, bindings, { engines: table }).engine).toBe('mine')
    expect(infer(casinoModel, { sizes: { T: 3 }, data: { x: [0, 1, 2] } }, { engines: table }).engine).toBe(
      'forward-backward',
    )
    expect(() => infer(sprinkler, bindings, { engine: 'no such engine' })).toThrow()
  })

  it('every engine the table picks satisfies the Algorithm protocol', () => {
    checkProtocol(infer(casinoModel, { sizes: { T: 4 }, data: { x: [5, 0, 5, 5] } }).algorithm, undefined, {
      steps: 10,
    })
    checkProtocol(infer(sprinkler, { data: { wet: 1 } }).algorithm, undefined, { steps: 10 })
    checkProtocol(infer(continuous, { data: { x: [1, 2] } }).algorithm, undefined, { steps: 6 })
    checkProtocol(infer(continuous, { data: { x: [1, 2] } }, { engine: 'gibbs' }).algorithm, undefined, {
      steps: 6,
      random: true,
    })
  })
})
