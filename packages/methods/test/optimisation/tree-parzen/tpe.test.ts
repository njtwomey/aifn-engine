import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { bool, int, oneOf, real, space, when } from 'aifn-compute/foundation/space'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { treeParzenAlgorithms, treeParzenSteps, type TreeParzenState } from 'aifn-methods/optimisation/tree-parzen'
import { fixture } from '../../fixtures'
import { expectProtocol } from '../../protocol'
import { expectInfo } from '../../registry'

type Case = {
  x: number[] | number[][]
  losses: number[]
  gamma: number
  rule: 'scott' | 'hyperopt'
  good: number[]
  l: number[]
  g: number[]
}
const F = fixture<Record<'scott' | 'hyperopt' | 'log' | 'joint', Case>>('optimisation/tree-parzen')
const flat = (x: Parameters<typeof toFlat>[0]) => Array.from(toFlat(x) as ArrayLike<number>)
const rel = (a: number[], b: number[], tol: number) =>
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(tol * Math.abs(b[i]) + 1e-14))

describe('treeParzenSteps: densities against scipy truncnorm mixtures', () => {
  for (const name of ['scott', 'hyperopt', 'log'] as const) {
    it(`one dimension, ${name}`, () => {
      const c = F[name]
      const xs = c.x as number[]
      const s = name === 'log' ? space({ x: real(1e-3, 1, { scale: 'log' }) }) : space({ x: real(0, 1) })
      const loss = new Map(xs.map((x, i) => [x, c.losses[i]]))
      const alg = treeParzenSteps(({ x }) => loss.get(x) as number, s, {
        gamma: c.gamma,
        bandwidth: c.rule,
        init: xs.length,
        grid: 11,
      })
      const st = run(alg, { points: xs.map((x) => ({ x })) }, 0, { stream: stream(0) })
      expect([...st.good].sort((a, b) => a - b)).toEqual([...c.good].sort((a, b) => a - b))
      rel(flat(st.marginals[0].good), c.l, 1e-10)
      rel(flat(st.marginals[0].bad), c.g, 1e-10)
      rel(
        flat(st.marginals[0].ratio),
        c.l.map((v, i) => v / c.g[i]),
        1e-10,
      )
      expect(st.threshold).toBe(c.losses[c.good[c.good.length - 1]])
    })
  }

  it('two dimensions: the joint product-kernel densities', () => {
    const c = F.joint
    const xs = c.x as number[][]
    const loss = new Map(xs.map((x, i) => [`${x[0]},${x[1]}`, c.losses[i]]))
    const alg = treeParzenSteps(
      ({ a, b }) => loss.get(`${a},${b}`) as number,
      space({ a: real(0, 1), b: real(0, 1) }),
      {
        gamma: c.gamma,
        init: xs.length,
        jointGrid: 5,
      },
    )
    const st = run(alg, { points: xs.map(([a, b]) => ({ a, b })) }, 0, { stream: stream(0) })
    expect(st.joint).not.toBeNull()
    rel(flat(st.joint!.good), c.l, 1e-10)
    rel(flat(st.joint!.bad), c.g, 1e-10)
  })
})

describe('treeParzenSteps: behaviour', () => {
  it('each state proposes the candidate with the largest l/g, and the next step evaluates it', () => {
    const tr = trace(
      treeParzenSteps(({ x }) => (x - 0.3) ** 2, space({ x: real(0, 1) }), { init: 5 }),
      {},
      6,
      { stream: stream(2), keep: 'all' },
    )
    const steps = tr.steps as TreeParzenState[]
    for (let k = 0; k < steps.length; k++) {
      const s = steps[k]
      const scores = flat(s.scores)
      const best = scores.indexOf(Math.max(...scores))
      expect(flat(s.candidates)[best]).toBeCloseTo(s.next.x as number, 12)
      expect(s.observations.length).toBe(5 + k)
      if (k > 0) expect(s.observations[s.observations.length - 1]).toEqual(steps[k - 1].next)
      // l(x)/g(x) at the chosen point beats every other candidate's.
      expect(scores.every((v) => v <= scores[best])).toBe(true)
    }
  })

  it('finds the minimum of a one-dimensional function, better than random search on average', () => {
    const f = ({ x }: { x: number }) => Math.sin(5 * x) * (1 - Math.tanh(x * x)) + 0.1 * x
    // A dense grid gives the true minimum.
    let fmin = Infinity
    for (let k = 0; k <= 20000; k++) fmin = Math.min(fmin, f({ x: -2 + (4 * k) / 20000 }))
    const s = space({ x: real(-2, 2) })
    let gapTpe = 0
    let gapRandom = 0
    for (let seed = 0; seed < 5; seed++) {
      const tpe = run(treeParzenSteps(f, s, { init: 10 }), {}, 30, { stream: stream(seed) })
      const rnd = run(treeParzenSteps(f, s, { init: 40 }), {}, 0, { stream: stream(seed) })
      expect(tpe.observations.length).toBe(rnd.observations.length)
      gapTpe += tpe.best - fmin
      gapRandom += rnd.best - fmin
    }
    expect(gapTpe / 5).toBeLessThan(2e-3)
    expect(gapTpe).toBeLessThan(gapRandom)
  })

  it('log, integer, choice and boolean dimensions', () => {
    const s = space({
      lr: real(1e-4, 1, { scale: 'log' }),
      layers: int(1, 6),
      act: oneOf(['relu', 'tanh', 'gelu']),
      bn: bool(),
    })
    const f = ({ lr, layers, act, bn }: { lr: number; layers: number; act: string; bn: boolean }) =>
      (Math.log10(lr) + 2) ** 2 + 0.3 * Math.abs(layers - 3) + (act === 'gelu' ? 0 : 0.5) + (bn ? 0 : 0.4)
    const st = run(treeParzenSteps(f, s, { init: 10 }), {}, 40, { stream: stream(3) })
    expect(st.best).toBeLessThan(0.5)
    expect(Number.isInteger(st.next.layers)).toBe(true)
    expect(['relu', 'tanh', 'gelu']).toContain(st.next.act)
    const act = st.marginals[2]
    expect(act.kind).toBe('categorical')
    expect(act.values).toEqual(['relu', 'tanh', 'gelu'])
    // Each categorical marginal is a distribution over its options.
    expect(flat(act.good).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(flat(st.marginals[3].bad).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(st.marginals[0].values[0]).toBeCloseTo(1e-4, 12)
    expect(st.joint).toBeNull()
  })

  it('the continuous marginals integrate to one; univariate mode works too', () => {
    for (const multivariate of [true, false]) {
      const st = run(
        treeParzenSteps(({ a, b }) => a * a + b, space({ a: real(0, 1), b: real(0, 1) }), { multivariate, grid: 401 }),
        {},
        3,
        { stream: stream(4) },
      )
      for (const m of st.marginals) {
        const v = flat(m.good)
        // Trapezoid rule over [0, 1].
        const area = v.reduce((a, x, k) => a + (k === 0 || k === v.length - 1 ? x / 2 : x), 0) / (v.length - 1)
        expect(area).toBeCloseTo(1, 4)
      }
    }
  })

  it('rejects what it cannot model', () => {
    const f = () => 0
    expect(() => treeParzenSteps(f, space({ a: oneOf(['x', 'y']), b: real(0, 1, { when: when('a', 'x') }) }))).toThrow(
      /conditional/,
    )
    expect(() => treeParzenSteps(f, space({ x: real(0, 1) }), { gamma: 0 })).toThrow(/gamma/)
    expect(() => treeParzenSteps(f, space({ x: real(0, 1) }), { candidates: 0 })).toThrow(/candidates/)
  })

  it('follows the trace protocol and registers', () => {
    const s = space({ x: real(0, 1, { scale: 'linear' }), k: oneOf(['a', 'b']) })
    expectProtocol(
      treeParzenSteps(({ x, k }) => x + (k === 'a' ? 0 : 1), s, { init: 3, grid: 5 }),
      {},
      { n: 5 },
    )
    expectInfo(treeParzenAlgorithms, 'algorithm')
  })
})
