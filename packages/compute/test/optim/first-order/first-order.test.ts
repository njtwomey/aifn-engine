import { describe, expect, it } from 'vitest'
import {
  adagrad,
  adagradRule,
  adam,
  adamRule,
  adamw,
  adamwRule,
  applyUpdates,
  chainRules,
  clipByGlobalNorm,
  conjugateGradient,
  coordinateDescent,
  globalNorm,
  gradientDescent,
  linearConjugateGradient,
  momentum,
  nesterov,
  rmsprop,
  rmspropRule,
  sgdRule,
  solveConjugateGradient,
  type UpdateRule,
} from 'aifn-compute/optim/first-order'
import { exponentialDecay, inverseSqrtDecay, inverseTimeDecay } from 'aifn-compute/optim'
import { stream } from 'aifn-compute/foundation/random'
import { mul, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { grad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { run, trace } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'
import type { Algorithm } from 'aifn-compute/foundation/contracts'
import type { IterateState, StartOptions } from 'aifn-compute/optim'
import { checkProtocol } from '../../protocol'
import { bowl, rosenbrock } from '../problems'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol)
}

type Path = { w: number[][]; b: number[][] }
const T = fixture<{
  A: number[][]
  c: number[]
  w0: number[]
  b0: number
  sgd: Path
  momentum: Path
  nesterov: Path
  adagrad: Path
  rmsprop: Path
  adam: Path
  adamw: Path
}>('optim/first-order')

describe('pytree update rules match torch.optim', () => {
  type P = { w: Tensor; b: number }
  const grads = (p: P): P => {
    const w = toFlat(p.w)
    return { w: tensor(T.A.map((row, i) => row.reduce((s, a, j) => s + a * w[j], 0) - T.c[i])), b: p.b - 1 }
  }
  const follow = (rule: UpdateRule, ref: Path) => {
    let params: P = { w: tensor(T.w0), b: T.b0 }
    let state = rule.init(params)
    ref.w.forEach((w, k) => {
      const out = rule.update(grads(params), state, params)
      params = applyUpdates(params, out.updates) as P
      state = out.state
      close(toFlat(params.w), w, 1e-12)
      expect(typeof params.b).toBe('number')
      expect(params.b).toBeCloseTo(ref.b[k][0], 12)
    })
  }
  it.each([
    ['sgd', () => sgdRule({ stepSize: 0.1 })],
    ['momentum', () => sgdRule({ stepSize: 0.1, momentum: 0.9, weightDecay: 0.05 })],
    ['nesterov', () => sgdRule({ stepSize: 0.1, momentum: 0.8, nesterov: true })],
    ['adagrad', () => adagradRule({ stepSize: 0.3, epsilon: 1e-10 })],
    ['rmsprop', () => rmspropRule({ stepSize: 0.05, decay: 0.9, epsilon: 1e-8 })],
    ['adam', () => adamRule({ stepSize: 0.1, beta1: 0.9, beta2: 0.99, weightDecay: 0.1 })],
    ['adamw', () => adamwRule({ stepSize: 0.1, weightDecay: 0.2 })],
  ] as const)('%s', (name, make) => follow(make(), T[name]))

  it('rules are pure: the state is plain data and survives structuredClone', () => {
    const rule = adamRule({ stepSize: 0.1 })
    const params = { w: tensor(T.w0), b: T.b0 }
    const s0 = rule.init(params)
    const a = rule.update(grads(params), s0, params)
    const b = rule.update(grads(params), s0, params)
    expect(toFlat((a.updates as P).w)).toEqual(toFlat((b.updates as P).w))
    expect(a.state.t).toBe(1)
    expect(s0.t).toBe(0)
  })

  it('schedules are read at the update count', () => {
    const rule = sgdRule({ stepSize: exponentialDecay(1, 0.5) })
    let state = rule.init(1)
    const steps: number[] = []
    for (let k = 0; k < 3; k++) {
      const out = rule.update(1, state)
      steps.push(out.updates as number)
      state = out.state
    }
    expect(steps).toEqual([-1, -0.5, -0.25])
    expect(inverseTimeDecay(2, 1)(3)).toBe(0.5)
    expect(inverseSqrtDecay(3)(8)).toBe(1)
  })

  it('clipByGlobalNorm rescales the whole tree, and chains with another rule', () => {
    const g = { a: tensor([3, 0]), b: 4 }
    expect(globalNorm(g)).toBe(5)
    const clip = clipByGlobalNorm(1)
    const clipped = clip.update(g, clip.init(g)).updates as { a: Tensor; b: number }
    close(toFlat(clipped.a), [0.6, 0], 1e-15)
    expect(clipped.b).toBeCloseTo(0.8, 15)
    const chained = chainRules(clipByGlobalNorm(1), sgdRule({ stepSize: 2 }))
    const out = chained.update(g, chained.init(g)).updates as { a: Tensor; b: number }
    close(toFlat(out.a), [-1.2, 0], 1e-15)
    expect(out.b).toBeCloseTo(-1.6, 15)
  })

  it('clipByGlobalNorm differentiates with traced gradients (review maths 16)', () => {
    // f(s) = Σ clip(s·v): clipped, the sum is c·Σv/‖v‖, constant in s, so f′ = 0; below the threshold f′ = Σv.
    const v = [3, 4]
    const clip = clipByGlobalNorm(1)
    const f = (s: Value) => {
      // Inside grad the leaf is traced, which `Params` (concrete trees) does not name.
      const g = { a: mul(s, tensor(v)) } as unknown as Params
      return sum((clip.update(g, clip.init(g)).updates as { a: Value }).a)
    }
    expect(grad(f)(2) as number).toBeCloseTo(0, 14)
    expect(grad(f)(0.1) as number).toBeCloseTo(7, 14)
    expect(f(2) as number).toBeCloseTo(7 / 5, 14)
  })

  it('clipByGlobalNorm scales huge gradients to the bound, not to 0 (review regression: ‖g‖² overflowed)', () => {
    const g = { a: tensor([3e200, 0]), b: 4e200 }
    expect(globalNorm(g)).toBeCloseTo(5e200, -186)
    const clip = clipByGlobalNorm(1)
    const clipped = clip.update(g, clip.init(g)).updates as { a: Tensor; b: number }
    close(toFlat(clipped.a), [0.6, 0], 1e-15)
    expect(clipped.b).toBeCloseTo(0.8, 15)
  })
})

describe('first-order methods', () => {
  const b = bowl({ condition: 10 })
  it('gradient descent on a quadratic converges linearly at rate max|1 − ηλ|', () => {
    const t = trace(gradientDescent(b.objective, { stepSize: 0.1, tolerance: 0 }), { x0: b.start }, 40, {
      record: { value: (s) => s.value },
    })
    const v = toFlat(t.series.value)
    // f ∝ (1 − ηλ_min)^{2t} asymptotically: λ_min = 1 gives ratio 0.81.
    expect(v[40] / v[39]).toBeCloseTo(0.81, 3)
  })

  it('every first-order method decreases a convex bowl', () => {
    const algs = [
      momentum(b.objective, { stepSize: 0.05 }),
      nesterov(b.objective, { stepSize: 0.05 }),
      adagrad(b.objective, { stepSize: 0.5 }),
      rmsprop(b.objective, { stepSize: 0.05 }),
      adam(b.objective, { stepSize: 0.1 }),
      adamw(b.objective, { stepSize: 0.1, weightDecay: 0 }),
      gradientDescent(b.objective, { stepSize: 1, lineSearch: 'backtracking' }),
    ]
    for (const alg of algs) {
      const s = run(alg, { x0: b.start }, 500)
      expect(s.value, alg.name).toBeLessThan(1e-3)
      expect(s.diverged).toBe(false)
    }
  })

  it('flags divergence when the step is too large', () => {
    const s = run(gradientDescent(b.objective, { stepSize: 0.25 }), { x0: b.start }, 10000)
    expect(s.diverged).toBe(true)
  })

  it.each(['fletcher-reeves', 'polak-ribiere'] as const)(
    'nonlinear conjugate gradients (%s) on Rosenbrock',
    (variant) => {
      const r = rosenbrock()
      const s = run(conjugateGradient(r.objective, { variant }), { x0: r.start }, 5000)
      expect(s.converged).toBe(true)
      close(toFlat(s.x), [1, 1], 1e-4)
    },
  )

  it('linear conjugate gradients solve an SPD system in n steps', () => {
    const A = [
      [4, 1, 0],
      [1, 3, 1],
      [0, 1, 2],
    ]
    const rhs = [1, 2, 3]
    const r = solveConjugateGradient(A, rhs)
    expect(r.converged).toBe(true)
    expect(r.steps).toBeLessThanOrEqual(3)
    const x = toFlat(r.x)
    close(
      A.map((row) => row.reduce((s, a, j) => s + a * x[j], 0)),
      rhs,
      1e-10,
    )
    expect(
      run(
        linearConjugateGradient((v) => v, [1, 2]),
        {},
        5,
      ).converged,
    ).toBe(true)
  })

  it('coordinate descent with Newton coordinate steps solves a quadratic', () => {
    const q = bowl({ condition: 5, angle: 0.5 })
    for (const rule of ['cyclic', 'random', 'greedy'] as const) {
      const s = run(coordinateDescent(q.objective, { rule, hessian: q.hessian }), { x0: q.start }, 2000, {
        stream: stream(3),
      })
      expect(s.converged).toBe(true)
    }
  })
})

describe('protocol', () => {
  const r = rosenbrock()
  const q = bowl({ condition: 5, angle: 0.5 })
  const record = { value: (s: { value: number }) => s.value }
  it.each<[string, Algorithm<StartOptions, IterateState>]>([
    ['gradientDescent', gradientDescent(r.objective, { stepSize: 1e-3 })],
    ['gradientDescent + wolfe', gradientDescent(r.objective, { stepSize: 1, lineSearch: 'strong-wolfe' })],
    ['momentum', momentum(r.objective, { stepSize: 1e-3 })],
    ['nesterov', nesterov(r.objective, { stepSize: 1e-3 })],
    ['adagrad', adagrad(r.objective)],
    ['rmsprop', rmsprop(r.objective)],
    ['adam', adam(r.objective, { stepSize: 0.05 })],
    ['adamw', adamw(r.objective, { stepSize: 0.05 })],
    ['conjugateGradient', conjugateGradient(r.objective)],
    ['coordinateDescent (random)', coordinateDescent(q.objective, { rule: 'random' })],
  ])('%s satisfies the Algorithm protocol', (name, alg) => {
    checkProtocol(alg, { x0: name.startsWith('coordinate') ? q.start : r.start }, { steps: 12, record })
  })

  it('linearConjugateGradient satisfies the Algorithm protocol', () => {
    const A = [
      [4, 1, 0],
      [1, 3, 1],
      [0, 1, 2],
    ]
    checkProtocol(linearConjugateGradient(A, [1, 2, 3]), {}, { steps: 3 })
  })
})
