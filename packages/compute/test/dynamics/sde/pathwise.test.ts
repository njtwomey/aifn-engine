/**
 * Pathwise (reparameterised) gradients of an Euler–Maruyama expectation: the increments come from a fixed stream, so
 * the Monte Carlo estimate is a smooth function of the drift parameters and `unrolled` differentiates it. Checked
 * against central differences with common random numbers (the same stream at both ends), and against the exact
 * gradient of the scheme's expectation, which the recursion for its mean and variance gives.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import {
  add,
  get,
  mean,
  mul,
  square,
  sub,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { run, unrolled } from 'aifn-compute/foundation/trace'
import { eulerMaruyama, type SdeInitial } from 'aifn-compute/dynamics/sde'

// dX = θ₀(θ₁ − X) dt + σ dW (Ornstein–Uhlenbeck), x₀ = 1; the loss is the estimate of E[X_T²].
const sigma = 0.6
const h = 0.02
const steps = 50
const paths = 4000
const start: SdeInitial = { x0: 1, paths }
const scheme = (theta: Value) =>
  eulerMaruyama(
    { drift: (_t, x) => mul(get(theta, 0), sub(get(theta, 1), x)) as Tensor, diffusion: () => sigma },
    { stepSize: h },
  )
const theta0 = tensor([1.5, 0.4])
const flat = (v: Value): number[] => Array.from(toFlat(v as Tensor))

const estimate = (theta: Value, seed: number) =>
  mean(square(unrolled(scheme, start, steps, { params: theta, stream: stream(seed) }).x))
const estimateRaw = (theta: number[], seed: number) =>
  unwrap(mean(square(run(scheme(tensor(theta)), start, steps, { stream: stream(seed) }).x))) as number

describe('Euler–Maruyama pathwise gradients', () => {
  it('match central differences with common random numbers', () => {
    const g = flat(grad((th: Value) => estimate(th, 7))(theta0) as Value)
    const eps = 1e-6
    flat(theta0).forEach((_, i) => {
      const up = flat(theta0)
      const down = flat(theta0)
      up[i] += eps
      down[i] -= eps
      const fd = (estimateRaw(up, 7) - estimateRaw(down, 7)) / (2 * eps)
      expect(g[i]).toBeCloseTo(fd, 6)
    })
  })

  it('are unbiased for the gradient of the scheme’s expectation', () => {
    // X_{n+1} = r X_n + θ₀θ₁h + σ√h Z with r = 1 − θ₀h: m ← r m + θ₀θ₁h, v ← r² v + σ²h; E[X_N²] = m² + v.
    const exact = (th: Value) => {
      const r = sub(1, mul(get(th, 0), h))
      const c = mul(mul(get(th, 0), get(th, 1)), h)
      let m: Value = 1
      let v: Value = 0
      for (let n = 0; n < steps; n++) {
        m = add(mul(r, m), c)
        v = add(mul(square(r), v), sigma * sigma * h)
      }
      return add(square(m), v)
    }
    const want = flat(grad(exact)(theta0) as Value)
    // Average the pathwise gradient over independent streams; each is an unbiased estimate.
    const seeds = [1, 2, 3, 4]
    const got = seeds
      .map((s) => flat(grad((th: Value) => estimate(th, s))(theta0) as Value))
      .reduce((a, b) => a.map((x, i) => x + b[i] / seeds.length), [0, 0])
    got.forEach((gi, i) => expect(Math.abs(gi - want[i])).toBeLessThan(0.03 * Math.max(1, Math.abs(want[i]))))
  })
})
