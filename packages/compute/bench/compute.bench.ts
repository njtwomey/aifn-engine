/**
 * Compute micro-benchmarks (design K §0 and §10.5): the baseline table, re-measured against today's rules. Run with
 * `make bench` (not part of `make check`; reported, not gated). Each group pairs aifn operations with their plain-JS
 * reference where the table has one, and prints the mean time per call in µs.
 */
import { test, type TestContext } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { cholesky } from 'aifn-compute/numerics/linalg'
import { filterAll, kalmanStep, parseModel } from 'aifn-compute/inference/filtering'
import { normal, stream } from 'aifn-compute/foundation/random'
import { run } from 'aifn-compute/foundation/trace'
import { adam, gradientDescent } from 'aifn-compute/optim/first-order'
import { rungeKutta } from 'aifn-compute/dynamics/ode'
import {
  add,
  exp,
  eye,
  fromData,
  greater,
  matmul,
  mul,
  sin,
  square,
  sum,
  linearCombination,
  toFlat,
  where,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'

type Fn = () => void

/** Benchmark the entries together and print each one's mean time per call. */
async function group(bench: TestContext['bench'], title: string, entries: readonly [string, Fn][]): Promise<void> {
  const result = await bench.compare(...entries.map(([name, fn]) => bench(name, fn)), { time: 300 })
  const lines = entries.map(([name]) => `  ${name.padEnd(40)} ${(result.get(name).latency.mean * 1000).toFixed(3)} µs`)
  process.stderr.write(`${title}\n${lines.join('\n')}\n`)
}

const s = stream('bench')
const vector = (n: number, seed = 1) => fromData(Float64Array.from({ length: n }, (_, k) => Math.sin(k * seed + 0.5)))
const v = vector(1000)
const w = vector(1000, 2)
const va = v.data as Float64Array
const wa = w.data as Float64Array
const tall = fromData(
  Float64Array.from({ length: 1000 }, (_, k) => k / 1000),
  [1000, 1],
)
const wide = fromData(
  Float64Array.from({ length: 1000 }, (_, k) => k / 1000),
  [10, 100],
)
const rows = fromData(
  Float64Array.from({ length: 1000 }, (_, k) => Math.cos(k)),
  [100, 10],
)
const row = vector(10, 3)
const m64 = fromData(
  Float64Array.from({ length: 64 * 64 }, (_, k) => Math.cos(k)),
  [64, 64],
)
const b50 = fromData(
  Float64Array.from({ length: 2500 }, (_, k) => Math.sin(k)),
  [50, 50],
)
const spd50 = add(matmul(b50, b50), mul(50, eye(50)))
const f = (x: Value) => sum(mul(sin(x), square(x)))
const g = (x: Value) => mul(sin(x), square(x))
const df = grad(f)
const dg = grad(g)
const ddg = grad(dg as (x: Value) => Value)

test('scalar dispatch', ({ bench }) =>
  group(bench, 'exp(0.3)', [
    ['exp primitive', () => void exp(0.3)],
    ['Math.exp', () => void Math.exp(0.3)],
  ]))

test('elementwise', ({ bench }) =>
  group(bench, 'add, 1000-vectors', [
    ['add(v, w)', () => void add(v, w)],
    ['add(m, row), 100×10 + 10', () => void add(rows, row)],
    ['mul(v, 2)', () => void mul(v, 2)],
    ['where(v > 0, v, w)', () => void where(greater(v, 0), v, w)],
    [
      'raw loop',
      () => {
        const out = new Float64Array(1000)
        for (let k = 0; k < 1000; k++) out[k] = va[k] + wa[k]
      },
    ],
  ]))

test('reductions', ({ bench }) =>
  group(bench, 'sum along axis 1', [
    ['sum(m, 1), 10×100', () => void sum(wide, 1)],
    ['sum(m, 1), 1000×1', () => void sum(tall, 1)],
  ]))

test('gradients', ({ bench }) =>
  group(bench, 'gradients', [
    ['grad(Σ sin x · x²), 1000-vector', () => void df(v)],
    ['grad(f), scalar', () => void dg(0.7)],
    ['grad(grad(f)), scalar', () => void ddg(0.7)],
  ]))

test('linear algebra', ({ bench }) =>
  group(bench, 'linear algebra', [
    ['matmul 64×64', () => void matmul(m64, m64)],
    ['cholesky 50×50', () => void cholesky(spd50)],
  ]))

// Small fixed-size matrices in a per-step recursion: the Kalman filter as the inner loop of a likelihood (ARMA, EM).
const harvey = parseModel(
  {
    A: [
      [0.5, 1],
      [0.2, 0],
    ],
    C: [[1, 0]],
    Q: [
      [1, 0.4],
      [0.4, 0.16],
    ],
    R: 0,
    m0: [0, 0],
    P0: eye(2),
  },
  'bench',
)
const series = Array.from({ length: 2000 }, (_, t) => [Math.sin(0.1 * t)])
const m2 = fromData(Float64Array.from([1, 2, 3, 4]), [2, 2])

test('small matrices', ({ bench }) =>
  group(bench, 'small matrices', [
    ['matmul 2×2', () => void matmul(m2, m2)],
    ['kalmanStep, 2 states', () => void kalmanStep(harvey, harvey.m0, harvey.P0, [0.3])],
    ['filterAll, 2 states × 2000', () => void filterAll(harvey, series)],
  ]))

test('normal draws', ({ bench }) =>
  group(bench, 'normal draws', [
    ['normal(s, 0, 1), scalar', () => void normal(s, 0, 1)],
    ['normal(s, 0, 1, { shape: [1000] })', () => void normal(s, 0, 1, { shape: [1000] })],
    [
      'Math.random × 1000',
      () => {
        const out = new Float64Array(1000)
        for (let k = 0; k < 1000; k++) out[k] = Math.random()
      },
    ],
  ]))

// Algorithms whose steps are written with primitives (so `unrolled` differentiates through them). Measured with a plain
// timing loop on 2026-10-01, raw arrays before → primitives after: gradient descent 500 steps 3.6 → 2.3 ms, Adam
// 6.5 → 4.8 ms, RK4 1000 steps 2.9 → 3.5 ms (one `linearCombination` per stage), Euler 1.1 → 1.1 ms.
const quadratic10 = (x: Vector) => {
  const a = toFlat(x)
  let value = 0
  const g = new Float64Array(10)
  for (let i = 0; i < 10; i++) {
    value += 0.5 * (i + 1) * a[i] * a[i]
    g[i] = (i + 1) * a[i]
  }
  return { value, grad: fromData(g, [10]) as Vector }
}
const start10 = { x0: Array.from({ length: 10 }, (_, i) => Math.cos(i)) }
const oscillator = (_t: number, x: Tensor) => {
  const a = toFlat(x)
  return [a[1], -a[0]]
}

const two = fromData(Float64Array.of(1, 2), [2])

/** Plain-JS classical RK4 on the oscillator: the reference for the primitive-based solver. */
function rk4Reference(steps: number, h: number): Float64Array {
  let x = Float64Array.of(1, 0)
  const f = (y: Float64Array) => Float64Array.of(y[1], -y[0])
  for (let t = 0; t < steps; t++) {
    const k1 = f(x)
    const k2 = f(Float64Array.from(x, (v, i) => v + 0.5 * h * k1[i]))
    const k3 = f(Float64Array.from(x, (v, i) => v + 0.5 * h * k2[i]))
    const k4 = f(Float64Array.from(x, (v, i) => v + h * k3[i]))
    x = Float64Array.from(x, (v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]))
  }
  return x
}

test('algorithms on primitives', ({ bench }) =>
  group(bench, 'first-order methods and Runge–Kutta', [
    [
      'gradientDescent, d = 10, 500 steps',
      () => void run(gradientDescent(quadratic10, { stepSize: 0.05, tolerance: 0 }), start10, 500),
    ],
    ['adam, d = 10, 500 steps', () => void run(adam(quadratic10, { stepSize: 0.05, tolerance: 0 }), start10, 500)],
    [
      'rungeKutta rk4, 2 states, 1000 steps',
      () => void run(rungeKutta(oscillator, 'rk4', { stepSize: 0.01 }), { x0: [1, 0] }, 1000),
    ],
    [
      'rungeKutta euler, 2 states, 1000 steps',
      () => void run(rungeKutta(oscillator, 'euler', { stepSize: 0.01 }), { x0: [1, 0] }, 1000),
    ],
    ['plain-JS RK4 reference, 1000 steps', () => void rk4Reference(1000, 0.01)],
    ['linearCombination of three 2-vectors', () => void linearCombination([two, two, two], [1, 0.5, 0.25])],
  ]))
