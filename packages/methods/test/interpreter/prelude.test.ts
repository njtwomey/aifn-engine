import { expect, test } from 'vitest'
import { runProgram } from 'aifn-compute/interpreter'
import { prelude } from 'aifn-methods/interpreter'

test('fitLinear recovers a line from a program', () => {
  const r = runProgram(
    'seed(1)\nconst x = array.linspace(0, 1, 50)\nconst e = random.normal(50, 0, 0.01)\nreturn learn.fitLinear(x, x.map((v, i) => 2 * v + 1 + e[i]))',
    {
      prelude,
    },
  )
  expect(r.ok).toBe(true)
  const m = (r as { value: { weights: number[]; intercept: number; r2: number; predict(x: number[]): number[] } }).value
  expect(m.weights[0]).toBeCloseTo(2, 1)
  expect(m.intercept).toBeCloseTo(1, 1)
  expect(m.r2).toBeGreaterThan(0.99)
  expect(m.predict([0])[0]).toBeCloseTo(1, 1)
})

test('fitLogistic separates two clouds made by a program', () => {
  const r = runProgram(
    `seed(3)
function make(n = 160, gap = 4) {
  const labels = random.bernoulli(0.5, n)
  const X = array.stack([random.normal(n).map((z, i) => z + gap * labels[i]), random.normal(n)], 1)
  return learn.fitLogistic(X, labels)
}`,
    { prelude },
  )
  expect(r.ok).toBe(true)
  const m = (r as { value: { accuracy: number; weights: number[] } }).value
  expect(m.accuracy).toBeGreaterThan(0.9)
  expect(m.weights[0]).toBeGreaterThan(0)
})
