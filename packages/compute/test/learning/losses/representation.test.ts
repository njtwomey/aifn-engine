import { describe, expect, it } from 'vitest'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { getLoss, infoNce, learnedTemperature } from 'aifn-compute/learning/losses'
import { fixture } from '../../fixtures'

type Case = { a: number[][]; b: number[][]; value: number; grad: number[][] }
const F = fixture<Record<string, unknown>>('learning/losses') as { infoNce: Case }
const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/** InfoNCE written out row by row, with numbers only. */
function direct(a: number[][], b: number[][], tau: number, symmetric: boolean): number {
  const unit = (r: number[]) => r.map((v) => v / Math.hypot(...r))
  const [A, B] = [a.map(unit), b.map(unit)]
  const s = A.map((ai) => B.map((bj) => ai.reduce((t, v, k) => t + v * bj[k], 0) / tau))
  const ce = (m: number[][]) =>
    m.reduce((t, row, i) => t + Math.log(row.reduce((z, v) => z + Math.exp(v), 0)) - row[i], 0) / m.length
  const forward = ce(s)
  if (!symmetric) return forward
  const sT = s.map((_, i) => s.map((row) => row[i]))
  return 0.5 * (forward + ce(sT))
}

describe('infoNce', () => {
  it('matches torch (symmetric, cosine, τ = 0.5) in value and gradient', () => {
    const c = F.infoNce
    const r = valueAndGrad((a: Value) => infoNce(a, tensor(c.b), { temperature: 0.5, symmetric: true }))(tensor(c.a))
    expect(num(r.value as Value)).toBeCloseTo(c.value, 10)
    const g = toFlat(r.grad as Tensor)
    c.grad.flat().forEach((e, i) => expect(g[i]).toBeCloseTo(e, 10))
  })

  it('matches a direct transcription in both directions', () => {
    const a = [
      [1, 0.2],
      [-0.3, 1],
      [0.5, -0.8],
    ]
    const b = [
      [0.9, 0.1],
      [0.1, 1.2],
      [0.4, -1],
    ]
    for (const symmetric of [false, true])
      expect(num(infoNce(tensor(a), tensor(b), { temperature: 0.3, symmetric }))).toBeCloseTo(
        direct(a, b, 0.3, symmetric),
        12,
      )
  })

  it('learns its temperature: the gradient in log(1/τ) matches finite differences', () => {
    const a = tensor([
      [1, 0.2],
      [-0.3, 1],
      [0.5, -0.8],
    ])
    const b = tensor([
      [0.9, 0.1],
      [0.1, 1.2],
      [0.4, -1],
    ])
    const loss = (s: Value) => infoNce(a, b, { temperature: learnedTemperature(s), symmetric: true })
    const s0 = Math.log(1 / 0.2)
    const r = valueAndGrad(loss)(s0)
    const h = 1e-6
    const fd = (num(loss(s0 + h)) - num(loss(s0 - h))) / (2 * h)
    expect(num(r.grad as Value)).toBeCloseTo(fd, 6)
    expect(num(r.value as Value)).toBeCloseTo(direct(toRows(a), toRows(b), 0.2, true), 12)
  })

  it('clips the logit scale at maxScale, so the temperature stops falling there', () => {
    expect(num(learnedTemperature(Math.log(1 / 0.07)))).toBeCloseTo(0.07, 12)
    expect(num(learnedTemperature(10))).toBeCloseTo(1 / 100, 12)
    expect(num(learnedTemperature(5, { maxScale: 1000 }))).toBeCloseTo(Math.exp(-5), 12)
    expect(num(valueAndGrad((s: Value) => learnedTemperature(s))(10).grad as Value) + 0).toBe(0)
  })

  it('is registered as a representation loss', () => {
    expect(getLoss('infoNce').info.family).toBe('representation')
  })
})

function toRows(t: Tensor): number[][] {
  const v = toFlat(t)
  const [n, d] = t.shape
  return Array.from({ length: n }, (_, i) => Array.from(v.slice(i * d, (i + 1) * d)))
}
