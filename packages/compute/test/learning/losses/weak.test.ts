/**
 * Weak-supervision losses against their formulas on hand-computed inputs and the laws that define them: the uPU risk
 * equals the positive–negative risk when the unlabelled set is the exact mixture; nnPU clips the negative part and its
 * training objective flips sign below −β; the proportion loss of singleton bags is the cross-entropy; the unbiased
 * complementary-label risk averages to the cross-entropy over uniform complements; and the gradients exist.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { fromRows, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  complementaryLabelLoss,
  nonNegativePu,
  proportionLoss,
  softmaxCrossEntropy,
  unbiasedPu,
} from 'aifn-compute/learning/losses'

const num = (v: unknown) => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])
const sig = (z: number) => 1 / (1 + Math.exp(-z))
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

describe('PU risks', () => {
  const pos = [2, 0.5, -0.3]
  const neg = [-1.5, 0.2, -2.5, -0.1]
  const prior = 0.4

  it('uPU equals π R_P⁺ + R_U⁻ − π R_P⁻ with the sigmoid surrogate', () => {
    const unl = [...pos, ...neg]
    const want =
      prior * mean(pos.map((z) => sig(-z))) + mean(unl.map((z) => sig(z))) - prior * mean(pos.map((z) => sig(z)))
    expect(num(unbiasedPu(tensor(pos), tensor(unl), { prior }))).toBeCloseTo(want, 12)
  })

  it('uPU is the PN risk when the unlabelled set is the exact π-mixture', () => {
    // Unlabelled = three copies of the positives and two of the negatives, so π = 6/12.
    const p = [1.2, -0.4]
    const n = [-0.8, 0.3, -2]
    const unlabelled = [...p, ...p, ...p, ...n, ...n]
    const realPi = 6 / unlabelled.length
    const pn =
      realPi * mean(p.map((z) => Math.log1p(Math.exp(-z)))) + (1 - realPi) * mean(n.map((z) => Math.log1p(Math.exp(z))))
    const r = num(unbiasedPu(tensor(p), tensor(unlabelled), { prior: realPi, surrogate: 'logistic' }))
    expect(r).toBeCloseTo(pn, 12)
  })

  it('nnPU clips the negative part at zero, and its training objective flips below −β', () => {
    // Positives scored high and unlabelled scored like positives: the negative part goes below zero.
    const p = [3, 3, 3]
    const u = [3, 3]
    const upu = num(unbiasedPu(tensor(p), tensor(u), { prior: 0.9 }))
    const nn = num(nonNegativePu(tensor(p), tensor(u), { prior: 0.9 }))
    const positivePart = 0.9 * sig(-3)
    const negativePart = sig(3) - 0.9 * sig(3)
    expect(upu).toBeCloseTo(positivePart + negativePart, 12)
    expect(negativePart).toBeGreaterThan(0)
    expect(nn).toBeCloseTo(positivePart + Math.max(0, negativePart), 12)
    const below = num(nonNegativePu(tensor([3]), tensor([-3]), { prior: 0.9, training: true, gamma: 0.5 }))
    const neg2 = sig(-3) - 0.9 * sig(3)
    expect(neg2).toBeLessThan(0)
    expect(below).toBeCloseTo(-0.5 * neg2, 12)
  })

  it('has gradients in the scores', () => {
    const g = grad((s: Tensor) => unbiasedPu(s, tensor(neg), { prior }))(tensor(pos)) as Tensor
    pos.forEach((z, i) =>
      expect(toFlat(g)[i]).toBeCloseTo((prior / pos.length) * (-sig(-z) * sig(z) - sig(z) * sig(-z)), 10),
    )
  })
})

describe('proportionLoss', () => {
  it('is the cross-entropy when every bag holds one instance with a one-hot proportion', () => {
    const logits = fromRows([
      [1, 0, -1],
      [0.2, 0.5, 2],
    ])
    const got = num(
      proportionLoss(
        logits,
        [0, 1],
        fromRows([
          [1, 0, 0],
          [0, 0, 1],
        ]),
      ),
    )
    const want = num(softmaxCrossEntropy(logits, [0, 2]))
    expect(got).toBeCloseTo(want, 10)
  })

  it('compares each bag’s mean probability with its proportions', () => {
    const logits = fromRows([
      [0, 0],
      [Math.log(3), 0],
      [0, Math.log(4)],
    ])
    // Bag 0 = rows 0, 1: mean p = ((0.5 + 0.75)/2, (0.5 + 0.25)/2) = (0.625, 0.375). Bag 1 = row 2: (0.2, 0.8).
    const pr = [
      [0.5, 0.5],
      [0, 1],
    ]
    const want = (-(0.5 * Math.log(0.625) + 0.5 * Math.log(0.375)) - Math.log(0.8)) / 2
    expect(num(proportionLoss(logits, [0, 0, 1], fromRows(pr)))).toBeCloseTo(want, 10)
  })
})

describe('complementaryLabelLoss', () => {
  const logits = fromRows([
    [1, -0.5, 0.3],
    [0.1, 2, -1],
  ])
  const p = (r: number[]) => {
    const e = r.map(Math.exp)
    const s = e.reduce((a, b) => a + b)
    return e.map((v) => v / s)
  }
  const rows = [
    [1, -0.5, 0.3],
    [0.1, 2, -1],
  ]

  it('the unbiased risk averaged over uniform complements is the cross-entropy of the true class', () => {
    // For one example with true class y, E_ȳ[(K − 1) log p_ȳ − Σ log p_k] over ȳ ≠ y equals −log p_y.
    rows.forEach((r, i) => {
      const y = i
      const others = [0, 1, 2].filter((k) => k !== y)
      const avg = mean(others.map((bar) => num(complementaryLabelLoss(fromRows([r]), [bar], { method: 'unbiased' }))))
      expect(avg).toBeCloseTo(-Math.log(p(r)[y]), 10)
    })
  })

  it('the forward loss is −log((1 − p_ȳ)/(K − 1))', () => {
    const got = num(complementaryLabelLoss(logits, [2, 0]))
    const want = mean([-Math.log((1 - p(rows[0])[2]) / 2), -Math.log((1 - p(rows[1])[0]) / 2)])
    expect(got).toBeCloseTo(want, 12)
  })

  it('the forward loss and its gradient stay finite when p_ȳ rounds to 1', () => {
    // z_ȳ leads by 50: 1 − p_ȳ ≈ 2e-22 underflows against 1, but log(1 − p_ȳ) = −50 + log 2 exactly enough.
    const z = fromRows([[50, 0, 0]])
    const got = num(complementaryLabelLoss(z, [0]))
    const lse = 50 + Math.log1p(2 * Math.exp(-50))
    expect(got).toBeCloseTo(Math.log(2) - (Math.log(2) - lse), 9)
    const g = Array.from(toFlat(grad((x: Tensor) => complementaryLabelLoss(x, [0]) as Tensor)(z) as Tensor))
    g.forEach((v) => expect(Number.isFinite(v)).toBe(true))
    // d/dz_ȳ = p_ȳ ≈ 1; the other two share −(1 − p_ȳ)/2 + … ≈ −0.5 each.
    expect(g[0]).toBeCloseTo(1, 9)
    expect(g[1]).toBeCloseTo(-0.5, 9)
  })
})
