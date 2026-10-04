import { describe, expect, it } from 'vitest'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { normals, stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { getLoss, mixtureDensityHead, mixtureDensityNll, mixtureHeadSize } from 'aifn-compute/learning/losses'

const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}
const LOG_2PI = Math.log(2 * Math.PI)

/** The NLL row by row from the definition, in plain numbers (no log-sum-exp). */
function direct(out: number[][], y: number[][], K: number, D: number, link: 'exp' | 'softplus', floor: number) {
  let total = 0
  out.forEach((o, i) => {
    const logits = o.slice(0, K)
    const m = Math.max(...logits)
    const z = logits.reduce((a, l) => a + Math.exp(l - m), 0)
    let p = 0
    for (let k = 0; k < K; k++) {
      let dens = Math.exp(logits[k] - m) / z
      for (let j = 0; j < D; j++) {
        const mu = o[K + k * D + j]
        const s = o[K + K * D + k * D + j]
        const sd = floor + (link === 'exp' ? Math.exp(s) : Math.log1p(Math.exp(s)))
        dens *= Math.exp(-0.5 * ((y[i][j] - mu) / sd) ** 2 - Math.log(sd) - 0.5 * LOG_2PI)
      }
      p += dens
    }
    total += -Math.log(p)
  })
  return total / out.length
}

const rows = (t: Tensor, d: number) => {
  const f = toFlat(t)
  return Array.from({ length: f.length / d }, (_, i) => Array.from(f.slice(i * d, (i + 1) * d)))
}

describe('mixtureDensityNll', () => {
  it('is registered as a regression loss on a distribution head', () => {
    expect(getLoss('mixtureDensityNll').info.family).toBe('regression')
    expect(getLoss('mixtureDensityNll').info.inputs).toBe('distribution')
    expect(mixtureHeadSize(3, 2)).toBe(15)
  })

  for (const [K, D] of [
    [1, 1],
    [3, 1],
    [2, 2],
  ] as const)
    for (const link of ['exp', 'softplus'] as const)
      it(`matches the definition (K ${K}, D ${D}, ${link}) and its gradient checks`, () => {
        const n = 5
        const out = normals(stream(`mdn-${K}-${D}`), [n, mixtureHeadSize(K, D)], 0, 0.7) as Tensor
        const y = normals(stream(`y-${K}-${D}`), D === 1 ? [n] : [n, D]) as Tensor
        const floor = 1e-3
        const value = num(mixtureDensityNll(out, y, { components: K, dims: D, scale: link, floor }))
        expect(value).toBeCloseTo(direct(rows(out, mixtureHeadSize(K, D)), rows(y, D), K, D, link, floor), 10)
        const report = gradCheck((o: Tensor) => mixtureDensityNll(o, y, { components: K, dims: D, scale: link }), out)
        expect(report.ok).toBe(true)
      })

  it('K = 1 is the Gaussian negative log-likelihood', () => {
    const out = tensor([
      [0, 0.3, Math.log(0.5)],
      [2, -1, 0],
    ])
    const y = [0.1, -0.4]
    const expected =
      (0.5 * ((0.1 - 0.3) / 0.5) ** 2 + Math.log(0.5) + 0.5 * LOG_2PI + 0.5 * 0.6 ** 2 + 0.5 * LOG_2PI) / 2
    expect(num(mixtureDensityNll(out, y, { components: 1, floor: 0 }))).toBeCloseTo(expected, 12)
  })

  it('stays finite where a direct sum underflows', () => {
    // Every component is 60 standard deviations away: exp(−1800) underflows, the log-sum-exp does not.
    const out = tensor([[0, 0, 0, 0, Math.log(0.1), Math.log(0.1)]])
    const v = num(mixtureDensityNll(out, [6], { components: 2, floor: 0 }))
    expect(Number.isFinite(v)).toBe(true)
    expect(v).toBeCloseTo(1800 + Math.log(0.1) + 0.5 * LOG_2PI, 8)
  })

  it('rejects a head of the wrong width', () => {
    expect(() => mixtureDensityNll(tensor([[0, 0, 0, 0]]), [1], { components: 2 })).toThrow(/expected outputs/)
  })
})

describe('mixtureDensityHead', () => {
  // Two well-separated components at −2 and 3 (weights ¼, ¾, σ 0.5 and 1) and one row with a single component.
  const head = fromData(Float64Array.of(Math.log(1), Math.log(3), -2, 3, Math.log(0.5), 0), [1, 6])
  const mix = mixtureDensityHead(head, { components: 2, floor: 0 })

  it('reads weights, means and scales, and the law of total variance', () => {
    expect(mix.row(0).weights[1]).toBeCloseTo(0.75, 12)
    expect(mix.row(0).means).toEqual([[-2], [3]])
    expect(mix.row(0).scales[0][0]).toBeCloseTo(0.5, 12)
    expect(mix.mean()[0]).toBeCloseTo(0.25 * -2 + 0.75 * 3, 12)
    const m = 1.75
    expect(mix.variance()[0]).toBeCloseTo(0.25 * (0.25 + (-2 - m) ** 2) + 0.75 * (1 + (3 - m) ** 2), 12)
  })

  it('its density agrees with the compute Mixture of Normals and with the loss', () => {
    const law = mix.distribution()
    for (const y of [-2, 0, 0.5, 3, 5]) expect(mix.logDensity(0, [y])).toBeCloseTo(num(law.logProb(y)), 10)
    // A batch: one weight vector per row.
    const two = mixtureDensityHead(
      tensor([
        [0, 0, -1, 1, 0, 0],
        [0, 2, 0, 4, -1, 0.5],
      ]),
      { components: 2, floor: 0 },
    )
    const batch = two.distribution()
    expect(batch.batchShape).toEqual([2])
    const lp = toFlat(unwrap(batch.logProb(tensor([0.3, 3.5]))) as Tensor)
    expect(lp[0]).toBeCloseTo(two.logDensity(0, [0.3]), 10)
    expect(lp[1]).toBeCloseTo(two.logDensity(1, [3.5]), 10)
    expect(toFlat(unwrap(batch.mean()) as Tensor)[1]).toBeCloseTo(two.mean()[1], 10)
    const draws = batch.sample(stream('batch'), { shape: [500] }) as Tensor
    expect(draws.shape).toEqual([500, 2])
    expect(-mix.logDensity(0, [0.5])).toBeCloseTo(num(mixtureDensityNll(head, [0.5], { components: 2, floor: 0 })), 10)
  })

  it('finds both modes, the heavier first, and merges coincident components', () => {
    const modes = mix.modes(0)
    expect(modes.length).toBe(2)
    expect(modes[0].value[0]).toBeCloseTo(3, 3)
    expect(modes[1].value[0]).toBeCloseTo(-2, 3)
    const same = mixtureDensityHead(tensor([[0, 0, 1, 1, 0, 0]]), { components: 2, floor: 0 })
    expect(same.modes(0).length).toBe(1)
    // Two overlapping components (1 σ apart) have one mode between them.
    const close = mixtureDensityHead(tensor([[0, 0, -0.5, 0.5, 0, 0]]), { components: 2, floor: 0 })
    const m = close.modes(0)
    expect(m.length).toBe(1)
    expect(m[0].value[0]).toBeCloseTo(0, 6)
  })

  it('draws from the mixture (seeded): component shares and means', () => {
    const draws = mix.sample(stream('draws'), 0, 4000).map((d) => d[0])
    const right = draws.filter((v) => v > 0.5).length / draws.length
    expect(right).toBeGreaterThan(0.72)
    expect(right).toBeLessThan(0.78)
    expect(draws.reduce((a, v) => a + v, 0) / draws.length).toBeCloseTo(1.75, 1)
    expect(mix.sample(stream('draws'), 0, 3)).toEqual(mix.sample(stream('draws'), 0, 3))
  })

  it('handles diagonal 2-d targets', () => {
    const two = mixtureDensityHead(tensor([[0, 0, 2, 2, -2, -2, 0, 0, 0, 0]]), { components: 2, dims: 2, floor: 0 })
    expect(two.mean()[0]).toBeCloseTo(0, 12)
    expect(
      two
        .modes(0)
        .map((m) => m.value[0])
        .sort(),
    ).toEqual([expect.closeTo(-2, 3), expect.closeTo(2, 3)])
    const marginals = two.distribution()
    expect(marginals.batchShape).toEqual([1, 2])
    // The first coordinate's marginal: ½N(2, 1) + ½N(−2, 1).
    const at = (y: number) =>
      Math.log(0.5 * Math.exp(-0.5 * (y - 2) ** 2) + 0.5 * Math.exp(-0.5 * (y + 2) ** 2)) - 0.5 * LOG_2PI
    expect(toFlat(unwrap(marginals.logProb(tensor([[0.7, 0.7]]))) as Tensor)[0]).toBeCloseTo(at(0.7), 10)
  })
})
