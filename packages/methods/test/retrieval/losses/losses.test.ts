/**
 * Ranking and retrieval losses against torch. The golden values are cases of the compute losses fixture
 * (`compute/test/fixtures/learning/losses.json`, whose generator writes every loss), loaded by path, and of this
 * package's `fixtures/retrieval/losses.json` (in-batch softmax).
 */
import { describe, expect, it } from 'vitest'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { infoNce, type Loss } from 'aifn-compute/learning/losses'
import {
  approxNdcg,
  bpr,
  contrastive,
  inBatchSoftmax,
  lambdaRank,
  lambdaWeights,
  listMle,
  listNet,
  listwiseSoftmax,
  noiseContrastiveEstimation,
  pairwiseHinge,
  pointwiseBce,
  pointwiseSquaredError,
  rankNet,
  retrievalLossRegistry,
  sampledSoftmax,
  triplet,
  warp,
  warpRankWeight,
  warpWeights,
} from 'aifn-methods/retrieval/losses'
import { fork, integers, stream } from 'aifn-compute/foundation/random'
import { fixture } from '../../fixtures'

const F = fixture<Record<string, Record<string, unknown>>>('../../../compute/test/fixtures/learning/losses')
const T = (x: unknown) => tensor(x as number[][])
const flat = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v as number])
const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/** Check a loss of the predictions against a fixture's value and gradient. */
function check(f: (x: Value) => Value, x: unknown, fc: unknown, tol = 1e-10) {
  const c = fc as { value: number; grad: unknown }
  const r = valueAndGrad(f)(T(x))
  expect(Math.abs(num(r.value as Value) - c.value)).toBeLessThan(tol * Math.max(1, Math.abs(c.value)))
  const g = toFlat(r.grad as Tensor)
  flat(c.grad).forEach((e, i) => expect(Math.abs(g[i] - e)).toBeLessThan(tol * Math.max(1, Math.abs(e))))
}

describe('ranking losses match a direct transcription', () => {
  const c = F.ranking
  const rel = c.rel as unknown as number[]
  const losses: Record<string, (s: Value, r: number[]) => Value> = {
    rankNet,
    pairwiseHinge,
    lambdaRank,
    listwiseSoftmax,
    listNet,
    listMle,
    pointwiseBce,
    pointwiseSquaredError,
    approxNdcg,
  }
  for (const [name, loss] of Object.entries(losses)) {
    it(name, () => check((s) => loss(s, rel), c.scores, c[name]))
  }

  it('batches of lists reduce over lists', () => {
    const s = c.scores as unknown as number[]
    const one = num(listMle(tensor(s), rel))
    const two = num(listMle(tensor([s, s]), tensor([rel, rel]), { reduction: 'sum' }))
    expect(two).toBeCloseTo(2 * one, 12)
  })

  it('lambda weights vanish for equal grades and a list without relevance', () => {
    const w = toFlat(lambdaWeights(tensor([1, 2, 3]), [0, 0, 0]))
    expect(w.every((v) => v === 0)).toBe(true)
  })

  it('BPR is RankNet on one pair', () => {
    expect(num(bpr(tensor([2]), tensor([0.5])))).toBeCloseTo(num(rankNet(tensor([2, 0.5]), [1, 0])), 12)
  })
})

describe('retrieval and representation losses', () => {
  it('sampled softmax with the logQ correction and NCE match torch', () => {
    const s = F.sampledSoftmax
    check((p) => sampledSoftmax(p, T(s.neg), { logQ: T(s.logQ) }), s.pos, s)
    const n = F.nce
    check(
      (p) =>
        noiseContrastiveEstimation(p, T(n.neg), {
          logNoisePositive: n.logQPositive as number[],
          logNoiseNegatives: T(n.logQ),
        }),
      n.pos,
      n,
    )
  })

  it('the triplet loss matches torch (InfoNCE is tested with compute)', () => {
    const t = F.triplet
    check((a) => triplet(a, T(t.p), T(t.n)), t.a, t)
  })

  it('in-batch softmax is InfoNCE with dot products at τ = 1', () => {
    const a = tensor([
      [1, 0.5],
      [0.2, -1],
    ])
    const b = tensor([
      [0.3, 0.1],
      [-0.4, 2],
    ])
    expect(num(inBatchSoftmax(a, b))).toBeCloseTo(num(infoNce(a, b, { similarity: 'dot', temperature: 1 })), 12)
  })

  it('the contrastive loss pulls similar pairs and pushes dissimilar ones', () => {
    const x1 = tensor([[0, 0]])
    const x2 = tensor([[0.3, 0.4]])
    expect(num(contrastive(x1, x2, [1]))).toBeCloseTo(0.125, 12)
    expect(num(contrastive(x1, x2, [0]))).toBeCloseTo(0.125, 12)
  })
})

describe('registry', () => {
  it('every ranking and retrieval loss is registered with its note', () => {
    const all = Object.values(retrievalLossRegistry) as Loss[]
    expect(all.length).toBeGreaterThanOrEqual(15)
    for (const l of all) {
      expect(l.info.kind).toBe('loss')
      for (const n of l.info.notes ?? []) expect(n).toMatch(/^[a-z0-9-]+$/)
    }
    expect(all.filter((l) => l.info.family === 'ranking').map((l) => l.info.key)).toContain('listMle')
  })
})

describe('WARP (Weston, Bengio & Usunier, 2011)', () => {
  const harmonic = (r: number) => Array.from({ length: r }, (_, k) => 1 / (k + 1)).reduce((a, b) => a + b, 0)

  it("L(r) is the harmonic number (or r with constant weights); the note's numbers", () => {
    expect(warpRankWeight(0)).toBe(0)
    expect(warpRankWeight(1000)).toBeCloseTo(7.485, 3)
    expect(warpRankWeight(100)).toBeCloseTo(5.187, 3)
    expect(warpRankWeight(10)).toBeCloseTo(2.929, 3)
    expect(warpRankWeight(7, 'constant')).toBe(7)
  })

  it('exact: violators share L(r)/r, so the loss is L(r) times the mean violating hinge', () => {
    // Positive 2; negatives within the margin of 1 (or above): 1.5, 2.5, 3 → r = 3. 0 and -1 do not violate.
    const s = 2
    const negs = [1.5, 0, 2.5, -1, 3]
    const w = toFlat(warpWeights(tensor([s]), tensor([negs])))
    const each = harmonic(3) / 3
    expect(w).toEqual([each, 0, each, 0, each].map((v) => expect.closeTo(v, 14)))
    const hinges = negs.map((v) => Math.max(0, 1 - s + v))
    const want = (harmonic(3) * hinges.reduce((a, b) => a + b, 0)) / 3
    expect(num(warp(tensor([s]), tensor([negs])))).toBeCloseTo(want, 12)
    // The weights are constants: the gradient in a violator's score is its weight, in the positive −L(r).
    const { grad } = valueAndGrad((n: Value) => warp(tensor([s]), n))(tensor([negs]))
    expect(toFlat(grad as Tensor)).toEqual(w.map((v) => expect.closeTo(v, 12)))
    const gp = valueAndGrad((p: Value) => warp(p, tensor([negs])))(tensor([s])).grad
    expect(toFlat(gp as Tensor)[0]).toBeCloseTo(-harmonic(3), 12)
    // No violator: zero loss.
    expect(num(warp(tensor([10]), tensor([negs])))).toBe(0)
  })

  it('sampled: the first violator found on draw N gets L(⌊M/N⌋), replayed from the same stream', () => {
    const M = 40
    const negs = Array.from({ length: M }, (_, j) => (j === 17 || j === 31 ? 5 : -5))
    const s = 0
    for (const seed of [1, 2, 3, 4, 5]) {
      const st = stream(seed)
      const replay = fork(st)
      const w = toFlat(warpWeights(tensor([s]), tensor([negs]), { stream: st }))
      let N = 0
      let j = -1
      while (j !== 17 && j !== 31) {
        j = integers(replay, M)
        N++
      }
      const want = new Array(M).fill(0)
      want[j] = harmonic(Math.floor(M / N))
      expect(w).toEqual(want.map((v) => expect.closeTo(v, 14)))
    }
  })

  it('sampled: every negative violating gives N = 1 and the top weight L(M); rarer violators give smaller weights', () => {
    const M = 50
    const all = toFlat(warpWeights(tensor([0]), tensor([new Array(M).fill(1)]), { stream: stream('w') }))
    expect(all.filter((v) => v !== 0)).toEqual([expect.closeTo(harmonic(M), 12)])
    // One violator in M: N is geometric with mean M, so the mean weight over many draws is far below L(M).
    const one = Array.from({ length: M }, (_, j) => (j === 0 ? 1 : -5))
    let total = 0
    const runs = 200
    for (let r = 0; r < runs; r++)
      total += toFlat(warpWeights(tensor([0]), tensor([one]), { stream: stream(`one-${r}`), maxDraws: 10 * M }))[0]
    expect(total / runs).toBeLessThan(harmonic(M) / 2)
    expect(total / runs).toBeGreaterThan(0)
    // With a cap below the first hit, the positive contributes nothing.
    expect(num(warp(tensor([0]), tensor([[-5, -5, -5, 1]]), { stream: stream(9), maxDraws: 0 }))).toBe(0)
  })

  it('is registered as a ranking loss', () => {
    expect(retrievalLossRegistry.warp.info.family).toBe('ranking')
    expect(retrievalLossRegistry.warp.info.glossary).toBe('warp')
  })
})

describe('in-batch softmax against torch', () => {
  type Case = { value: number; gradQueries: number[][]; gradItems: number[][]; temperature?: number }
  const B = fixture<{
    inBatchSoftmax: { queries: number[][]; items: number[][]; logQ: number[] } & Record<
      'plain' | 'corrected' | 'symmetric',
      Case
    >
  }>('retrieval/losses').inBatchSoftmax
  const cases: ['plain' | 'corrected' | 'symmetric', object][] = [
    ['plain', {}],
    ['corrected', { temperature: 0.5, logQ: B.logQ }],
    ['symmetric', { temperature: 0.5, logQ: B.logQ, symmetric: true }],
  ]
  for (const [name, options] of cases)
    it(`${name}: value and both towers' gradients`, () => {
      const c = B[name]
      const r = valueAndGrad((q: Value, v: Value) => inBatchSoftmax(q, v, options), { argnums: [0, 1] })(
        tensor(B.queries),
        tensor(B.items),
      )
      expect(num(r.value as Value)).toBeCloseTo(c.value, 12)
      const [gq, gv] = r.grad as Tensor[]
      flat(c.gradQueries).forEach((e, i) => expect(Math.abs(toFlat(gq)[i] - e)).toBeLessThan(1e-12))
      flat(c.gradItems).forEach((e, i) => expect(Math.abs(toFlat(gv)[i] - e)).toBeLessThan(1e-12))
    })
})
