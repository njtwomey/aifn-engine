import { describe, expect, it } from 'vitest'
import * as R from 'aifn-compute/foundation/random'
import { chiSquareSf } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromRows, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'

/** The next 32-bit word of a stream. */
const word = (s: R.Stream): number => R.randomBits(s, 1)[0]

// All tests use fixed streams, so they are deterministic. Thresholds are set at significance levels near 1e-3 (or
// several standard errors), so a correct sampler passes for almost any seed and the fixed seeds cannot mask a bug.

const N = 20_000

function moments(xs: ArrayLike<number>): { mean: number; variance: number } {
  let m = 0
  for (let i = 0; i < xs.length; i++) m += xs[i]
  m /= xs.length
  let v = 0
  for (let i = 0; i < xs.length; i++) v += (xs[i] - m) ** 2
  return { mean: m, variance: v / (xs.length - 1) }
}

function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const ma = moments(a).mean
  const mb = moments(b).mean
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb)
    saa += (a[i] - ma) ** 2
    sbb += (b[i] - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}

/** Pearson chi-square p-value for observed counts against expected probabilities, pooling cells with expectation < 5. */
function chiSquarePValue(observed: number[], probabilities: number[], n: number): number {
  let stat = 0
  let cells = 0
  let obsPool = 0
  let expPool = 0
  for (let i = 0; i < observed.length; i++) {
    obsPool += observed[i]
    expPool += n * probabilities[i]
    if (expPool >= 5) {
      stat += (obsPool - expPool) ** 2 / expPool
      cells++
      obsPool = 0
      expPool = 0
    }
  }
  if (expPool > 0) {
    stat += (obsPool - expPool) ** 2 / Math.max(expPool, 1e-300)
    cells++
  }
  return chiSquareSf(stat, cells - 1)
}

describe('philox4x32-10', () => {
  it('matches the Random123 known-answer vectors', () => {
    const out = new Uint32Array(4)
    R.philox4x32(0, 0, 0, 0, 0, 0, out)
    expect([...out]).toEqual([0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8])
    const f = 0xffffffff
    R.philox4x32(f, f, f, f, f, f, out)
    expect([...out]).toEqual([0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd])
    R.philox4x32(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, out)
    expect([...out]).toEqual([0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1])
  })
})

describe('streams', () => {
  it('are deterministic, and their first values are fixed (a change here changes every figure)', () => {
    const a = R.stream(42)
    const b = R.stream(42)
    const xs = Array.from({ length: 10 }, () => word(a))
    expect(Array.from({ length: 10 }, () => word(b))).toEqual(xs)
    expect(xs.slice(0, 4)).toMatchInlineSnapshot(`
      [
        450079237,
        2369274609,
        3627297871,
        3524454626,
      ]
    `)
    expect(R.uniform(R.child(R.stream(7), 'chain', 3))).toMatchInlineSnapshot(`0.911639436531366`)
  })

  it('have readable, one-to-one keys', () => {
    expect(R.child(R.child(R.stream(7), 'chain', 3), 'env').key.path).toBe('7/chain:3/env')
    expect(R.child(R.stream('a/b'), 'x:y').key.path).toBe('a%2Fb/x%3Ay')
    expect(R.child(R.stream(7), 'a', 'b').key.path).not.toBe(R.child(R.stream(7), 'a:b').key.path)
  })

  it('treat numbers and their decimal strings as the same name', () => {
    expect(word(R.stream(7))).toBe(word(R.stream('7')))
    expect(word(R.child(R.stream(1), 3))).toBe(word(R.child(R.stream(1), '3')))
  })

  it('give different sequences for different paths, including ones that look alike', () => {
    const first = (s: R.Stream) => [word(s), word(s)].join(',')
    const root = R.stream(1)
    const variants = [
      root,
      R.child(root, 'a', 'b'),
      R.child(root, 'a:b'),
      R.child(R.child(root, 'a'), 'b'),
      R.child(root, 'ab'),
      R.child(root, ''),
      R.child(root),
      R.stream(2),
      R.stream('1/a'),
    ]
    expect(new Set(variants.map(first)).size).toBe(variants.length)
    // Many siblings: no two share their first 64 bits.
    const seen = new Set<string>()
    for (let k = 0; k < 20_000; k++) seen.add(first(R.child(root, k)))
    expect(seen.size).toBe(20_000)
  })

  it('give a child with no parts and a child with one empty part different paths', () => {
    const root = R.stream(1)
    expect(R.child(root).key.path).not.toBe(R.child(root, '').key.path)
    // replicate caches by path, so the two must not share results.
    const draw = (r: R.Stream) => R.uniform(r)
    const none = R.replicate(2, R.child(root), draw)
    const empty = R.replicate(2, R.child(root, ''), draw)
    expect(empty).not.toEqual(none)
  })

  it("child draws do not depend on the parent's or siblings' usage", () => {
    const fresh = R.child(R.stream(3), 'chain', 1)
    const expected = Array.from({ length: 50 }, () => R.uniform(fresh))
    const p = R.stream(3)
    for (let i = 0; i < 1234; i++) word(p)
    const sibling = R.child(p, 'chain', 0)
    for (let i = 0; i < 99; i++) R.uniform(sibling)
    const late = R.child(p, 'chain', 1)
    expect(Array.from({ length: 50 }, () => R.uniform(late))).toEqual(expected)
  })

  it('produce uncorrelated sequences across parent, children and siblings', () => {
    const s = R.stream(11)
    const draw = (t: R.Stream) => Float64Array.from({ length: N }, () => R.uniform(t))
    const parent = draw(R.stream(11))
    const pairs: [Float64Array, Float64Array][] = [
      [parent, draw(R.child(s, 0))],
      [draw(R.child(s, 0)), draw(R.child(s, 1))],
      [draw(R.child(s, 'x')), draw(R.child(s, 'x', ''))],
      [draw(R.child(R.stream(11), 5)), draw(R.child(R.stream(12), 5))],
    ]
    const bound = 4 / Math.sqrt(N)
    for (const [a, b] of pairs) expect(Math.abs(correlation(a, b))).toBeLessThan(bound)
    // Lag-1 autocorrelation within a stream.
    expect(Math.abs(correlation(parent.subarray(0, N - 1), parent.subarray(1)))).toBeLessThan(bound)
  })

  it('uniform lies in [0, 1) with the right moments, and every bit is fair', () => {
    const s = R.stream('bits')
    const u = Float64Array.from({ length: N }, () => R.uniform(s))
    expect(Math.min(...u)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...u)).toBeLessThan(1)
    const m = moments(u)
    expect(Math.abs(m.mean - 0.5)).toBeLessThan(4 * Math.sqrt(1 / 12 / N))
    expect(Math.abs(m.variance - 1 / 12)).toBeLessThan(0.003)
    const ones = new Array(32).fill(0)
    for (let i = 0; i < N; i++) {
      const w = word(s)
      for (let b = 0; b < 32; b++) ones[b] += (w >>> b) & 1
    }
    for (const c of ones) expect(Math.abs(c / N - 0.5)).toBeLessThan(4 * Math.sqrt(0.25 / N))
  })

  it('int(n) is uniform, including n above 2^32', () => {
    const s = R.stream('int')
    const counts = new Array(7).fill(0)
    for (let i = 0; i < N; i++) counts[R.integers(s, 7)]++
    expect(chiSquarePValue(counts, new Array(7).fill(1 / 7), N)).toBeGreaterThan(1e-3)
    const big = 3 * 2 ** 40 + 1
    const xs = Float64Array.from({ length: N }, () => R.integers(s, big))
    expect(xs.every((x) => Number.isInteger(x) && x >= 0 && x < big)).toBe(true)
    expect(Math.abs(moments(xs).mean / big - 0.5)).toBeLessThan(4 * Math.sqrt(1 / 12 / N))
    expect(() => R.integers(s, 0)).toThrow(DomainError)
  })
})

describe('other samplers', () => {
  it('normal uses both Box–Muller outputs and is unclamped', () => {
    const s = R.stream('normal')
    const z = Float64Array.from(toFlat(R.normals(s, 2 * N)))
    const m = moments(z)
    expect(Math.abs(m.mean)).toBeLessThan(4 / Math.sqrt(2 * N))
    expect(Math.abs(m.variance - 1)).toBeLessThan(0.05)
    // The pair (cos, sin) of one draw is uncorrelated.
    const even = z.filter((_, i) => i % 2 === 0)
    const odd = z.filter((_, i) => i % 2 === 1)
    expect(Math.abs(correlation(even, odd))).toBeLessThan(4 / Math.sqrt(N))
    // Two normals per two uniforms: n normals use n uniforms (2n words).
    const t = R.stream('count')
    R.normals(t, 10)
    const u = R.stream('count')
    for (let i = 0; i < 20; i++) word(u)
    expect(word(t)).toBe(word(u))
  })

  it('bernoulli', () => {
    const s = R.stream('bernoulli')
    let k = 0
    for (let i = 0; i < N; i++) k += R.bernoulli(s, 0.3) ? 1 : 0
    expect(Math.abs(k / N - 0.3)).toBeLessThan(4 * Math.sqrt(0.21 / N))
  })

  it('categorical and the alias table match the weights', () => {
    const w = [0.1, 0, 0.5, 0.25, 0.15]
    const p = w.map((x) => x / 1)
    const s = R.stream('categorical')
    const table = R.aliasTable(w.map((x) => 7 * x))
    for (const draw of [() => R.categorical(s, w), () => R.aliasSample(s, table)]) {
      const counts = new Array(w.length).fill(0)
      for (let i = 0; i < N; i++) counts[draw()]++
      expect(counts[1]).toBe(0)
      expect(
        chiSquarePValue(
          counts.filter((_, i) => i !== 1),
          p.filter((_, i) => i !== 1),
          N,
        ),
      ).toBeGreaterThan(1e-3)
    }
    expect(() => R.categorical(s, [0, 0])).toThrow(DomainError)
  })

  it('permutation and shuffle are uniform over all 24 orders of 4 items', () => {
    const s = R.stream('perm')
    const counts = new Map<string, number>()
    for (let i = 0; i < N; i++) {
      const key = toFlat(R.permutation(s, 4)).join('')
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    expect(counts.size).toBe(24)
    expect(chiSquarePValue([...counts.values()], new Array(24).fill(1 / 24), N)).toBeGreaterThan(1e-3)
    const arr = ['a', 'b', 'c']
    expect(R.shuffle(s, arr)).toBe(arr)
    expect([...arr].sort()).toEqual(['a', 'b', 'c'])
  })

  it('choice: uniform and weighted, with and without replacement', () => {
    const s = R.stream('choice')
    // Uniform without replacement: distinct, uniform marginals.
    const marg = new Array(10).fill(0)
    for (let i = 0; i < N / 10; i++) {
      const c = toFlat(R.choice(s, 10, 4, { replace: false }))
      expect(new Set(c).size).toBe(4)
      for (const k of c) marg[k]++
    }
    expect(chiSquarePValue(marg, new Array(10).fill(0.1), (N / 10) * 4)).toBeGreaterThan(1e-3)
    // Weighted without replacement: ordered pairs (i, j) have probability wᵢ/W · wⱼ/(W − wᵢ).
    const w = [1, 2, 3, 0]
    const pairs = new Map<string, number>()
    for (let i = 0; i < N; i++) {
      const c = toFlat(R.choice(s, 4, 2, { replace: false, weights: w }))
      pairs.set(`${c[0]}${c[1]}`, (pairs.get(`${c[0]}${c[1]}`) ?? 0) + 1)
    }
    const keys: string[] = []
    const probs: number[] = []
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        if (i !== j) {
          keys.push(`${i}${j}`)
          probs.push((w[i] / 6) * (w[j] / (6 - w[i])))
        }
    expect(
      chiSquarePValue(
        keys.map((k) => pairs.get(k) ?? 0),
        probs,
        N,
      ),
    ).toBeGreaterThan(1e-3)
    expect([...pairs.keys()].every((k) => keys.includes(k))).toBe(true)
    // Weighted with replacement.
    const counts = new Array(4).fill(0)
    for (const k of toFlat(R.choice(s, 4, N, { weights: w }))) counts[k]++
    expect(counts[3]).toBe(0)
    expect(chiSquarePValue(counts.slice(0, 3), [1 / 6, 2 / 6, 3 / 6], N)).toBeGreaterThan(1e-3)
    expect(() => R.choice(s, 3, 4, { replace: false })).toThrow(DomainError)
  })
})

describe('replicate', () => {
  it('runs fn on child k, caches by key and reuses a prefix', () => {
    const s = R.stream('rep')
    let calls = 0
    const fn = (r: R.Stream) => {
      calls++
      return R.uniform(r)
    }
    const five = R.replicate(5, s, fn)
    expect(calls).toBe(5)
    expect(five).toEqual([0, 1, 2, 3, 4].map((k) => R.uniform(R.child(s, k))))
    const eight = R.replicate(8, s, fn)
    expect(calls).toBe(8)
    expect(eight.slice(0, 5)).toEqual(five)
    // Another stream has other keys, so nothing is reused.
    R.replicate(2, R.stream('other'), fn)
    expect(calls).toBe(10)
    // An explicit cache, and no cache.
    const cache = new Map<string, number>()
    R.replicate(3, s, fn, { cache })
    expect(cache.size).toBe(3)
    R.replicate(3, s, fn, { cache: false })
    expect(calls).toBe(16)
  })
})

describe('samplers: shapes, broadcasting and element order', () => {
  const scalarCalls: [string, (s: R.Stream) => number, (s: R.Stream, shape: number[]) => Tensor][] = [
    ['uniform', (s) => R.uniform(s, -1, 3), (s, shape) => R.uniform(s, -1, 3, { shape })],
    ['normal', (s) => R.normal(s, 1, 2), (s, shape) => R.normal(s, 1, 2, { shape })],
    ['exponential', (s) => R.exponential(s, 2), (s, shape) => R.exponential(s, 2, { shape })],
    ['bernoulli', (s) => R.bernoulli(s, 0.3), (s, shape) => R.bernoulli(s, 0.3, { shape })],
  ]
  for (const [name, one, many] of scalarCalls) {
    it(`${name}: numbers give a number; a shape gives the same draws in row-major order`, () => {
      expect(typeof one(R.child(R.stream('shape'), name))).toBe('number')
      const want = toFlat(many(R.child(R.stream('shape'), name), [24]))
      // A number is the first draw of a block. Samplers that use one block word per draw also give successive numbers
      // equal to the block; normal draws pairs (Box–Muller, both outputs kept in a block), so its numbers do not.
      expect(one(R.child(R.stream('shape'), name))).toBe(want[0])
      if (name !== 'normal') {
        const a = R.child(R.stream('shape'), name)
        expect(Array.from({ length: 24 }, () => one(a))).toEqual(want)
      }
      for (const shape of [[24], [4, 6], [2, 3, 4], [1, 24, 1]]) {
        const t = many(R.child(R.stream('shape'), name), shape)
        expect(t.shape).toEqual(shape)
        expect(t.dtype).toBe('float64')
        expect(toFlat(t)).toEqual(want)
      }
      // Rank 0: a scalar tensor holding the first draw.
      const r0 = many(R.child(R.stream('shape'), name), [])
      expect(r0.shape).toEqual([])
      expect(toFlat(r0)).toEqual([want[0]])
    })
  }

  it('normals(s, n) is normal(s, 0, 1, { shape: [n] })', () => {
    const want = toFlat(R.normal(R.stream('normals'), 0, 1, { shape: [7] }))
    const z = R.normals(R.stream('normals'), 7)
    expect(z.shape).toEqual([7])
    expect(toFlat(z)).toEqual(want)
    expect(R.normals(R.stream('normals'), [7, 1]).shape).toEqual([7, 1])
  })

  it('tensor parameters broadcast, element by element, against each other and the shape', () => {
    const mean = tensor([0, 10, 20], [3, 1])
    const sd = tensor([1, 0.5])
    const t = R.normal(R.stream('bc'), mean, sd)
    expect(t.shape).toEqual([3, 2])
    // One block of standard normals in row-major order, scaled and shifted element by element.
    const z = toFlat(R.normal(R.stream('bc'), 0, 1, { shape: [3, 2] }))
    const want = [0, 10, 20].flatMap((m, i) => [1, 0.5].map((d, j) => m + d * z[2 * i + j]))
    toFlat(t).forEach((v, k) => expect(v).toBeCloseTo(want[k], 12))
    // A shape the parameters broadcast to; and one they do not.
    expect(R.normal(R.stream('bc'), mean, sd, { shape: [4, 3, 2] }).shape).toEqual([4, 3, 2])
    expect(() => R.normal(R.stream('bc'), mean, sd, { shape: [3] })).toThrow(ShapeError)
    // A number next to a tensor, and an int32 tensor parameter.
  })

  it('broadcast parameters give the right distribution in every cell', () => {
    const t = toRows(
      R.normal(R.stream('cells'), tensor([-5, 0, 5], [3, 1]), tensor([1, 2, 3], [3, 1]), { shape: [3, N] }),
    )
    t.forEach((row, i) => {
      const m = moments(row)
      expect(Math.abs(m.mean - (5 * i - 5))).toBeLessThan(4.5 * Math.sqrt((i + 1) ** 2 / N))
      expect(Math.abs(m.variance / (i + 1) ** 2 - 1)).toBeLessThan(0.05)
    })
  })

  it('categorical and aliasSample: numbers for one draw, int32 tensors for batches and shapes', () => {
    const w = [0.2, 0.5, 0.3]
    expect(typeof R.categorical(R.stream('cat'), w)).toBe('number')
    expect(R.categorical(R.stream('cat'), w)).toBe(R.categorical(R.stream('cat'), tensor(w)))
    const s = R.stream('cat')
    const want = Array.from({ length: 6 }, () => R.categorical(s, w))
    const t = R.categorical(R.stream('cat'), w, { shape: [2, 3] }) as Tensor
    expect(t.dtype).toBe('int32')
    expect(t.shape).toEqual([2, 3])
    expect(toFlat(t)).toEqual(want)
    // A batch of weight vectors: row i always picks index i.
    const batch = fromRows([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 1, 0],
    ])
    expect(toFlat(R.categorical(R.stream('cat'), batch) as Tensor)).toEqual([0, 1, 2, 1])
    expect((R.categorical(R.stream('cat'), batch, { shape: [5, 4] }) as Tensor).shape).toEqual([5, 4])
    expect(() => R.categorical(R.stream('cat'), batch, { shape: [3] })).toThrow(ShapeError)
    const table = R.aliasTable(tensor(w))
    const a = R.aliasSample(R.stream('alias'), table, { shape: [3, 2] })
    expect(a.dtype).toBe('int32')
    // A block draws its columns, then its coins, so a shape is one block in row-major order (not six single draws).
    expect(toFlat(a)).toEqual(toFlat(R.aliasSample(R.stream('alias'), table, { shape: [6] })))
  })

  it('index samplers return int32 tensors', () => {
    const p = R.permutation(R.stream('idx'), 5)
    expect(p.dtype).toBe('int32')
    expect([...toFlat(p)].sort()).toEqual([0, 1, 2, 3, 4])
    const c = R.choice(R.stream('idx'), 10, [2, 3], { replace: false, weights: tensor(Array(10).fill(1)) })
    expect(c.dtype).toBe('int32')
    expect(c.shape).toEqual([2, 3])
    expect(new Set(toFlat(c)).size).toBe(6)
  })
})

describe('negative weights (review 2026-10-01)', () => {
  it('categorical and aliasTable refuse a negative weight rather than biasing the others', () => {
    // [-1, 1, 1] sums to 1, so the scan used to reach index 1 for every u and never pick index 2.
    expect(() => R.categorical(R.stream(1), [-1, 1, 1])).toThrow(DomainError)
    expect(() =>
      R.categorical(
        R.stream(1),
        tensor([
          [0.5, 0.5],
          [2, -1],
        ]),
      ),
    ).toThrow(DomainError)
    expect(() => R.aliasTable([-1, 1, 1])).toThrow(DomainError)
    expect(R.categorical(R.stream(1), [0, 1, 0])).toBe(1)
  })
})
