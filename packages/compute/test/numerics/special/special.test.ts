import { describe, expect, it } from 'vitest'
import * as S from 'aifn-compute/numerics/special'
import {
  get,
  logsumexp,
  tensor,
  toFlat,
  toRows,
  transpose,
  unwrap,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { grad, jvp } from 'aifn-compute/foundation/autodiff'
import { NotDifferentiableError } from 'aifn-compute/foundation/errors'
import { registry } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'
import { checkGradient } from '../../foundation/tensor/check-gradient'

type Unary = { x: number[]; y: number[] }
type Binary = { a: number[]; b: number[]; y: number[] }
type Ternary = { a: number[]; b: number[]; c: number[]; y: number[] }
type VectorCase = { x: number[]; t?: number; y: number | number[] }
type Fixture = {
  unary: Record<string, Unary>
  binary: Record<string, Binary>
  ternary: Record<string, Ternary>
  vector: Record<string, VectorCase[]>
  softplus: Unary
}
const f = fixture<Fixture>('numerics/special')

/**
 * Error of `got` against `want`: relative, but measured against `floor` when |want| is below it (so functions with
 * zeros are checked in absolute terms there). Equal infinities and NaNs match.
 */
function error(got: number, want: number, floor: number): number {
  if (Object.is(got, want) || (Number.isNaN(got) && Number.isNaN(want))) return 0
  if (!Number.isFinite(want) || !Number.isFinite(got)) return Infinity
  return Math.abs(got - want) / Math.max(Math.abs(want), floor)
}

/** Relative tolerance and absolute floor for each fixture-checked function. */
const TOLERANCE: Record<string, [rel: number, floor: number]> = {
  erf: [1e-14, 1e-300],
  erfc: [1e-13, 1e-300],
  erfcx: [1e-13, 1e-300],
  logErfc: [1e-14, 1e-300],
  normalPdf: [1e-13, 1e-300],
  // The acceptance bar from the survey (§4.3): Φ to relative 1e-12 far into the lower tail.
  normalCdf: [1e-12, 1e-320],
  normalLogCdf: [1e-12, 1e-300],
  normalQuantile: [1e-13, 1e-300],
  erfinv: [1e-13, 1e-300],
  erfcinv: [1e-13, 1e-300],
  truncatedNormalV: [1e-12, 1e-300],
  truncatedNormalW: [1e-11, 1e-300],
  logGamma: [1e-14, 1],
  gamma: [1e-13, 1e-300],
  digamma: [1e-14, 1],
  trigamma: [1e-13, 1e-300],
  softplus: [1e-15, 1e-300],
  sigmoid: [1e-15, 1e-300],
  logSigmoid: [1e-15, 1e-300],
  logit: [1e-14, 1e-300],
  log1mexp: [1e-14, 1e-300],
  logExpm1: [1e-14, 1e-300],
  log1pmx: [1e-14, 1e-300],
  binaryEntropy: [1e-14, 1e-300],
  besselI0: [1e-13, 1e-300],
  besselI1: [1e-13, 1e-300],
  logBesselI0: [1e-13, 1],
  besselRatio: [1e-13, 1e-300],
  ellipk: [1e-14, 1],
  ellipe: [1e-14, 1],
  xlogy: [1e-15, 1e-300],
  xlog1py: [1e-15, 1e-300],
  regularisedGammaP: [1e-11, 1e-300],
  regularisedGammaQ: [1e-11, 1e-300],
  logBeta: [1e-13, 1],
  logChoose: [1e-13, 1],
  polygamma: [1e-13, 1e-300],
  studentTCdf: [1e-11, 1e-300],
  studentTQuantile: [1e-10, 1e-300],
  chiSquareCdf: [1e-11, 1e-300],
  chiSquareSf: [1e-11, 1e-300],
  normalLogIntervalProbability: [1e-13, 1e-300],
  truncatedNormalVDraw: [1e-10, 1e-300],
  truncatedNormalWDraw: [1e-8, 1e-300],
  logAddExp: [1e-15, 1e-300],
  logDiffExp: [1e-14, 1e-300],
  regularisedBeta: [1e-11, 1e-300],
  regularisedBetaInverse: [1e-11, 1e-300],
  logRegularisedGammaP: [1e-11, 1e-300],
  logRegularisedGammaQ: [1e-11, 1e-300],
  regularisedGammaPInverse: [1e-11, 1e-300],
  regularisedGammaQInverse: [1e-11, 1e-300],
  studentTLogCdf: [1e-11, 1e-300],
  logRegularisedBeta: [1e-11, 1e-300],
}

function worst(errors: number[], inputs: string[]): { max: number; at: string } {
  let max = 0
  let at = ''
  errors.forEach((e, i) => {
    if (e > max || Number.isNaN(e)) {
      max = Number.isNaN(e) ? Infinity : e
      at = inputs[i]
    }
  })
  return { max, at }
}

const call = S as unknown as Record<string, (...args: number[]) => number>

describe('special: unary functions against scipy / mpmath', () => {
  for (const [name, { x, y }] of Object.entries(f.unary)) {
    it(name, () => {
      const [rel, floor] = TOLERANCE[name]
      const w = worst(
        x.map((xi, i) => error(call[name](xi), y[i], floor)),
        x.map((xi) => `x = ${xi}`),
      )
      expect(w.max, `worst at ${w.at}`).toBeLessThanOrEqual(rel)
    })
  }
})

describe('special: binary functions against scipy / mpmath', () => {
  for (const [name, { a, b, y }] of Object.entries(f.binary)) {
    it(name, () => {
      const [rel, floor] = TOLERANCE[name]
      const w = worst(
        a.map((ai, i) => error(call[name](ai, b[i]), y[i], floor)),
        a.map((ai, i) => `(${ai}, ${b[i]})`),
      )
      expect(w.max, `worst at ${w.at}`).toBeLessThanOrEqual(rel)
    })
  }
})

describe('special: ternary functions against scipy', () => {
  for (const [name, { a, b, c, y }] of Object.entries(f.ternary)) {
    it(name, () => {
      const [rel, floor] = TOLERANCE[name]
      const w = worst(
        a.map((ai, i) => error(call[name](ai, b[i], c[i]), y[i], floor)),
        a.map((ai, i) => `(${ai}, ${b[i]}, ${c[i]})`),
      )
      expect(w.max, `worst at ${w.at}`).toBeLessThanOrEqual(rel)
    })
  }
})

describe('special: vector functions against scipy', () => {
  it('logsumexp (aifn-compute/foundation/tensor) against scipy', () => {
    for (const c of f.vector.logSumExp) expect(logsumexp(tensor(c.x))).toBeCloseTo(c.y as number, 12)
    expect(logsumexp(tensor([]))).toBe(-Infinity)
    expect(logsumexp(tensor([-Infinity, -Infinity]))).toBe(-Infinity)
    expect(logsumexp(tensor([1, Infinity]))).toBe(Infinity)
    expect(logsumexp(tensor([1, NaN]))).toBeNaN()
  })
  it('softmax and logSoftmax with temperature', () => {
    for (const c of f.vector.softmax) {
      const got = toFlat(S.softmax(tensor(c.x), { temperature: c.t }))
      ;(c.y as number[]).forEach((yi, i) => expect(error(got[i], yi, 1e-300)).toBeLessThan(1e-12))
    }
    for (const c of f.vector.logSoftmax) {
      const got = toFlat(S.logSoftmax(tensor(c.x), { temperature: c.t }))
      ;(c.y as number[]).forEach((yi, i) => expect(error(got[i], yi, 1)).toBeLessThan(1e-13))
    }
    expect(toFlat(S.softmax(tensor([0, -Infinity])))).toEqual([1, 0])
  })
})

describe('special: edge cases', () => {
  it('softplus does not overflow and matches numpy', () => {
    expect(S.softplus(1000)).toBe(1000)
    expect(S.softplus(-1000)).toBe(0)
    f.softplus.x.forEach((x, i) => expect(S.softplus(x)).toBeCloseTo(f.softplus.y[i], 12))
    expect(S.log1pexp).toBe(S.softplus)
  })
  it('logAddExp and logDiffExp handle −∞', () => {
    expect(S.logAddExp(-Infinity, -Infinity)).toBe(-Infinity)
    expect(S.logDiffExp(3, -Infinity)).toBe(3)
    expect(S.logDiffExp(3, 3)).toBe(-Infinity)
    expect(S.logDiffExp(2, 3)).toBeNaN()
  })
  it('the normal functions are consistent', () => {
    expect(S.normalCdf(0)).toBe(0.5)
    expect(S.normalQuantile(0.5)).toBe(0)
    expect(S.normalQuantile(0)).toBe(-Infinity)
    expect(S.normalQuantile(1)).toBe(Infinity)
    expect(S.normalLogCdf(-1e5)).toBeCloseTo(-5e9 - Math.log(1e5 * Math.sqrt(2 * Math.PI)), 0)
    for (const p of [1e-300, 1e-100, 1e-20, 1e-5, 0.3]) {
      expect(error(S.normalCdf(S.normalQuantile(p)), p, 0)).toBeLessThan(1e-12)
    }
  })
  it('gamma-family special values', () => {
    expect(S.logGamma(1)).toBe(0)
    expect(S.logGamma(2)).toBe(0)
    expect(S.logGamma(0)).toBe(Infinity)
    expect(S.logGamma(-3)).toBe(Infinity)
    expect(S.gamma(5)).toBe(24)
    expect(S.gamma(0.5)).toBeCloseTo(Math.sqrt(Math.PI), 14)
    expect(S.digamma(1)).toBeCloseTo(-0.5772156649015329, 15)
    expect(S.logFactorial(10)).toBeCloseTo(Math.log(3628800), 13)
    expect(S.logChoose(10, 11)).toBe(-Infinity)
    expect(S.logChoose(10, 0)).toBe(0)
  })
  it('binary entropy in bits', () => {
    expect(S.binaryEntropy(0.5, 2)).toBeCloseTo(1, 15)
    expect(S.binaryEntropy(0)).toBe(0)
    expect(S.binaryEntropy(1.2)).toBeNaN()
  })
  it('Student t with ν = ∞ is the normal', () => {
    expect(S.studentTCdf(-3, Infinity)).toBe(S.normalCdf(-3))
    expect(S.studentTQuantile(0.01, Infinity)).toBe(S.normalQuantile(0.01))
  })
  it('Student t far tails beyond t² overflow (where scipy.stats.t.ppf is wrong)', () => {
    // For ν = ½ the tail is ∝ |t|^{−ν}, so p = 1e-100 needs t ≈ −1e199.
    const t = S.studentTQuantile(1e-100, 0.5)
    expect(t).toBeLessThan(-1e198)
    expect(error(S.studentTCdf(t, 0.5), 1e-100, 0)).toBeLessThan(1e-12)
  })
  it('the inverse incomplete beta inverts the incomplete beta', () => {
    for (const [a, b] of [
      [0.1, 0.1],
      [0.5, 3],
      [2, 2],
      [30, 0.7],
      [200, 300],
      [5000, 0.5],
    ]) {
      for (const p of [1e-12, 0.01, 0.4, 0.9, 0.999]) {
        const x = S.regularisedBetaInverse(a, b, p)
        // (0.1, 0.1) at p = 0.999 has its root within 1e-18 of 1, which no double can resolve.
        if (x > 1 - 1e-12) continue
        // The best double x can miss p by the change of I over a few ulps of x (steep near 0 and 1).
        const spread = Math.abs(S.regularisedBeta(a, b, x * (1 + 4e-16)) - S.regularisedBeta(a, b, x * (1 - 4e-16)))
        expect(Math.abs(S.regularisedBeta(a, b, x) - p)).toBeLessThanOrEqual(1e-12 * p + spread)
      }
    }
  })
})

// Derivatives: every kernel's rules against central finite differences, and against scipy where it gives them.

/**
 * Central difference with Richardson extrapolation. The step is 1e-4 of |x| (capped at 1e-4), so it stays inside
 * domains that end near x (e.g. erfcinv at 1e-6, erfinv at −0.99).
 */
function numericDerivative(g: (x: number) => number, x: number): number {
  const h = x === 0 ? 1e-4 : 1e-4 * Math.min(Math.abs(x), 1)
  const d = (s: number) => (g(x + s) - g(x - s)) / (2 * s)
  return (4 * d(h / 2) - d(h)) / 3
}

function expectDerivative(analytic: number, numeric: number, label: string) {
  expect(Math.abs(analytic - numeric) / Math.max(Math.abs(numeric), 1e-6), label).toBeLessThan(1e-6)
}

/** Interior points at which each unary kernel is smooth and finite differences are well conditioned. */
const UNARY_POINTS: Record<string, number[]> = {
  erf: [-2, -0.3, 0.1, 1.2, 3],
  erfc: [-2, 0.1, 1.2, 3, 6],
  erfcx: [-2, 0.1, 1.2, 3, 20],
  logErfc: [-2, 0.1, 1.2, 5, 40],
  erfinv: [-0.99, -0.3, 0.05, 0.7, 0.99],
  erfcinv: [1e-6, 0.3, 1, 1.7, 1.99],
  normalPdf: [-3, -0.5, 0.2, 2],
  normalLogPdf: [-3, 0.2, 2],
  normalCdf: [-8, -1, 0.3, 4],
  normalLogCdf: [-50, -8, -1, 0.3, 6],
  normalQuantile: [1e-10, 0.01, 0.4, 0.9, 0.99],
  truncatedNormalV: [-40, -5, -2.5, 0, 3, 8],
  truncatedNormalW: [-40, -5, -2.5, 0, 3, 8],
  logGamma: [-2.5, 0.3, 1.5, 7.2, 40],
  gamma: [-2.5, 0.3, 1.5, 7.2, 40],
  digamma: [-2.5, 0.3, 1.5, 7.2, 40],
  trigamma: [-2.5, -0.3, 0.3, 1.5, 7.2, 40],
  logFactorial: [0.5, 3, 20, 1000],
  softplus: [-30, -1, 0, 2, 30],
  sigmoid: [-30, -1, 0, 2, 30],
  logSigmoid: [-30, -1, 0, 2, 30],
  logit: [1e-5, 0.2, 0.5, 0.9],
  log1mexp: [-20, -1, -0.1, -1e-3],
  logExpm1: [1e-3, 0.1, 1, 30],
  log1pmx: [-0.9, -0.05, 0.02, 3],
  binaryEntropy: [1e-4, 0.2, 0.5, 0.8],
  besselI0: [-5, -0.3, 0.2, 3, 40],
  besselI1: [-5, -0.3, 0.2, 3, 40],
  logBesselI0: [0.2, 3, 29, 31, 200],
  besselRatio: [0.2, 3, 29, 31, 200],
  ellipk: [-3, -0.5, 0.2, 0.7, 0.95],
  ellipe: [-3, -0.5, 0.2, 0.7, 0.95],
}

const BINARY_POINTS: Record<string, [number, number][]> = {
  logAddExp: [
    [0, 1],
    [-3, 5],
    [700, 699],
  ],
  logDiffExp: [
    [1, 0],
    [5, -3],
    [700, 699.5],
  ],
  logBeta: [
    [0.3, 2],
    [5, 50],
    [300, 20],
  ],
  logChoose: [
    [20, 7],
    [1000, 3.5],
  ],
  polygamma: [
    [1, 0.4],
    [2, 3],
    [3, 30],
  ],
  regularisedGammaP: [
    [0.5, 0.3],
    [3, 2],
    [50, 55],
  ],
  regularisedGammaQ: [
    [0.5, 0.3],
    [3, 2],
    [50, 55],
  ],
  chiSquareCdf: [
    [0.5, 1],
    [3, 4],
    [60, 50],
  ],
  chiSquareSf: [
    [0.5, 1],
    [3, 4],
    [60, 50],
  ],
  studentTCdf: [
    [-3, 1],
    [0.4, 5],
    [2, 30],
  ],
  logRegularisedGammaP: [
    [0.5, 0.3],
    [3, 2],
    [50, 55],
  ],
  logRegularisedGammaQ: [
    [0.5, 0.3],
    [3, 2],
    [50, 55],
  ],
  regularisedGammaPInverse: [
    [0.5, 0.01],
    [3, 0.4],
    [50, 0.99],
  ],
  regularisedGammaQInverse: [
    [0.5, 0.01],
    [3, 0.4],
    [50, 0.99],
  ],
  studentTLogCdf: [
    [-30, 1],
    [0.4, 5],
    [2, 30],
  ],
  studentTQuantile: [
    [0.01, 1],
    [0.4, 5],
    [0.99, 30],
  ],
  normalLogIntervalProbability: [
    [-3, -2],
    [-1, 1],
    [2, 4],
    [-30, -29.9],
  ],
  truncatedNormalVDraw: [
    [-2, 0.5],
    [0.3, 1],
    [5, 0.2],
  ],
  // The forward value loses digits at t = 5, ε = 0.2, where central differences are no longer a check.
  truncatedNormalWDraw: [
    [-2, 0.5],
    [0.3, 1],
    [2.5, 0.2],
  ],
  xlogy: [
    [0.5, 2],
    [-1.5, 0.3],
    [3, 40],
  ],
  xlog1py: [
    [0.5, 2],
    [-1.5, -0.3],
    [3, 40],
  ],
}

// The registered special functions by arity, and which arguments have a derivative (the registry is the one source).
const specials = registry.list().filter((p) => p.module === 'numerics/special')
const byArity = (n: number) => specials.filter((p) => p.arity === n).map((p) => p.name)
const hasDerivative = (name: string, i: number) => {
  const d = registry.get(`numerics/special/${name}`)!.differentiable
  return Array.isArray(d) ? d[i] : d
}

describe('special: derivatives (grad) against finite differences', () => {
  for (const name of byArity(1)) {
    it(`d ${name}`, () => {
      expect(hasDerivative(name, 0), `${name} has a derivative`).toBe(true)
      const points = UNARY_POINTS[name]
      expect(points, `points for ${name}`).toBeDefined()
      for (const x of points) {
        expectDerivative(
          grad(prim[name])(x) as number,
          numericDerivative((v) => call[name](v), x),
          `${name}'(${x})`,
        )
        expect(jvp(prim[name], x, 1).tangent as number).toBeCloseTo(grad(prim[name])(x) as number, 12)
      }
    })
  }
  for (const name of byArity(2)) {
    it(`∂ ${name}`, () => {
      const points = BINARY_POINTS[name]
      expect(points, `points for ${name}`).toBeDefined()
      for (const [a, b] of points) {
        if (hasDerivative(name, 0))
          expectDerivative(
            grad((u: Value) => prim[name](u, b))(a) as number,
            numericDerivative((u) => call[name](u, b), a),
            `∂a ${name}`,
          )
        if (hasDerivative(name, 1))
          expectDerivative(
            grad((u: Value) => prim[name](a, u))(b) as number,
            numericDerivative((u) => call[name](a, u), b),
            `∂b ${name}`,
          )
      }
    })
  }
  it('∂x regularisedBeta and ∂p regularisedBetaInverse', () => {
    for (const [a, b, x] of [
      [0.5, 0.5, 0.3],
      [2, 5, 0.2],
      [40, 30, 0.6],
    ]) {
      expectDerivative(
        grad((u: Value) => S.regularisedBeta(a, b, u))(x) as number,
        numericDerivative((u) => call.regularisedBeta(a, b, u), x),
        'regularisedBeta',
      )
      const y = call.regularisedBeta(a, b, x)
      expectDerivative(
        grad((u: Value) => S.regularisedBetaInverse(a, b, u))(y) as number,
        numericDerivative((u) => call.regularisedBetaInverse(a, b, u), y),
        'inverse',
      )
    }
  })
})

describe('special: derivatives against scipy identities', () => {
  it('d logGamma = digamma, d digamma = trigamma and d trigamma = ψ₂, the last two checked against scipy', () => {
    for (const x of [0.3, 2.5, 40]) {
      expect(grad(S.logGamma)(x)).toBe(S.digamma(x))
      expect(grad(S.digamma)(x)).toBe(S.trigamma(x))
      expect(grad(S.trigamma)(x) as number).toBeCloseTo(S.polygamma(2, x), 10)
    }
  })
})

// Primitives: every export accepts numbers, tensors of any rank and traced values, and broadcasts its arguments.

const UNARY_NAMES = byArity(1)
const BINARY_NAMES = byArity(2)
type Fn = (...args: Value[]) => Value
const prim = S as unknown as Record<string, Fn>

/** Check a tensor result against a list of scalar results, elementwise, with the fixture tolerance. */
function expectElementwise(got: Value, want: number[], shape: number[], label: string) {
  const raw = unwrap(got) as Tensor
  expect(raw.shape, `${label} shape`).toEqual(shape)
  expect(raw.dtype, `${label} dtype`).toBe('float64')
  toFlat(raw).forEach((g, i) => expect(error(g, want[i], 1e-300), `${label}[${i}]`).toBe(0))
}

describe('special: tensors of rank 0–3 match the scalar results elementwise', () => {
  for (const name of UNARY_NAMES) {
    it(name, () => {
      const x = f.unary[name]?.x ?? UNARY_POINTS[name]
      const want = x.map((v) => call[name](v))
      expect(typeof call[name](x[0]), 'a number in gives a number out').toBe('number')
      expectElementwise(prim[name](tensor(x[0])), [want[0]], [], `${name} rank 0`)
      expectElementwise(prim[name](tensor(x)), want, [x.length], `${name} rank 1`)
      // Ranks 2 and 3 on the first 12 (cycled) inputs, and a transposed (strided) view.
      const twelve = Array.from({ length: 12 }, (_, i) => x[i % x.length])
      const want12 = twelve.map((v) => call[name](v))
      expectElementwise(prim[name](tensor(twelve, [3, 4])), want12, [3, 4], `${name} rank 2`)
      expectElementwise(prim[name](tensor(twelve, [2, 3, 2])), want12, [2, 3, 2], `${name} rank 3`)
      const t = transpose(tensor(twelve, [3, 4]))
      expectElementwise(
        prim[name](t),
        toFlat(t).map((v) => call[name](v)),
        [4, 3],
        `${name} transposed`,
      )
    })
  }
  it('int32 tensors give float64 results', () => {
    const n = tensor([0, 1, 5, 10], [4], 'int32')
    expectElementwise(
      S.logFactorial(n),
      [0, 1, 5, 10].map((v) => S.logFactorial(v)),
      [4],
      'logFactorial',
    )
  })
  it('binaryEntropy in any base and softmax along an axis', () => {
    expectElementwise(
      S.binaryEntropy(tensor([0.1, 0.5], [2, 1]), 2),
      [0.1, 0.5].map((p) => S.binaryEntropy(p, 2)),
      [2, 1],
      'H',
    )
    const x = tensor([1, 2, 3, 4, 5, 7], [2, 3])
    const rows = toRows(S.softmax(x))
    expect(rows[1]).toEqual(toFlat(S.softmax(tensor([4, 5, 7]))))
    const columns = toRows(S.softmax(x, { axis: 0, temperature: 2 }))
    const first = toFlat(S.softmax(tensor([1, 4]), { temperature: 2 }))
    expect([columns[0][0], columns[1][0]]).toEqual(first)
    expect(logsumexp(x, 1)).toEqual(tensor([logsumexp(tensor([1, 2, 3])), logsumexp(tensor([4, 5, 7]))]))
    expect(() => S.softmax(3 as unknown as Tensor)).toThrow()
  })
})

describe('special: two- and three-argument functions broadcast', () => {
  for (const name of BINARY_NAMES) {
    it(name, () => {
      const { a, b } = f.binary[name]
      const as = a.slice(0, 3)
      const bs = b.slice(0, 4)
      const want = as.flatMap((ai) => bs.map((bj) => call[name](ai, bj)))
      expectElementwise(prim[name](tensor(as, [3, 1]), tensor(bs, [1, 4])), want, [3, 4], `${name} [3,1]×[1,4]`)
      expectElementwise(
        prim[name](tensor(as), bs[0]),
        as.map((ai) => call[name](ai, bs[0])),
        [3],
        `${name} with a number`,
      )
      expectElementwise(
        prim[name](as[0], tensor(bs)),
        bs.map((bj) => call[name](as[0], bj)),
        [4],
        `${name} number first`,
      )
    })
  }
  for (const name of ['regularisedBeta', 'regularisedBetaInverse'] as const) {
    it(name, () => {
      const [as, bs, cs] = [
        [0.5, 3],
        [2, 0.7, 40],
        [0.2, 0.9],
      ]
      const want = as.flatMap((ai) => bs.flatMap((bj) => cs.map((ck) => call[name](ai, bj, ck))))
      const got = prim[name](tensor(as, [2, 1, 1]), tensor(bs, [1, 3, 1]), tensor(cs, [1, 1, 2]))
      expectElementwise(got, want, [2, 3, 2], name)
      expectElementwise(
        prim[name](2, 5, tensor(cs)),
        cs.map((c) => call[name](2, 5, c)),
        [2],
        `${name} numbers`,
      )
    })
  }
})

/** Moderate interior points (well away from domain edges) for finite-difference gradient checks through the tape. */
const GRADIENT_POINTS: Record<string, number[]> = {
  erf: [-1.2, 0.3, 0.8, 1.5],
  erfc: [-1.2, 0.3, 0.8, 1.5],
  erfcx: [-1.2, 0.3, 0.8, 1.5],
  logErfc: [-1.2, 0.3, 0.8, 1.5],
  erfinv: [-0.6, -0.1, 0.3, 0.7],
  erfcinv: [0.3, 0.8, 1.2, 1.6],
  normalPdf: [-1.2, 0.3, 0.8, 1.5],
  normalLogPdf: [-1.2, 0.3, 0.8, 1.5],
  normalCdf: [-1.2, 0.3, 0.8, 1.5],
  normalLogCdf: [-3, 0.3, 0.8, 1.5],
  normalQuantile: [0.1, 0.3, 0.6, 0.9],
  truncatedNormalV: [-3, 0.3, 0.8, 1.5],
  truncatedNormalW: [-3, 0.3, 0.8, 1.5],
  logGamma: [0.6, 1.3, 2.5, 4],
  gamma: [0.6, 1.3, 2.5, 4],
  digamma: [0.6, 1.3, 2.5, 4],
  trigamma: [0.6, 1.3, 2.5, 4],
  logFactorial: [0.6, 1.3, 2.5, 4],
  softplus: [-1.2, 0.3, 0.8, 1.5],
  sigmoid: [-1.2, 0.3, 0.8, 1.5],
  logSigmoid: [-1.2, 0.3, 0.8, 1.5],
  logit: [0.1, 0.3, 0.6, 0.9],
  log1mexp: [-2, -1, -0.5, -0.2],
  logExpm1: [0.2, 0.5, 1, 2],
  log1pmx: [-0.5, 0.2, 0.8, 1.5],
  binaryEntropy: [0.1, 0.3, 0.6, 0.9],
  besselI0: [-1.2, 0.3, 0.8, 1.5],
  besselI1: [-1.2, 0.3, 0.8, 1.5],
  logBesselI0: [0.3, 0.8, 1.5, 4],
  besselRatio: [0.3, 0.8, 1.5, 4],
  ellipk: [-0.4, 0.1, 0.5, 0.8],
  ellipe: [-0.4, 0.1, 0.5, 0.8],
}

/** Points and the differentiable arguments of each binary primitive, as ([a…], [b…], wrt). */
const BINARY_GRADIENT: Record<string, [number[], number[], number[]]> = {
  logAddExp: [
    [0.2, -1, 1.5],
    [0.4, 1.1],
    [0, 1],
  ],
  logDiffExp: [
    [2, 3, 2.5],
    [0.4, 1.1],
    [0, 1],
  ],
  logBeta: [
    [0.7, 2, 5],
    [1.5, 3],
    [0, 1],
  ],
  logChoose: [
    [9, 12, 20],
    [2.5, 4],
    [0, 1],
  ],
  polygamma: [[1, 2, 3], [0.8, 2.5], [1]],
  regularisedGammaP: [[0.8, 2, 5], [1.5, 3], [1]],
  regularisedGammaQ: [[0.8, 2, 5], [1.5, 3], [1]],
  logRegularisedGammaP: [[0.8, 2, 5], [1.5, 3], [1]],
  logRegularisedGammaQ: [[0.8, 2, 5], [1.5, 3], [1]],
  regularisedGammaPInverse: [[0.8, 2, 5], [0.1, 0.7], [1]],
  regularisedGammaQInverse: [[0.8, 2, 5], [0.1, 0.7], [1]],
  studentTLogCdf: [[-1, 0.3, 2], [1.5, 6], [0]],
  chiSquareCdf: [[0.8, 2, 5], [1.5, 3], [0]],
  chiSquareSf: [[0.8, 2, 5], [1.5, 3], [0]],
  studentTCdf: [[-1, 0.3, 2], [1.5, 6], [0]],
  studentTQuantile: [[0.1, 0.4, 0.8], [1.5, 6], [0]],
  normalLogIntervalProbability: [
    [-2, -1, 0.1],
    [0.5, 1.4],
    [0, 1],
  ],
  truncatedNormalVDraw: [[-1, 0.3, 2], [0.5, 1], [0]],
  truncatedNormalWDraw: [[-1, 0.3, 2], [0.5, 1], [0]],
  xlogy: [
    [0.5, -1, 2],
    [0.4, 1.1],
    [0, 1],
  ],
  xlog1py: [
    [0.5, -1, 2],
    [0.4, 1.1],
    [0, 1],
  ],
}

describe('special: gradients through the tape against finite differences', () => {
  for (const name of UNARY_NAMES) {
    it(`∇ ${name} on a [2, 2] tensor`, () => {
      checkGradient((x) => prim[name](x), [tensor(GRADIENT_POINTS[name], [2, 2])], { eps: 1e-5, tol: 1e-6 })
      checkGradient((x) => prim[name](x), [GRADIENT_POINTS[name][1]], { eps: 1e-5, tol: 1e-6 })
    })
  }
  for (const [name, [as, bs, wrt]] of Object.entries(BINARY_GRADIENT)) {
    it(`∇ ${name} with broadcasting [3, 1] × [2]`, () => {
      // Arguments without a derivative are constants (tracing them would make differentiation an error).
      const check = (a: Raw, b: Raw) => {
        if (wrt.length === 2) checkGradient((u, v) => prim[name](u, v), [a, b], { eps: 1e-5, tol: 1e-6 })
        else if (wrt[0] === 0) checkGradient((u) => prim[name](u, b), [a], { eps: 1e-5, tol: 1e-6 })
        else checkGradient((v) => prim[name](a, v), [b], { eps: 1e-5, tol: 1e-6 })
      }
      check(tensor(as, [3, 1]), tensor(bs))
      check(as[0], tensor(bs))
      check(tensor(as), bs[1])
    })
  }
  it('∇ regularisedBeta and regularisedBetaInverse in their last argument, broadcast', () => {
    for (const name of ['regularisedBeta', 'regularisedBetaInverse']) {
      const a = tensor([0.8, 3], [2, 1])
      checkGradient((c) => prim[name](a, 2.5, c), [tensor([0.2, 0.5, 0.7])], { eps: 1e-5, tol: 1e-6 })
    }
  })
  it('∇ logSumExp, softmax and logSoftmax along axes, with temperature', () => {
    const x = tensor([0.3, -1.2, 2.5, 0.9, 0.1, -0.4], [2, 3])
    checkGradient((v) => logsumexp(v), [x])
    checkGradient((v) => logsumexp(v, 1), [x])
    for (const options of [{}, { axis: 0 }, { temperature: 0.5 }, { axis: 0, temperature: 3 }]) {
      checkGradient((v) => S.softmax(v, options), [x])
      checkGradient((v) => S.logSoftmax(v, options), [x])
    }
    checkGradient((p) => S.binaryEntropy(p, 2), [tensor([0.2, 0.7])])
  })
})

/** f' and f'' at x in argument `wrt`, by nested grad (every derivative is itself a primitive). */
function secondDerivative(fn: Fn, args: number[], wrt: number): [number, number] {
  const at = (v: Value) => fn(...args.map((a, i) => (i === wrt ? v : a)))
  const d1 = (v: Value) => grad(at)(v) as Value
  return [unwrap(d1(args[wrt])) as number, unwrap(grad(d1)(args[wrt])) as number]
}

describe('special: second derivatives where the derivative is itself a primitive', () => {
  const unaryCases: [string, number][] = [
    ['logGamma', 2.5],
    ['gamma', 1.7],
    ['digamma', 0.8],
    ['erf', 0.4],
    ['erfc', 0.4],
    ['erfcx', 0.9],
    ['logErfc', 0.9],
    ['erfinv', 0.3],
    ['erfcinv', 0.7],
    ['normalPdf', 0.6],
    ['normalLogPdf', 0.6],
    ['normalCdf', -0.6],
    ['normalLogCdf', -1.5],
    ['normalQuantile', 0.3],
    ['truncatedNormalV', -0.7],
    ['truncatedNormalW', 1.2],
    ['logFactorial', 3.5],
    ['softplus', 0.7],
    ['sigmoid', 0.7],
    ['logSigmoid', -0.7],
    ['logit', 0.3],
    ['log1mexp', -0.8],
    ['logExpm1', 0.8],
    ['log1pmx', 0.4],
    ['binaryEntropy', 0.3],
  ]
  for (const [name, x] of unaryCases) {
    it(`d² ${name}`, () => {
      const [, d2] = secondDerivative(prim[name], [x], 0)
      expectDerivative(
        d2,
        numericDerivative((v) => grad(prim[name])(v) as number, x),
        `${name}''(${x})`,
      )
    })
  }
  const binaryCases: [string, [number, number], number][] = [
    ['logAddExp', [0.3, 1.1], 0],
    ['logAddExp', [0.3, 1.1], 1],
    ['logDiffExp', [2, 0.5], 0],
    ['logDiffExp', [2, 0.5], 1],
    ['logBeta', [1.5, 3], 0],
    ['logBeta', [1.5, 3], 1],
    ['logChoose', [12, 4.5], 0],
    ['logChoose', [12, 4.5], 1],
    ['polygamma', [1, 1.5], 1],
    ['normalLogIntervalProbability', [-1, 0.5], 0],
    ['normalLogIntervalProbability', [-1, 0.5], 1],
  ]
  for (const [name, [a, b], wrt] of binaryCases) {
    it(`∂² ${name} in argument ${wrt + 1}`, () => {
      const [, d2] = secondDerivative(prim[name], [a, b], wrt)
      const first = (v: number) => grad((u: Value) => (wrt === 0 ? prim[name](u, b) : prim[name](a, u)))(v) as number
      expectDerivative(d2, numericDerivative(first, wrt === 0 ? a : b), `${name}`)
    })
  }
  it('softmax has second derivatives (a composition of primitives)', () => {
    const g = (v: Value) => grad((u: Value) => get(S.softmax(u), 0))(v) as Value
    const h = grad((v: Value) => get(g(v), 0))(tensor([0.3, -1.2, 2.5]))
    // ∂²p₀/∂x₀² = p₀(1 − p₀)(1 − 2p₀).
    const p0 = toFlat(S.softmax(tensor([0.3, -1.2, 2.5])))[0]
    expect(toFlat(unwrap(h) as Tensor)[0]).toBeCloseTo(p0 * (1 - p0) * (1 - 2 * p0), 12)
  })
})

describe('special: unavailable derivatives are errors, not zeros', () => {
  const differentiate = (fn: Fn, args: number[], wrt: number) =>
    grad((v: Value) => fn(...args.map((a, i) => (i === wrt ? v : a))))(args[wrt])
  it('in shape, order and degrees-of-freedom arguments', () => {
    expect(() => differentiate(prim.polygamma, [1, 2], 0)).toThrow(NotDifferentiableError)
    expect(() => differentiate(prim.regularisedGammaP, [2, 1], 0)).toThrow(NotDifferentiableError)
    expect(() => differentiate(prim.studentTCdf, [0.3, 4], 1)).toThrow(NotDifferentiableError)
    expect(() => differentiate(prim.chiSquareSf, [2, 3], 1)).toThrow(NotDifferentiableError)
    expect(() => differentiate(prim.truncatedNormalVDraw, [0.3, 1], 1)).toThrow(NotDifferentiableError)
    expect(() => differentiate(prim.truncatedNormalWDraw, [0.3, 1], 1)).toThrow(NotDifferentiableError)
    expect(() => differentiate(prim.regularisedBeta, [2, 3, 0.4], 0)).toThrow(NotDifferentiableError)
    expect(() => differentiate(prim.regularisedBetaInverse, [2, 3, 0.4], 1)).toThrow(NotDifferentiableError)
  })
  it('every other derivative is itself differentiable (second derivatives of the former first-order rules)', () => {
    for (const [fn, args, wrt] of [
      [prim.trigamma, [1.5], 0],
      [prim.regularisedGammaP, [2, 1], 1],
      [prim.regularisedBeta, [2, 3, 0.4], 2],
      [prim.besselI1, [0.7], 0],
      [prim.studentTCdf, [0.3, 4], 0],
    ] as [Fn, number[], number][]) {
      const [, d2] = secondDerivative(fn, args, wrt)
      const first = (v: number) => grad((u: Value) => fn(...args.map((a, i) => (i === wrt ? u : a))))(v) as number
      expectDerivative(d2, numericDerivative(first, args[wrt]), 'second derivative')
    }
  })
})

describe('special: log tails and inverses of the incomplete gamma function', () => {
  // scipy.special.gammaincinv / gammainccinv at p = 1e-300, 1e-12, 0.5, 1 − 1e-12 (the last two by symmetry of the
  // pair: P⁻¹(a, 1 − 1e-12) is Q⁻¹(a, 1e-12) up to the rounding of 1 − 1e-12).
  const CASES: [number, number, number, number][] = [
    // a, p, P⁻¹(a, p), Q⁻¹(a, p)
    [2.7, 1e-300, 1.3139960164441055e-111, 701.4835659623169],
    [2.7, 1e-12, 6.0991297752091184e-5, 33.20156775905441],
    [2.7, 0.999999999999, 33.20159103738972, 6.0990798024662964e-5],
    [0.4, 1e-100, 7.415354682563958e-251, 226.20632559340964],
    [50, 0.01, 35.03244746269989, 67.90336158551338],
    [1e4, 0.999999, 10482.561164638257, 9531.835117190369],
  ]
  it('P⁻¹ and Q⁻¹ match scipy to 1e-12 relative, both tails', () => {
    for (const [a, p, pi, qi] of CASES) {
      const P = S.regularisedGammaPInverse(a, p) as number
      const Q = S.regularisedGammaQInverse(a, p) as number
      expect(Math.abs(P - pi), `P⁻¹(${a}, ${p})`).toBeLessThanOrEqual(1e-12 * pi)
      expect(Math.abs(Q - qi), `Q⁻¹(${a}, ${p})`).toBeLessThanOrEqual(1e-12 * qi)
    }
  })
  it('log P and log Q do not underflow and keep relative accuracy near 0 (mpmath)', () => {
    // log Q(3.5, 750) (χ²₇ survival at 1500) and log P(2.7, 1.8e-30) ≈ −1.17e-81 as log Q.
    expect(Math.abs((S.logRegularisedGammaQ(3.5, 750) as number) + 734.6474561470056)).toBeLessThan(1e-12 * 735)
    const q = S.logRegularisedGammaQ(2.7, 1.8e-30) as number
    expect(Math.abs(q + 1.1722812872809138e-81)).toBeLessThan(1e-12 * 1.2e-81)
  })
  it('the Student t log cdf is finite where the cdf underflows (mpmath: −1611.233…)', () => {
    const v = S.studentTLogCdf(-1e200, 3.5) as unknown as number
    expect(Number.isFinite(v)).toBe(true)
    expect(Math.abs(v + 1611.2330849764724)).toBeLessThan(1e-9)
  })
})

describe('special: review regressions (relative accuracy near 0, underflowing inverses)', () => {
  // References from mpmath at 50 digits.
  it('logErfc keeps its relative accuracy near 0 (was log(1 − erf x), 5e-5 relative error at 1e-12)', () => {
    expect(Math.abs(S.logErfc(1e-12) / -1.1283791670961491e-12 - 1)).toBeLessThan(1e-13)
    // log erfc(−x) = log(1 + erf x) = 2x/√π − (2x/√π)²/2 + …
    expect(Math.abs(S.logErfc(-1e-12) / 1.128379167094876e-12 - 1)).toBeLessThan(1e-13)
  })

  it('logBesselI0 keeps its relative accuracy near 0 (was x + log of the scaled value, 1e-6 relative at 1e-5)', () => {
    expect(Math.abs(S.logBesselI0(1e-5) / 2.499999999984375e-11 - 1)).toBeLessThan(1e-13)
    expect(Math.abs(S.logBesselI0(0.5) / 0.06154971918548131 - 1)).toBeLessThan(1e-13)
  })

  it('normalLogIntervalProbability is accurate on a narrow interval about 0 (was 1.5e-9 relative)', () => {
    // log(Φ(1e-9) − Φ(−1e-9)) = log(2e-9 φ(0)) to O(1e-18).
    const ref = Math.log(2e-9 / Math.sqrt(2 * Math.PI))
    expect(Math.abs(S.normalLogIntervalProbability(-1e-9, 1e-9) / ref - 1)).toBeLessThan(1e-14)
  })

  it('regularisedBetaInverse gives 0 or 1, not NaN, where the root underflows', () => {
    expect(S.regularisedBetaInverse(0.001, 30, 0.01)).toBe(0)
    expect(S.regularisedBetaInverse(0.5, 0.5, 1e-200)).toBe(0)
    expect(S.regularisedBetaInverse(1, 0.001, 0.99)).toBe(1)
  })
})
