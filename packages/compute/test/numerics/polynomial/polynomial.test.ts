/** Polynomials in descending powers (numpy's convention), real or complex128. */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import {
  companionMatrix,
  polyDerivative,
  polyDivide,
  polyFromRoots,
  polyMul,
  polyval,
  residue,
  residuez,
  roots,
  polynomialRoots,
} from 'aifn-compute/numerics/polynomial'
import { eig } from 'aifn-compute/numerics/linalg'
import {
  complex,
  realPart,
  tensor,
  toComplexFlat,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { NumericalError } from 'aifn-compute/foundation/errors'

/** Complex values sorted by real part then imaginary part, descending, as [re, im] pairs rounded for comparison. */
const sorted = (t: Tensor, digits = 9) =>
  toComplexFlat(t)
    .map(({ re, im }) => [+re.toFixed(digits) + 0, +im.toFixed(digits) + 0])
    .sort((a, b) => b[0] - a[0] || b[1] - a[1])

describe('roots and the companion matrix', () => {
  it('real and complex roots of (x − 1)(x − 2)(x − 3)(x² + 1)', () => {
    const r = roots([1, -6, 12, -12, 11, -6])
    expect(r.dtype).toBe('complex128')
    expect(sorted(r)).toEqual([
      [3, 0],
      [2, 0],
      [1, 0],
      [0, 1],
      [0, -1],
    ])
    // Leading zeros are dropped; zero roots come from trailing zeros.
    expect(sorted(roots([0, 2, -2, 0]))).toEqual([
      [1, 0],
      [0, 0],
    ])
    expect(sorted(eig(companionMatrix([1, -6, 12, -12, 11, -6])).values)).toEqual(sorted(r))
  })
  it('a degree-10 Wilkinson-type polynomial to modest accuracy', () => {
    const p = polyFromRoots([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sorted(roots(toFlat(p)), 5).forEach(([re, im], i) => {
      expect(re).toBeCloseTo(10 - i, 5)
      expect(im).toBeCloseTo(0, 5)
    })
  })
})

describe('evaluation and arithmetic', () => {
  it('polyval: numbers, tensors, complex points and derivatives', () => {
    expect(polyval([1, -6, 12, -12, 11, -6], 3)).toBeCloseTo(0, 12)
    expect(toFlat(polyval([1, 0, -1], tensor([2, 3])))).toEqual([3, 8])
    // p(i) for p = x² + 1 is 0.
    const z = polyval([1, 0, 1], complex(0, 1) as unknown as Tensor) as Tensor
    expect(toComplexFlat(z)[0]).toEqual({ re: 0, im: 0 })
    expect(grad((x: Value) => polyval([1, 0, -1], x))(3)).toBe(6)
    expect(toFlat(polyDerivative([1, 0, -1]) as Tensor)).toEqual([2, 0])
    expect(toFlat(polyDerivative([4, 3, 2, 1], 2) as Tensor)).toEqual([24, 6])
  })
  it('polyMul, polyDivide and polyFromRoots invert one another', () => {
    const a = [1, -3, 2]
    const b = [2, 1]
    const ab = toFlat(polyMul(a, b) as Tensor)
    expect(ab).toEqual([2, -5, 1, 2])
    const { quotient, remainder } = polyDivide(ab, b)
    toFlat(quotient).forEach((v, k) => expect(v).toBeCloseTo(a[k], 12))
    expect(toFlat(remainder).every((v) => Math.abs(v) < 1e-12)).toBe(true)
    expect(toFlat(polyFromRoots([1, 2]))).toEqual(a)
    // Conjugate pairs give real coefficients with `real`.
    const q = polyFromRoots(
      [
        { re: 0, im: 1 },
        { re: 0, im: -1 },
      ],
      { real: true },
    )
    expect(q.dtype).toBe('float64')
    expect(toFlat(q)).toEqual([1, 0, 1])
  })
})

describe('partial fractions', () => {
  it('residue: 1/((s + 1)(s + 2)) = 1/(s + 1) − 1/(s + 2)', () => {
    const { residues, poles, direct } = residue([1], [1, 3, 2])
    const pr = toComplexFlat(poles).map((p, k) => [p.re, toComplexFlat(residues)[k].re])
    pr.sort((x, y) => y[0] - x[0])
    expect(pr[0][0]).toBeCloseTo(-1, 12)
    expect(pr[0][1]).toBeCloseTo(1, 12)
    expect(pr[1][0]).toBeCloseTo(-2, 12)
    expect(pr[1][1]).toBeCloseTo(-1, 12)
    expect(toFlat(direct)).toEqual([])
  })
  it('residue with a repeated pole and a direct term: (s² + 1)/(s + 1)² = 1 − 2/(s + 1) + 2/(s + 1)²', () => {
    const { residues, poles, direct } = residue([1, 0, 1], [1, 2, 1])
    expect(toComplexFlat(poles).map((p) => +p.re.toFixed(9))).toEqual([-1, -1])
    expect(toComplexFlat(residues).map((r) => +r.re.toFixed(9))).toEqual([-2, 2])
    expect(toFlat(direct)).toEqual([1])
  })
  it('residuez: 1/(1 − 0.5z⁻¹)(1 − 0.25z⁻¹) = 2/(1 − 0.5z⁻¹) − 1/(1 − 0.25z⁻¹)', () => {
    const { residues, poles } = residuez([1], [1, -0.75, 0.125])
    const pr = toComplexFlat(poles).map((p, k) => [p.re, toComplexFlat(residues)[k].re])
    pr.sort((x, y) => y[0] - x[0])
    expect(pr[0][0]).toBeCloseTo(0.5, 12)
    expect(pr[0][1]).toBeCloseTo(2, 12)
    expect(pr[1][0]).toBeCloseTo(0.25, 12)
    expect(pr[1][1]).toBeCloseTo(-1, 12)
  })
})

describe('polynomialRoots: roots with a convergence flag', () => {
  it('agrees with roots and reports convergence', () => {
    const r = polynomialRoots([1, -6, 11, -6, 0])
    expect(r.converged).toBe(true)
    expect(toFlat(realPart(r.roots))).toEqual(toFlat(realPart(roots([1, -6, 11, -6, 0]))))
  })
  it("Wilkinson's polynomial: roots throws exactly when the flag is false", () => {
    const w = polyFromRoots(
      Array.from({ length: 20 }, (_, k) => k + 1),
      { real: true },
    )
    const r = polynomialRoots(w)
    expect(r.roots.shape).toEqual([20])
    if (r.converged) expect(() => roots(w)).not.toThrow()
    else expect(() => roots(w)).toThrow(NumericalError)
  })
})
