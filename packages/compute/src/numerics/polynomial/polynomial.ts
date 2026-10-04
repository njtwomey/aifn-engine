/**
 * Polynomials with coefficients in descending powers, $p(x) = c_0 x^n + c_1 x^{n-1} + \dots + c_n$, as NumPy's `poly*` functions and
 * `scipy.signal` read them. Real or complex (complex128) coefficients and arguments.
 *
 * - `polyval` (Horner's rule), `polyDerivative` and `polyMul` are compositions of primitives, so they accept traced
 *   values and are differentiable; `polyMul` is the FIR `linearFilter`, i.e. the convolution of the coefficients.
 * - `roots` are the eigenvalues of the companion matrix (Edelman & Murakami, 1995, "Polynomial roots from companion
 *   matrix eigenvalues", Math. Comp. 64(210)), from `aifn-compute/numerics/linalg`'s `eig` (balancing and the EISPACK `hqr`
 *   shifted QR algorithm, the one copy of it). The companion matrix is already upper Hessenberg.
 * - `polyFromRoots` (numpy's `poly`), `polyDivide` (`polydiv`, deconvolution) and the partial-fraction expansions
 *   `residue` and `residuez` (scipy.signal's, including the grouping of repeated poles) run element by element.
 */

import {
  add,
  concat,
  fromData,
  isTensor,
  isTraced,
  mul,
  shapeOfValue,
  slice,
  tensor,
  zeros,
  type Tensor,
  type Traced,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { ComplexNumber, Scalar, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, DTypeError, NumericalError } from 'aifn-compute/foundation/errors'
import { linearFilter } from 'aifn-compute/foundation/convolution'
import { eig } from 'aifn-compute/numerics/linalg'
import * as cx from './scalar'

/**
 * A complex vector as accepted by the polynomial and systems functions: a complex128 or real tensor, a list of
 * `{ re, im }`, or real numbers.
 */
export type ComplexLike = Tensor | readonly (ComplexNumber | number)[] | VectorLike

/**
 * Convert a `ComplexLike` input into a rank-1 complex128 tensor (a copy).
 *
 * @param x The complex vector input: a tensor, array of numbers, or array of complex numbers.
 * @param where The caller name for error messages (default `'complexVector'`).
 * @returns A rank-1 complex128 tensor.
 *
 * @example Convert complex number representations to a tensor
 * const v = complexVector([{ re: 1, im: 2 }, { re: 3, im: 4 }])
 * print('shape =', v.shape)
 */
export function complexVector(x: ComplexLike, where = 'complexVector'): Tensor {
  return cx.toTensor(cx.readList(x as Value | VectorLike, where).list, true)
}

/**
 * Standardise coefficients as a rank-1 value: tensors and traced values as given, plain arrays as float64 tensors.
 *
 * @param p The polynomial coefficients as a vector, tensor, or scalar.
 * @param where The caller name for error messages.
 * @returns A rank-1 tensor or traced value holding the coefficients.
 */
function coefficientValue(p: Value | VectorLike, where: string): Value {
  const v: Value =
    isTraced(p) || isTensor(p)
      ? (p as Value)
      : typeof p === 'number'
        ? tensor([p])
        : tensor(Array.from(p as ArrayLike<number>))
  if (shapeOfValue(v).length !== 1) throw new DomainError(where, `${where}: coefficients must be a vector`)
  return v
}

// ── Values and calculus (differentiable compositions) ──────────────────────────────────────────────────────────────

/**
 * Evaluate polynomial $p(x)$ by Horner's rule, elementwise in $x$, coefficients highest power first (numpy's `polyval`).
 * Real coefficients (a plain array) at a number give a number; a tensor $x$ (real or complex) gives a tensor of
 * $x$'s shape; traced coefficients or $x$ give a traced value, differentiable in both.
 *
 * @param p The polynomial coefficients in descending degree order, $[c_d, \dots, c_0]$ representing
 *   $\sum_{k=0}^d c_{d-k} x^k$.
 * @param x The evaluation point or tensor of points.
 * @returns The evaluated polynomial value or tensor matching $x$'s shape.
 *
 * @example Evaluate a quadratic polynomial
 * print('p(2) =', polyval([1, 0, -1], 2))
 */
export function polyval(p: VectorLike, x: Scalar): Scalar
export function polyval(p: Value | VectorLike, x: Tensor): Tensor
export function polyval(p: Value | VectorLike, x: Traced): Traced
export function polyval(p: Value | VectorLike, x: Value): Value
export function polyval(p: Value | VectorLike, x: Value): Value {
  let v: Value = mul(0, x)
  if (!isTraced(p) && !isTensor(p)) {
    const c = typeof p === 'number' ? [p] : Array.from(p as ArrayLike<number>)
    for (const ck of c) v = add(mul(v, x), ck)
    return v
  }
  const pv = coefficientValue(p, 'polyval')
  const n = shapeOfValue(pv)[0]
  for (let k = 0; k < n; k++) v = add(mul(v, x), slice(pv, k))
  return v
}

/**
 * Compute the $m$-th derivative of polynomial $p$, with coefficients in descending powers (default $m = 1$). Returns
 * `[0]` when the polynomial degree is below $m$. Differentiable.
 *
 * @param p The polynomial coefficients in descending degree order.
 * @param m The order of derivative to compute (default 1).
 * @returns The derivative polynomial's coefficients in descending degree order.
 *
 * @example Differentiate a polynomial
 * print("p'(x) =", polyDerivative([3, 2, 1]))
 */
export function polyDerivative(p: Value | VectorLike, m = 1): Value {
  let v = coefficientValue(p, 'polyDerivative')
  for (let j = 0; j < m; j++) {
    const n = shapeOfValue(v)[0]
    if (n <= 1) return zeros([1])
    v = mul(slice(v, [0, n - 1]), tensor(Array.from({ length: n - 1 }, (_, k) => n - 1 - k)))
  }
  return v
}

/**
 * Compute the polynomial product $a(x) \cdot b(x)$ via full linear convolution of their coefficients (the FIR
 * `linearFilter` of $a$ padded). Differentiable.
 *
 * @param a The first polynomial coefficients in descending degree order.
 * @param b The second polynomial coefficients in descending degree order.
 * @returns The product polynomial's coefficients in descending degree order.
 *
 * @example Multiply two polynomials
 * print('(x + 1)(x - 1) =', polyMul([1, 1], [1, -1]))
 */
export function polyMul(a: Value | VectorLike, b: Value | VectorLike): Value {
  const av = coefficientValue(a, 'polyMul')
  const bv = coefficientValue(b, 'polyMul')
  const nb = shapeOfValue(bv)[0]
  const padded = nb > 1 ? concat([av, zeros([nb - 1])], 0) : av
  return linearFilter(bv, [1], padded)
}

// ── Roots and products of linear factors ──────────────────────────────────────────────────────────────────────────────

/**
 * The companion matrix of polynomial $p$ (first row $-c_k/c_0$, ones on the subdiagonal), leading zeros stripped;
 * real $p$ only.
 *
 * @param p The real polynomial coefficients in descending degree order.
 * @returns An $n \times n$ matrix whose eigenvalues are the roots of $p$.
 *
 * @example Form the companion matrix of a polynomial
 * print('C =', companionMatrix([1, -3, 2]))
 */
export function companionMatrix(p: VectorLike): Tensor {
  const { list, complex } = cx.readList(p, 'companionMatrix')
  if (complex) throw new DTypeError('companionMatrix', 'companionMatrix: real coefficients only', ['complex128'])
  const c = list.map((z) => z.re)
  while (c.length && c[0] === 0) c.shift()
  const n = Math.max(c.length - 1, 0)
  const a = new Float64Array(n * n)
  for (let j = 0; j < n; j++) a[j] = -c[j + 1] / c[0]
  for (let i = 1; i < n; i++) a[i * n + i - 1] = 1
  return fromData(a, [n, n])
}

/** The roots of a polynomial with whether the eigenvalue iteration that found them converged (`polynomialRoots`). */
export type PolynomialRoots = {
  /** The roots, complex128, sorted as `roots` sorts them; unreliable when `converged` is false. */
  roots: Tensor
  /** False when the QR iteration on the companion matrix did not converge. */
  converged: boolean
}

/**
 * The roots of $p$ (real coefficients, highest power first) and a convergence flag: the eigenvalues of the companion
 * matrix, as numpy's `roots`. Leading zeros are stripped; trailing zeros give roots at 0. Sorted by real part, then
 * imaginary part, both descending (conjugate pairs adjacent). Accuracy degrades for clustered or multiple roots (a
 * double root is found to about $\sqrt{\varepsilon}$). A QR iteration that does not converge is reported in `converged`, with the
 * eigenvalues it reached, so a figure can show the failure (Wilkinson's polynomial); `roots` throws instead.
 *
 * @param p The real polynomial coefficients in descending degree order.
 * @returns An object containing the complex roots as a tensor and a `converged` boolean flag.
 *
 * @example Find roots of a polynomial with convergence reporting
 * const res = polynomialRoots([1, -3, 2])
 * print('converged =', res.converged)
 */
export function polynomialRoots(p: VectorLike): PolynomialRoots {
  const { list, complex } = cx.readList(p, 'roots')
  if (complex)
    throw new DTypeError('roots', 'roots: real coefficients only (the companion eigenproblem is real)', ['complex128'])
  const c = list.map((z) => z.re)
  if (!c.every(Number.isFinite)) throw new DomainError('roots', 'roots: coefficients must be finite')
  while (c.length && c[0] === 0) c.shift()
  let trailing = 0
  while (c.length && c[c.length - 1] === 0) {
    c.pop()
    trailing++
  }
  const found: cx.C[] = []
  let converged = true
  if (c.length > 1) {
    const e = eig(companionMatrix(c), { vectors: false })
    converged = e.converged
    found.push(...cx.readList(e.values, 'roots').list)
  }
  for (let k = 0; k < trailing; k++) found.push(cx.of(0))
  found.sort((u, v) => v.re - u.re || v.im - u.im)
  return { roots: cx.toTensor(found, true), converged }
}

/**
 * The roots of $p$ (real coefficients, highest power first) as a complex128 vector: `polynomialRoots` without the flag.
 * Throws `NumericalError('not-converged')` when the QR iteration does not converge.
 *
 * @param p The real polynomial coefficients in descending degree order.
 * @returns A complex128 tensor holding the roots of the polynomial.
 *
 * @example Find the roots of a polynomial
 * print('roots =', roots([1, 0, -4]))
 */
export function roots(p: VectorLike): Tensor {
  const r = polynomialRoots(p)
  if (!r.converged) throw new NumericalError('roots', 'roots: the QR iteration did not converge', 'not-converged')
  return r.roots
}

/**
 * The coefficients (highest power first, leading 1) of $\prod (x - r_k)$, as numpy's `poly`. Real roots give float64.
 * Complex roots give complex128, unless they are closed under conjugation (to 1e-9 relative), when the imaginary parts
 * of the coefficients are rounding and float64 is returned. `{ real: true }` requires the real result and throws
 * otherwise.
 *
 * @param r The roots of the polynomial: a complex tensor or list of numbers/complex numbers.
 * @param options Options controlling output representation.
 * @param options.real Require real coefficients and throw if complex roots are not conjugate pairs (default false).
 * @returns A rank-1 tensor holding the monic polynomial coefficients.
 *
 * @example Reconstruct polynomial coefficients from roots
 * print('coeffs =', polyFromRoots([2, -2]))
 */
export function polyFromRoots(r: ComplexLike, { real = false }: { real?: boolean } = {}): Tensor {
  const { list, complex } = cx.readList(r as Value | VectorLike, 'polyFromRoots')
  let p: cx.C[] = [cx.of(1)]
  for (const z of list) p = cx.product(p, [cx.of(1), cx.of(-z.re, -z.im)])
  if (!complex) return cx.toTensor(p, false)
  const size = Math.max(1, ...p.map(cx.abs))
  const isReal = p.every((c) => Math.abs(c.im) <= 1e-9 * size)
  if (real && !isReal)
    throw new DomainError(
      'polyFromRoots',
      'polyFromRoots: complex roots must come in conjugate pairs for real coefficients',
    )
  return cx.toTensor(p, !isReal)
}

/**
 * Polynomial division $u = q \cdot v + r$ with $\deg r < \deg v$ (numpy's `polydiv`; deconvolution of coefficient
 * sequences). The remainder's leading zeros (relative 1e-14) are dropped, leaving at least one coefficient. Complex
 * when either input is. Not differentiable (element by element).
 *
 * @param u The dividend polynomial coefficients in descending degree order.
 * @param v The divisor polynomial coefficients in descending degree order.
 * @returns An object containing quotient tensor `quotient` and remainder tensor `remainder`.
 *
 * @example Divide polynomials with remainder
 * const { quotient, remainder } = polyDivide([1, 0, -1], [1, -1])
 * print('quotient =', quotient)
 */
export function polyDivide(u: ComplexLike, v: ComplexLike): { quotient: Tensor; remainder: Tensor } {
  const a = cx.readList(u as Value | VectorLike, 'polyDivide')
  const b = cx.readList(v as Value | VectorLike, 'polyDivide')
  const { q, r } = cx.divide(a.list, b.list)
  const complex = a.complex || b.complex
  return { quotient: cx.toTensor(q, complex), remainder: cx.toTensor(r, complex) }
}

// ── Partial fractions ────────────────────────────────────────────────────────────────────────────────────────────

/** A partial-fraction expansion. */
export type PartialFractions = {
  /** Residues, complex128, grouped by pole; a pole of multiplicity $m$ has $m$ residues, for powers $1 \dots m$. */
  residues: Tensor
  /** Poles, complex128, each repeated by its multiplicity, groups ordered by modulus. */
  poles: Tensor
  /** The direct (polynomial) term: descending powers of $s$ for `residue`, ascending powers of $z^{-1}$ for `residuez`. */
  direct: Tensor
}

/** Options for `residue` and `residuez`. */
export type ResidueOptions = {
  /**
   * Poles closer than this (absolute distance) are one repeated pole, replaced by their mean (scipy's `tol`, default
   * 1e-3). Computed roots of a pole of multiplicity $m$ scatter by about $\varepsilon^{1/m}$ relative, so the tolerance must exceed
   * that scatter and stay below the gap between distinct poles.
   */
  tolerance?: number
}

/**
 * Group roots within distance `tol` of each other into repeated poles (scipy's `unique_roots`, `rtype='avg'`),
 * ordered by modulus.
 *
 * @param ps The list of pole locations.
 * @param tol The distance threshold below which roots are merged into a repeated pole.
 * @returns An array of grouped poles with their multiplicities.
 */
function groupRoots(ps: readonly cx.C[], tol: number): { pole: cx.C; mult: number }[] {
  const used = ps.map(() => false)
  const groups: { pole: cx.C; mult: number }[] = []
  ps.forEach((p, i) => {
    if (used[i]) return
    const members = ps.map((_, j) => j).filter((j) => !used[j] && cx.abs(cx.sub(ps[j], p)) <= tol)
    members.forEach((j) => (used[j] = true))
    const mean = cx.scale(members.map((j) => ps[j]).reduce(cx.add, cx.of(0)), 1 / members.length)
    groups.push({ pole: mean, mult: members.length })
  })
  // Stable sort by modulus (scipy's `cmplx_sort`).
  return groups
    .map((g, k) => ({ g, k }))
    .sort((u, v) => cx.abs(u.g.pole) - cx.abs(v.g.pole) || u.k - v.k)
    .map((x) => x.g)
}

/**
 * Compute the residues of $\text{numerator}/\prod (x - p_i)^{m_i}$ over grouped poles, powers $1, \dots, m$ for each
 * pole (scipy's `_compute_residues`): simple poles by the cover-up rule, repeated ones by repeated division by $(x - p)$.
 *
 * @param groups The grouped poles with their multiplicities.
 * @param numerator The numerator polynomial coefficients in descending degree order.
 * @returns The computed complex residues.
 */
function computeResidues(groups: readonly { pole: cx.C; mult: number }[], numerator: readonly cx.C[]): cx.C[] {
  const linear = (p: cx.C) => [cx.of(1), cx.of(-p.re, -p.im)]
  const residues: cx.C[] = []
  groups.forEach(({ pole, mult }, i) => {
    let others: cx.C[] = [cx.of(1)]
    groups.forEach((g, j) => {
      if (j !== i) for (let r = 0; r < g.mult; r++) others = cx.product(others, linear(g.pole))
    })
    if (mult === 1) {
      residues.push(cx.div(cx.horner(numerator, pole), cx.horner(others, pole)))
      return
    }
    const m = linear(pole)
    const d = cx.horner(others, pole)
    const factor = cx.divide(others, m).q
    let numer = [...numerator]
    const block: cx.C[] = []
    for (let r = 0; r < mult; r++) {
      const { q } = cx.divide(numer, m)
      const rem = cx.horner(numer, pole)
      const coef = cx.div(rem, d)
      // numer ← q − coef·factor (aligned at the constant term)
      const len = Math.max(q.length, factor.length)
      numer = Array.from({ length: len }, (_, k) => {
        const qk = q[k - (len - q.length)] ?? cx.of(0)
        const fk = factor[k - (len - factor.length)] ?? cx.of(0)
        return cx.sub(qk, cx.mul(coef, fk))
      })
      block.push(coef)
    }
    residues.push(...block.reverse())
  })
  return residues
}

/**
 * Strip leading zero coefficients from a complex coefficient list.
 *
 * @param c The complex coefficient list in descending degree order.
 * @returns The slice of coefficients starting from the first nonzero entry, or empty.
 */
const trimFront = (c: cx.C[]) => {
  let k = 0
  while (k < c.length && c[k].re === 0 && c[k].im === 0) k++
  return c.slice(k)
}

/**
 * The partial-fraction expansion of $b(s)/a(s)$, coefficients in descending powers (scipy.signal's `residue`):
 * $b(s)/a(s) = \sum_i \sum_j r_{ij}/(s - p_i)^j + k(s)$. Poles within `tolerance` (default 1e-3) are merged into one
 * repeated pole.
 *
 * @param b Numerator polynomial coefficients in descending degree order.
 * @param a Denominator polynomial coefficients in descending degree order.
 * @param options Options controlling pole clustering tolerance.
 * @param options.tolerance Distance threshold below which poles are merged into a repeated pole (default 1e-3).
 * @returns The partial-fraction expansion containing residues, poles, and direct polynomial term.
 *
 * @example Partial-fraction expansion of a transfer function in s
 * const pf = residue([1], [1, 3, 2])
 * print('poles =', pf.poles)
 */
export function residue(b: ComplexLike, a: ComplexLike, { tolerance = 1e-3 }: ResidueOptions = {}): PartialFractions {
  const bl = cx.readList(b as Value | VectorLike, 'residue')
  const al = cx.readList(a as Value | VectorLike, 'residue')
  const complex = bl.complex || al.complex
  let num = trimFront(bl.list)
  const den = trimFront(al.list)
  if (den.length === 0) throw new DomainError('residue', 'residue: the denominator is zero')
  if (al.complex)
    throw new DTypeError('residue', 'residue: real denominators only (roots are real-coefficient)', ['complex128'])
  const poleList = cx.readList(roots(den.map((z) => z.re)), 'residue').list
  let direct: cx.C[] = []
  if (num.length === 0) {
    const groups = groupRoots(poleList, 0)
    return {
      residues: cx.toTensor(
        poleList.map(() => cx.of(0)),
        true,
      ),
      poles: cx.toTensor(
        groups.map((g) => g.pole),
        true,
      ),
      direct: cx.toTensor([], complex),
    }
  }
  if (num.length >= den.length) {
    const { q, r } = cx.divide(num, den)
    direct = q
    num = r
  }
  const groups = groupRoots(poleList, tolerance)
  const res = computeResidues(groups, num).map((r) => cx.div(r, den[0]))
  const poles = groups.flatMap((g) => Array.from({ length: g.mult }, () => g.pole))
  return { residues: cx.toTensor(res, true), poles: cx.toTensor(poles, true), direct: cx.toTensor(direct, complex) }
}

/**
 * The partial-fraction expansion of $b(z)/a(z)$ in ascending powers of $z^{-1}$ (scipy.signal's `residuez`):
 * $b(z)/a(z) = \sum_i \sum_j r_{ij}/(1 - p_i z^{-1})^j + \sum_k k_k z^{-k}$. Poles within `tolerance` (default 1e-3)
 * are merged.
 *
 * @param b Numerator polynomial coefficients in descending powers of $z^{-1}$.
 * @param a Denominator polynomial coefficients in descending powers of $z^{-1}$.
 * @param options Options controlling pole clustering tolerance.
 * @param options.tolerance Distance threshold below which poles are merged into a repeated pole (default 1e-3).
 * @returns The partial-fraction expansion containing residues, poles, and direct polynomial term.
 *
 * @example Partial-fraction expansion of a discrete-time filter in z^-1
 * const pf = residuez([1], [1, -0.5])
 * print('poles =', pf.poles)
 */
export function residuez(b: ComplexLike, a: ComplexLike, { tolerance = 1e-3 }: ResidueOptions = {}): PartialFractions {
  const bl = cx.readList(b as Value | VectorLike, 'residuez')
  const al = cx.readList(a as Value | VectorLike, 'residuez')
  const complex = bl.complex || al.complex
  if (al.complex) throw new DTypeError('residuez', 'residuez: real denominators only', ['complex128'])
  const trimBack = (c: cx.C[]) => {
    let k = c.length
    while (k > 0 && c[k - 1].re === 0 && c[k - 1].im === 0) k--
    return c.slice(0, k)
  }
  const bs = trimBack(bl.list)
  const as = trimBack(al.list)
  if (as.length === 0 || (as[0].re === 0 && as[0].im === 0))
    throw new DomainError('residuez', 'residuez: a[0] must be nonzero')
  const poleList = cx.readList(roots(as.map((z) => z.re)), 'residuez').list
  if (bs.length === 0)
    return {
      residues: cx.toTensor(
        poleList.map(() => cx.of(0)),
        true,
      ),
      poles: cx.toTensor(
        groupRoots(poleList, 0).map((g) => g.pole),
        true,
      ),
      direct: cx.toTensor([], complex),
    }
  let bRev = [...bs].reverse()
  const aRev = [...as].reverse()
  let kRev: cx.C[] = []
  if (bRev.length >= aRev.length) {
    const { q, r } = cx.divide(bRev, aRev)
    kRev = q
    bRev = r
  }
  const groups = groupRoots(poleList, tolerance)
  const inverted = groups.map((g) => ({ pole: cx.div(cx.of(1), g.pole), mult: g.mult }))
  const res = computeResidues(inverted, bRev)
  const poles: cx.C[] = []
  const powers: number[] = []
  groups.forEach((g) => {
    for (let j = 0; j < g.mult; j++) {
      poles.push(g.pole)
      powers.push(j + 1)
    }
  })
  // rᵢ ← rᵢ · (−pᵢ)^{power} / a_rev[0]
  const scaled = res.map((r, i) => {
    let f = cx.of(1)
    for (let j = 0; j < powers[i]; j++) f = cx.mul(f, cx.of(-poles[i].re, -poles[i].im))
    return cx.div(cx.mul(r, f), aRev[0])
  })
  return {
    residues: cx.toTensor(scaled, true),
    poles: cx.toTensor(poles, true),
    direct: cx.toTensor([...kRev].reverse(), complex),
  }
}
