import { describe, expect, it } from 'vitest'
import {
  apply2,
  cholesky,
  cholesky2,
  choleskyLogDet,
  choleskySolve,
  conditionNumber,
  det,
  det2,
  eig2,
  eigh,
  eigh2,
  inv2,
  inverse,
  kron,
  LinAlgError,
  logDet,
  lstsq,
  lu,
  luSolve,
  normFrobenius,
  pinv,
  qr,
  signDet,
  solve,
  solveTriangular,
  svd,
  svd2,
  matrixTrace,
  type Mat2,
} from 'aifn-compute/numerics/linalg'
import {
  abs,
  diag,
  diagonal,
  eye,
  matmul,
  max,
  shapeOf,
  sub,
  tensor,
  toFlat,
  transpose,
  type NestedArray,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type N = NestedArray
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<Record<string, any>>('numerics/linalg')
const T = (k: string): Tensor => tensor(F[k] as N)

/** Largest absolute difference between a value and a numpy result, after checking shapes. */
function maxError(actual: Value, expected: N): number {
  if (typeof actual === 'number') return Math.abs(actual - (expected as number))
  const t = actual as Tensor
  expect(t.shape).toEqual(shapeOf(expected))
  const want = (typeof expected === 'number' ? [expected] : (expected as N[]).flat(Infinity as 1)) as number[]
  return Math.max(0, ...toFlat(t).map((v, k) => Math.abs(v - want[k])))
}

function expectClose(actual: Value, expected: N, tol = 1e-12) {
  expect(maxError(actual, expected)).toBeLessThanOrEqual(tol)
}

/** ‖A − B‖_max for tensors. */
function distance(a: Tensor, b: Tensor): number {
  return max(abs(sub(a, b))) as number
}

/** Columns agree up to one sign per column. */
function expectColumnsUpToSign(actual: Tensor, expected: N, tol = 1e-10) {
  const want = tensor(expected)
  expect(actual.shape).toEqual(want.shape)
  const [m, k] = actual.shape
  for (let c = 0; c < k; c++) {
    let dot = 0
    for (let i = 0; i < m; i++) dot += actual.data[i * k + c] * want.data[i * k + c]
    const s = Math.sign(dot) || 1
    for (let i = 0; i < m; i++) expect(Math.abs(actual.data[i * k + c] - s * want.data[i * k + c])).toBeLessThan(tol)
  }
}

describe('cholesky', () => {
  it('matches numpy on an SPD matrix, without jitter', () => {
    const r = cholesky(T('spd'))
    expect(r.jitter).toBe(0)
    expect(r.failed).toBe(false)
    expect(r.failedAt).toBe(-1)
    expectClose(r.L, F.cholesky)
  })
  it('reads only the lower triangle', () => {
    const A = toFlat(T('spd'))
    A[1] = 1e6 // upper entry (0, 1)
    expectClose(cholesky(tensor(A, [5, 5])).L, F.cholesky)
  })
  it('adds and reports the smallest jitter for a rank-deficient PSD matrix, instead of returning NaN', () => {
    const psd = T('psd')
    expect(cholesky(psd, { jitter: false }).failed).toBe(true)
    const r = cholesky(psd)
    expect(r.failed).toBe(false)
    expect(r.jitter).toBeGreaterThan(0)
    expect(toFlat(r.L).every(Number.isFinite)).toBe(true)
    const rebuilt = matmul(r.L, transpose(r.L))
    expect(distance(rebuilt, addJitter(psd, r.jitter))).toBeLessThan(1e-12)
    // The first rung of the ladder (1e-12 times the mean diagonal) sufficed.
    const meanDiagonal = (toFlat(diagonal(psd)) as number[]).reduce((a, b) => a + b, 0) / 4
    expect(r.jitter).toBeCloseTo(meanDiagonal * 1e-12, 25)
  })
  it('reports failure on an indefinite matrix with a NaN-free partial factor', () => {
    const r = cholesky(
      tensor([
        [1, 2],
        [2, 1],
      ]),
    )
    expect(r.failed).toBe(true)
    expect(r.failedAt).toBe(1)
    // L is the partial factor of A + jitter·I for the last jitter tried.
    const l11 = Math.sqrt(1 + r.jitter)
    expect(toFlat(r.L)).toEqual([l11, 0, 2 / l11, 0])
  })
  it('handles a zero diagonal (a deterministic component) with jitter, where a floor of 0 gave NaN', () => {
    const r = cholesky(
      tensor([
        [1, 0],
        [0, 0],
      ]),
    )
    expect(r.failed).toBe(false)
    expect(r.jitter).toBeGreaterThan(0)
    expect(toFlat(r.L).every(Number.isFinite)).toBe(true)
  })
  it('throws on non-finite input', () => {
    expect(() => cholesky(tensor([[NaN]]))).toThrow(LinAlgError)
  })
  it('choleskySolve and choleskyLogDet', () => {
    const { L } = cholesky(T('spd'))
    expectClose(choleskySolve(L, T('rhs')), F.choSolve, 1e-12)
    expect(choleskyLogDet(L)).toBeCloseTo(F.spdLogDet, 12)
  })
})

function addJitter(a: Tensor, j: number): Tensor {
  return tensor(
    toFlat(a).map((v, k) => (k % (a.shape[0] + 1) === 0 ? v + j : v)),
    a.shape,
  )
}

describe('triangular solves against scipy', () => {
  it('lower, upper, transposed, unit diagonal, vector and matrix right-hand sides', () => {
    expectClose(solveTriangular(T('lower'), T('rhs')), F.triangular.lower)
    expectClose(solveTriangular(T('lower'), T('rhs'), { transpose: true }), F.triangular.lowerTrans)
    expectClose(solveTriangular(T('upper'), T('rhs'), { lower: false }), F.triangular.upper)
    expectClose(solveTriangular(T('upper'), T('vec'), { lower: false, transpose: true }), F.triangular.upperTrans)
    expectClose(solveTriangular(T('lower'), T('vec'), { unitDiagonal: true }), F.triangular.unit)
  })
  it('throws on a zero diagonal', () => {
    expect(() =>
      solveTriangular(
        tensor([
          [1, 0],
          [1, 0],
        ]),
        tensor([1, 1]),
      ),
    ).toThrow(/singular/)
  })
})

describe('LU, solve, inverse and determinants', () => {
  it('lu matches scipy (PA = LU)', () => {
    const f = lu(T('general'))
    expectClose(f.P, F.lu.P, 0)
    expectClose(f.L, F.lu.L)
    expectClose(f.U, F.lu.U)
    expect(f.singular).toBe(false)
    expect(distance(matmul(f.P, T('general')), matmul(f.L, f.U))).toBeLessThan(1e-14)
  })
  it('solve, luSolve, inverse, det and logDet match numpy', () => {
    expectClose(solve(T('general'), T('rhs')), F.solve)
    expectClose(solve(T('general'), T('vec')), F.solveVec)
    expectClose(luSolve(lu(T('general')), T('rhs')), F.solve)
    expectClose(inverse(T('general')), F.inverse)
    expect(det(T('general'))).toBeCloseTo(F.det, 12)
    expect(logDet(T('general'))).toBeCloseTo(F.logAbsDet, 12)
    expect(signDet(T('general'))).toBe(F.signDet)
  })
  it('reports a singular matrix instead of returning Infinity or NaN', () => {
    for (const k of ['singular', 'exactlySingular']) {
      const A = T(k)
      expect(lu(A).singular).toBe(true)
      expect(toFlat(lu(A).U).every(Number.isFinite)).toBe(true)
      expect(() => solve(A, tensor(new Array(A.shape[0]).fill(1)))).toThrow(LinAlgError)
      expect(() => inverse(A)).toThrow(/singular/)
      expect(() => luSolve(lu(A), tensor(new Array(A.shape[0]).fill(1)))).toThrow(/singular/)
      expect(Math.abs(det(A))).toBeLessThan(1e-12)
    }
    expect(det(T('exactlySingular'))).toBe(0)
    expect(logDet(T('exactlySingular'))).toBe(-Infinity)
    expect(signDet(T('exactlySingular'))).toBe(0)
  })
  it('an all-zero matrix: singular, no NaN', () => {
    const f = lu(
      tensor([
        [0, 0],
        [0, 0],
      ]),
    )
    expect(f.singular).toBe(true)
    expect(toFlat(f.L).every(Number.isFinite)).toBe(true)
  })
  it('ill-conditioned: the Hilbert matrix of order 8 (cond ≈ 1.5e10) solves with a small residual', () => {
    const H = T('hilbert')
    const x = solve(H, T('hilbertRhs'))
    const residual = sub(matmul(H, x), T('hilbertRhs'))
    expect(max(abs(residual)) as number).toBeLessThan(1e-14)
    expect(lu(H).singular).toBe(false)
    // The solution loses about log10(cond) digits: within 1e-5 of the exact ones.
    expect(max(abs(sub(x, 1))) as number).toBeLessThan(1e-5)
  })
})

describe('QR against numpy', () => {
  it('tall reduced and complete, wide reduced (LAPACK signs)', () => {
    const t = qr(T('tall'))
    expectClose(t.Q, F.qrTall.Q)
    expectClose(t.R, F.qrTall.R)
    const c = qr(T('tall'), { mode: 'complete' })
    expectClose(c.R, F.qrTallComplete.R)
    expect(distance(matmul(transpose(c.Q), c.Q), eye(6))).toBeLessThan(1e-14)
    expect(distance(matmul(c.Q, c.R), T('tall'))).toBeLessThan(1e-14)
    const w = qr(T('wide'))
    expectClose(w.Q, F.qrWide.Q)
    expectClose(w.R, F.qrWide.R)
  })
  it('handles a column that is already zero below the diagonal', () => {
    const { Q, R } = qr(
      tensor([
        [2, 1],
        [0, 3],
      ]),
    )
    expect(toFlat(Q)).toEqual([1, 0, 0, 1])
    expect(toFlat(R)).toEqual([2, 1, 0, 3])
  })
})

describe('eigh against numpy', () => {
  it('SPD: values descending, vectors up to sign', () => {
    const { values, vectors, converged } = eigh(T('spd'))
    expect(converged).toBe(true)
    expectClose(values, F.eigh.values, 1e-12 * 30)
    expectColumnsUpToSign(vectors, F.eigh.vectors, 1e-12)
  })
  it('ill-conditioned Hilbert matrix: eigenvalues to about ε‖H‖', () => {
    expectClose(eigh(T('hilbert')).values, F.hilbertEigenvalues, 1e-15)
  })
  it('repeated eigenvalues: an orthonormal basis that diagonalises A', () => {
    const { Q } = qr(T('general'))
    const A = matmul(matmul(Q, diag(tensor([2, 2, 1, 1, -3]))), transpose(Q))
    const { values, vectors } = eigh(A)
    expectClose(values, [2, 2, 1, 1, -3], 1e-14)
    expect(distance(matmul(transpose(vectors), vectors), eye(5))).toBeLessThan(1e-14)
    expect(distance(matmul(matmul(vectors, diag(values)), transpose(vectors)), A)).toBeLessThan(1e-14)
  })
  it('signs each eigenvector so that its largest component is positive', () => {
    const { vectors } = eigh(T('spd'))
    for (let c = 0; c < 5; c++) {
      const col = [0, 1, 2, 3, 4].map((i) => vectors.data[i * 5 + c])
      const big = col.reduce((b, v, i) => (Math.abs(v) > Math.abs(col[b]) ? i : b), 0)
      expect(col[big]).toBeGreaterThan(0)
    }
  })
})

describe('SVD, pinv, lstsq and condition numbers against numpy', () => {
  it('tall and wide thin SVDs', () => {
    for (const [k, f] of [
      ['tall', F.svdTall],
      ['wide', F.svdWide],
    ] as const) {
      const { U, S, V, converged } = svd(T(k))
      expect(converged).toBe(true)
      expectClose(S, f.S, 1e-14)
      expectColumnsUpToSign(U, f.U, 1e-13)
      expectColumnsUpToSign(V, f.V, 1e-13)
      expect(distance(matmul(matmul(U, diag(S)), transpose(V)), T(k))).toBeLessThan(1e-14)
    }
  })
  it('singular values of a rank-deficient matrix, with U completed to an orthonormal set', () => {
    const { U, S } = svd(T('singular'))
    expectClose(S, F.singularSvd, 1e-14)
    expect(distance(matmul(transpose(U), U), eye(3))).toBeLessThan(1e-14)
    const zero = svd(
      tensor([
        [0, 0],
        [0, 0],
        [0, 0],
      ]),
    )
    expect(toFlat(zero.S)).toEqual([0, 0])
    expect(distance(matmul(transpose(zero.U), zero.U), eye(2))).toBeLessThan(1e-15)
  })
  it('converges on a rank-deficient square matrix whose null column underflows (review regression)', () => {
    // A null column driven to ~1e-157 has a subnormal squared norm; the convergence test used to rotate it forever.
    const A = tensor([
      [1, 2, 3],
      [2, 4, 6],
      [1, 0, 1],
    ])
    const r = svd(A)
    expect(r.converged).toBe(true)
    expect(toFlat(r.S)[2]).toBeLessThan(1e-14)
    expect(distance(matmul(matmul(r.U, diag(r.S)), transpose(r.V)), A)).toBeLessThan(1e-14)
  })
  it('keeps a singular value near 1e-300 instead of underflowing it to 0 (review regression)', () => {
    expect(
      toFlat(
        svd(
          tensor([
            [1e-300, 0],
            [0, 1],
          ]),
        ).S,
      ),
    ).toEqual([1, 1e-300])
  })
  it('pinv, including a rank-deficient matrix', () => {
    expectClose(pinv(T('tall')), F.pinvTall, 1e-13)
    expectClose(pinv(T('wide')), F.pinvWide, 1e-13)
    expectClose(pinv(T('singular')), F.singularPinv, 1e-12)
  })
  it('lstsq: overdetermined, matrix right-hand side, and rank-deficient (minimum norm)', () => {
    const o = lstsq(T('over'), T('overB'))
    expectClose(o.x, F.overLstsq.x, 1e-13)
    expectClose(o.residuals, F.overLstsq.residuals, 1e-12)
    expect(o.rank).toBe(F.overLstsq.rank)
    expectClose(o.singularValues, F.overLstsq.singularValues, 1e-13)
    const om = lstsq(T('over'), T('overBm'))
    expectClose(om.x, F.overLstsqMatrix.x, 1e-13)
    expectClose(om.residuals, F.overLstsqMatrix.residuals, 1e-12)
    const s = lstsq(T('singular'), T('singularRhs'))
    expectClose(s.x, F.singularLstsq.x, 1e-12)
    expect(s.rank).toBe(2)
  })
  it('condition numbers, including the Hilbert matrix and a singular matrix', () => {
    expect(conditionNumber(T('general')) / F.condGeneral).toBeCloseTo(1, 12)
    // Jacobi finds the smallest singular value to high relative accuracy; LAPACK's is accurate to ε·σ_max.
    expectClose(svd(T('hilbert')).S, F.hilbertSingularValues, 1e-15)
    expect(conditionNumber(T('hilbert')) / F.hilbertCond).toBeCloseTo(1, 5)
    expect(conditionNumber(T('exactlySingular'))).toBe(Infinity)
  })
})

describe('kron, trace and Frobenius norm against numpy', () => {
  it('match', () => {
    expectClose(kron(T('k1'), T('k2')), F.kron, 1e-15)
    expect(matrixTrace(T('general'))).toBeCloseTo(F.traceGeneral, 14)
    expect(normFrobenius(T('general'))).toBeCloseTo(F.froGeneral, 14)
  })
})

describe('2×2 closed forms against numpy', () => {
  const m = (k: string) => F.two[k] as Mat2
  it('eigh2', () => {
    const { values, vectors } = eigh2(m('sym'))
    expect(values[0]).toBeCloseTo(F.twoOut.symEig[0][0], 14)
    expect(values[1]).toBeCloseTo(F.twoOut.symEig[0][1], 14)
    for (const j of [0, 1]) {
      const v = vectors[j]
      const Av = apply2(m('sym'), v)
      expect(Av[0]).toBeCloseTo(values[j] * v[0], 14)
      expect(Av[1]).toBeCloseTo(values[j] * v[1], 14)
      expect(Math.max(Math.abs(v[0]), Math.abs(v[1]))).toBe(Math.abs(v[0]) >= Math.abs(v[1]) ? v[0] : v[1])
    }
    // Agrees with the n×n eigh, signs included.
    const big = eigh(tensor(m('sym')))
    expect(toFlat(big.vectors)[0]).toBeCloseTo(vectors[0][0], 14)
    expect(toFlat(big.vectors)[2]).toBeCloseTo(vectors[0][1], 14)
  })
  it('eig2: real and complex', () => {
    const r = eig2(m('general'))
    expect(r.kind).toBe('real')
    if (r.kind === 'real') {
      expect(r.values[0]).toBeCloseTo(F.twoOut.generalEig[0], 14)
      expect(r.values[1]).toBeCloseTo(F.twoOut.generalEig[1], 14)
      const Av = apply2(m('general'), r.vectors[1])
      expect(Av[0]).toBeCloseTo(r.values[1] * r.vectors[1][0], 13)
    }
    const c = eig2(m('rotation'))
    expect(c.kind).toBe('complex')
    if (c.kind === 'complex') {
      expect(c.re).toBeCloseTo(F.twoOut.rotationEig[0], 15)
      expect(c.im).toBeCloseTo(F.twoOut.rotationEig[1], 15)
    }
    expect(
      eig2([
        [3, 0],
        [0, 3],
      ]),
    ).toEqual({
      kind: 'real',
      values: [3, 3],
      vectors: [
        [1, 0],
        [0, 1],
      ],
    })
  })
  it('inv2, det2, cholesky2', () => {
    expectClose(tensor(inv2(m('general'))!), F.twoOut.generalInv, 1e-15)
    expect(det2(m('general'))).toBe(-2)
    expect(
      inv2([
        [1, 2],
        [2, 4],
      ]),
    ).toBeNull()
    expectClose(tensor(cholesky2(m('sym'))!), F.twoOut.symChol, 1e-15)
    expect(
      cholesky2([
        [1, 2],
        [2, 1],
      ]),
    ).toBeNull()
  })
  it('svd2, including a nearly singular matrix whose small singular value keeps its accuracy', () => {
    for (const k of ['general', 'nearlySingular', 'rotation', 'sym']) {
      const { s, u, v } = svd2(m(k))
      const expected =
        k === 'general'
          ? F.twoOut.generalSvd
          : k === 'nearlySingular'
            ? F.twoOut.nearlySingularSvd
            : svd(tensor(m(k))).S.data
      expect(s[0]).toBeCloseTo(expected[0], 14)
      expect(Math.abs(s[1] - expected[1]) / Math.max(expected[1], 1e-300)).toBeLessThan(1e-5)
      // A = s₁u₁v₁ᵀ + s₂u₂v₂ᵀ.
      const A = m(k)
      for (const i of [0, 1]) {
        for (const j of [0, 1]) {
          expect(s[0] * u[0][i] * v[0][j] + s[1] * u[1][i] * v[1][j]).toBeCloseTo(A[i][j], 14)
        }
      }
    }
  })
})
