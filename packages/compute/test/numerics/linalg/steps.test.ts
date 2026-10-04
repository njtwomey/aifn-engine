/**
 * The step-through forms of `aifn-compute/numerics/linalg`: Gram–Schmidt (classical and modified), Householder QR steps,
 * Jacobi, Gauss–Seidel/SOR and the power iteration, checked against the factorisations and solvers they teach.
 */
import { describe, expect, it } from 'vitest'
import { matmul, tensor, toFlat, toRows, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  eigh,
  gaussSeidelSteps,
  gramSchmidt,
  gramSchmidtSteps,
  householderSteps,
  jacobiSteps,
  powerIterationSteps,
  qr,
  solve,
  solveStationary,
} from 'aifn-compute/numerics/linalg'

const close = (a: Tensor | number[], b: Tensor | number[], tol = 1e-10) => {
  const x = Array.isArray(a) ? a : Array.from(toFlat(a))
  const y = Array.isArray(b) ? b : Array.from(toFlat(b))
  expect(x.length).toBe(y.length)
  x.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(tol))
}

const A = [
  [12, -51, 4],
  [6, 167, -68],
  [-4, 24, -41],
  [1, 2, 3],
]

describe('gramSchmidt', () => {
  it.each(['classical', 'modified'] as const)('%s gives A = QR with orthonormal Q and positive diag(R)', (variant) => {
    const { Q, R, orthogonalityError, rankDeficient } = gramSchmidt(A, { variant })
    close(matmul(Q, R), tensor(A), 1e-10)
    close(
      matmul(transpose(Q), Q),
      tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]),
      1e-12,
    )
    expect(orthogonalityError).toBeLessThan(1e-12)
    expect(rankDeficient).toBe(false)
    const r = toRows(R) as number[][]
    for (let i = 0; i < 3; i++) {
      expect(r[i][i]).toBeGreaterThan(0)
      for (let j = 0; j < i; j++) expect(r[i][j]).toBe(0)
    }
  })

  it('matches Householder QR up to the signs of the columns', () => {
    const gs = gramSchmidt(A)
    const h = qr(tensor(A))
    const Rg = toRows(gs.R) as number[][]
    const Rh = toRows(h.R) as number[][]
    for (let i = 0; i < 3; i++) {
      const s = Math.sign(Rh[i][i])
      for (let j = i; j < 3; j++) expect(Math.abs(Rg[i][j] - s * Rh[i][j])).toBeLessThan(1e-9)
    }
  })

  it('takes one column per step', () => {
    const tr = trace(gramSchmidtSteps(A), undefined, 10, { keep: 'all' })
    expect(tr.steps.length).toBe(4)
    expect(tr.steps.map((s) => s.column)).toEqual([0, 1, 2, 3])
  })

  it('modified keeps orthogonality where classical loses it (Läuchli matrix)', () => {
    const e = 1e-8
    const lauchli = [
      [1, 1, 1],
      [e, 0, 0],
      [0, e, 0],
      [0, 0, e],
    ]
    const cgs = gramSchmidt(lauchli, { variant: 'classical' }).orthogonalityError
    const mgs = gramSchmidt(lauchli, { variant: 'modified' }).orthogonalityError
    expect(cgs).toBeGreaterThan(0.1)
    expect(mgs).toBeLessThan(1e-7)
  })

  it('flags a dependent column', () => {
    const { rankDeficient, R } = gramSchmidt([
      [1, 2],
      [2, 4],
    ])
    expect(rankDeficient).toBe(true)
    expect((toRows(R) as number[][])[1][1]).toBe(0)
  })
})

describe('householderSteps', () => {
  it('keeps QR = A at every step and ends at qr(A)', () => {
    const tr = trace(householderSteps(A), undefined, 10, { keep: 'all' })
    for (const s of tr.steps) {
      close(matmul(s.Q, s.R), tensor(A), 1e-9)
      expect(s.residual).toBeLessThan(1e-9)
    }
    const last = tr.steps[tr.steps.length - 1]
    expect(last.column).toBe(3)
    const h = qr(tensor(A))
    close((toRows(last.R) as number[][]).slice(0, 3).flat(), Array.from(toFlat(h.R)), 1e-9)
    // After step j the first j columns are zero below the diagonal.
    const after1 = toRows(tr.steps[1].R) as number[][]
    for (let i = 1; i < 4; i++) expect(Math.abs(after1[i][0])).toBe(0)
  })
})

const M = [
  [4, 1, 0],
  [1, 5, 2],
  [0, 2, 6],
]
const b = [1, 2, 3]

describe('stationary solvers', () => {
  const exact = Array.from(toFlat(solve(tensor(M), tensor(b))))
  it.each([
    ['jacobi', () => jacobiSteps(M, b)],
    ['gauss-seidel', () => gaussSeidelSteps(M, b)],
    ['sor', () => gaussSeidelSteps(M, b, { omega: 1.2 })],
  ] as const)('%s converges to the solution of a diagonally dominant system', (_, make) => {
    const s = run(make(), undefined, 500)
    expect(s.converged).toBe(true)
    close(s.x, exact, 1e-9)
  })

  it('Gauss–Seidel needs fewer sweeps than Jacobi here', () => {
    const j = solveStationary(M, b, { method: 'jacobi' })
    const g = solveStationary(M, b)
    expect(g.steps).toBeLessThan(j.steps)
  })

  it('Jacobi diverges when the spectral radius exceeds 1', () => {
    const s = run(
      jacobiSteps(
        [
          [1, 3],
          [3, 1],
        ],
        [1, 1],
      ),
      undefined,
      2000,
    )
    expect(s.converged).toBe(false)
  })
})

describe('powerIterationSteps', () => {
  const { values } = eigh(tensor(M))
  const ev = Array.from(toFlat(values))
  it('finds the dominant eigenvalue', () => {
    const s = run(powerIterationSteps(M), undefined, 500)
    expect(s.converged).toBe(true)
    expect(Math.abs(s.value - Math.max(...ev))).toBeLessThan(1e-9)
  })
  it('with a shift, finds the eigenvalue nearest it', () => {
    const target = Math.min(...ev)
    const s = run(powerIterationSteps(M, { shift: target + 0.3 }), undefined, 200)
    expect(Math.abs(s.value - target)).toBeLessThan(1e-9)
  })
})
