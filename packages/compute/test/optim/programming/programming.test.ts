import { describe, expect, it } from 'vitest'
import {
  activeSet,
  boxQuadprog,
  boxQuadraticProgram,
  branchAndBound,
  dp,
  dynamicProgram,
  gomory,
  hungarian,
  hungarianSteps,
  linearInteriorPoint,
  linprog,
  lpCentralPath,
  milp,
  quadprog,
  quadraticInteriorPoint,
  simplex,
  type LinearProgram,
} from 'aifn-compute/optim/programming'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-8) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol)
}

// Hillier and Lieberman's Wyndor Glass problem, written as a minimisation. Reference values from scipy's linprog.
const wyndor: LinearProgram = {
  c: [-3, -5],
  A_ub: [
    [1, 0],
    [0, 2],
    [3, 2],
  ],
  b_ub: [4, 12, 18],
}

describe('linear programming', () => {
  it('simplex and interior point agree with scipy on the Wyndor problem, with duals', () => {
    for (const method of ['simplex', 'interior-point'] as const) {
      const r = linprog(wyndor, { method })
      expect(r.status).toBe('optimal')
      const tol = method === 'simplex' ? 1e-10 : 1e-6
      close(toFlat(r.x), [2, 6], tol)
      expect(r.objective).toBeCloseTo(-36, 5)
      close(toFlat(r.report!.duals.ineq), [0, -1.5, -1], tol)
      expect(Math.abs(r.report!.dualityGap)).toBeLessThan(1e-6)
      expect(r.report!.complementarity).toBeLessThan(1e-6)
    }
  })

  it('handles free and doubly bounded variables and equality rows (scipy marginals)', () => {
    const p: LinearProgram = {
      c: [1, 2, -1],
      A_eq: [[1, 1, 1]],
      b_eq: [3],
      A_ub: [[1, -1, 0]],
      b_ub: [-1],
      bounds: [
        [null, null],
        [0, 4],
        [-2, 2],
      ],
    }
    const r = linprog(p)
    close(toFlat(r.x), [0, 1, 2])
    close(toFlat(r.report!.duals.ineq), [-0.5])
    close(toFlat(r.report!.duals.eq), [1.5])
    close(toFlat(r.report!.duals.upper), [0, 0, -2.5])
    expect(linprog(p, { method: 'interior-point' }).objective).toBeCloseTo(0, 6)
  })

  it('reports infeasible and unbounded problems', () => {
    expect(linprog({ c: [1], A_ub: [[1], [-1]], b_ub: [1, -2] }).status).toBe('infeasible')
    const u = linprog({ c: [-1, 0], A_ub: [[1, -1]], b_ub: [1] })
    expect(u.status).toBe('unbounded')
    expect(u.objective).toBe(-Infinity)
  })

  it('the interior-point method tells infeasible from unbounded and returns the certificates', () => {
    const ip = { method: 'interior-point' } as const
    // Unbounded: x₁ − x₂ ≤ 1 lets x₁ = x₂ → ∞ with cᵀx = −x₁ → −∞.
    const u = linprog({ c: [-1, 0], A_ub: [[1, -1]], b_ub: [1] }, ip)
    expect(u.status).toBe('unbounded')
    expect(u.objective).toBe(-Infinity)
    const d = toFlat(u.ray!)
    expect(-d[0]).toBeLessThan(0)
    expect(d[0] - d[1]).toBeLessThanOrEqual(1e-8)
    expect(Math.min(...d)).toBeGreaterThanOrEqual(-1e-8)
    // Infeasible: x ≤ 1 and x ≥ 2.
    expect(linprog({ c: [1], A_ub: [[1], [-1]], b_ub: [1, -2] }, ip).status).toBe('infeasible')
    // Infeasible equalities with x ≥ 0: x₁ + x₂ = −1.
    expect(linprog({ c: [1, 1], A_eq: [[1, 1]], b_eq: [-1] }, ip).status).toBe('infeasible')
    // Both primal and dual infeasible: the primal certificate is reported.
    const both = linprog(
      {
        c: [-1, -1],
        A_ub: [
          [1, -1],
          [-1, 1],
        ],
        b_ub: [-1, -1],
      },
      ip,
    )
    expect(both.status).toBe('infeasible')
    // The state records the certificate and τ → 0 while κ stays positive.
    const s = run(linearInteriorPoint({ c: [1], A_ub: [[1], [-1]], b_ub: [1, -2] }), {}, 200)
    expect(s.terminated).toBe(true)
    expect(s.tau).toBeLessThan(1e-6 * s.kappa)
    // An optimum is still found, with τ bounded away from zero.
    const w = run(linearInteriorPoint(wyndor), {}, 200)
    expect(w.status).toBe('optimal')
    expect(w.tau).toBeGreaterThan(1e-3)
  })

  it("Dantzig's rule visits every vertex of the Klee–Minty cube; Bland's rule does not cycle on Beale's example", () => {
    // Chvátal (1983), p. 47: max Σ 10^(n−j) x_j s.t. 2 Σ_{j<i} 10^(i−j) x_j + x_i ≤ 100^(i−1).
    const n = 3
    const c = Array.from({ length: n }, (_, j) => -(10 ** (n - j - 1)))
    const A = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (j < i ? 2 * 10 ** (i - j) : j === i ? 1 : 0)),
    )
    const b = Array.from({ length: n }, (_, i) => 100 ** i)
    const s = run(simplex({ c, A_ub: A, b_ub: b }, { rule: 'dantzig' }), {}, 100)
    expect(s.status).toBe('optimal')
    expect(s.pivots).toBe(2 ** n - 1)
    expect(s.objective).toBeCloseTo(-(100 ** (n - 1)), 8)
    // Beale (1955).
    const beale: LinearProgram = {
      c: [-0.75, 20, -0.5, 6],
      A_ub: [
        [0.25, -8, -1, 9],
        [0.5, -12, -0.5, 3],
        [0, 0, 1, 0],
      ],
      b_ub: [0, 0, 1],
    }
    expect(run(simplex(beale, { rule: 'dantzig' }), {}, 100).status).toBe('cycling')
    const bland = run(simplex(beale, { rule: 'bland' }), {}, 100)
    expect(bland.status).toBe('optimal')
    expect(bland.objective).toBeCloseTo(-1.25, 10)
  })

  it('the simplex and interior-point algorithms satisfy the Algorithm protocol', () => {
    expect(trace(simplex(wyndor), {}, 50).meta.stopped).toBe('done')
    checkProtocol(simplex(wyndor), {}, { steps: 6, record: { objective: (s) => s.objective } })
    checkProtocol(simplex(wyndor, { rule: 'bland' }), {}, { steps: 6 })
    checkProtocol(linearInteriorPoint(wyndor), {}, { steps: 8, record: { mu: (s) => s.mu } })
  })

  it('the central path tends to the optimum as μ → 0', () => {
    const path = lpCentralPath(wyndor, [10, 1, 0.1, 1e-4])
    expect(toFlat(path.converged)).toEqual([1, 1, 1, 1])
    close(toFlat(path.x).slice(6), [2, 6], 1e-3)
  })
})

describe('quadratic programming', () => {
  // min ½‖x − (1, 2.5)‖² s.t. the polygon of Nocedal and Wright, Example 16.3; optimum (1.4, 1.7).
  const qp = {
    Q: [
      [2, 0],
      [0, 2],
    ],
    c: [-2, -5],
    A: [
      [-1, 2],
      [1, 2],
      [1, -2],
      [-1, 0],
      [0, -1],
    ],
    b: [2, 6, 2, 0, 0],
  }
  it('active set and interior point reach the same KKT point', () => {
    for (const method of ['active-set', 'interior-point'] as const) {
      const r = quadprog(qp, { method, x0: method === 'active-set' ? [2, 0] : undefined })
      expect(r.status).toBe('optimal')
      close(toFlat(r.x), [1.4, 1.7], 1e-7)
      expect(r.report.stationarity).toBeLessThan(1e-7)
      close(toFlat(r.report.lambda), [0.8, 0, 0, 0, 0], 1e-7)
    }
    expect(quadprog(qp).status).toBe('optimal')
  })

  it('active set stops nonconvex on negative curvature, and runs when the constraints remove it', () => {
    // Q = diag(1, −1): indefinite. Without constraints on x₂ the subproblem's stationary point is a saddle.
    const saddle = {
      Q: [
        [1, 0],
        [0, -1],
      ],
      c: [0, 0],
      A: [[1, 0]],
      b: [5],
    }
    const s = run(activeSet(saddle), { x0: [0, 0] }, 5)
    expect(s.status).toBe('nonconvex')
    expect(s.terminated).toBe(true)
    // Fixing x₂ by equality leaves the positive direction only: the method solves it.
    const fixed = { ...saddle, E: [[0, 1]], e: [1], c: [-2, 0] }
    const f = run(activeSet(fixed), { x0: [0, 1] }, 10)
    expect(f.status).toBe('optimal')
    close(toFlat(f.x), [2, 1], 1e-9)
  })

  it('the curvature check sees through dependent equality rows (rank-revealing null space)', () => {
    // Q = diag(1, 1, −1) with x₁ fixed twice (two identical rows of E): null(E) still holds the negative direction x₃.
    const dependent = {
      Q: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
      ],
      c: [0, 0, 0],
      E: [
        [1, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
      ],
      e: [1, 1, 2],
      A: [[0, 1, 0]],
      b: [5],
    }
    expect(quadprog(dependent, { method: 'interior-point' }).status).toBe('nonconvex')
  })

  it('interior point stops nonconvex on an indefinite Q, and solves when E removes the negative direction', () => {
    // The lab's case: Q = diag(2, −2) on the polygon. Without the check the iterates reach a KKT point reported optimal.
    const indefinite = {
      ...qp,
      Q: [
        [2, 0],
        [0, -2],
      ],
    }
    const r = quadprog(indefinite, { method: 'interior-point' })
    expect(r.status).toBe('nonconvex')
    expect(r.steps).toBe(0)
    const s = run(quadraticInteriorPoint(indefinite), {}, 50)
    expect(s.status).toBe('nonconvex')
    expect(s.terminated).toBe(true)
    expect(s.converged).toBe(false)
    expect(quadprog(indefinite).status).toBe('nonconvex')
    // Q = diag(1, −1) with x₂ fixed by an equality: convex on null(E), so the interior point solves it.
    const fixed = {
      Q: [
        [1, 0],
        [0, -1],
      ],
      c: [-2, 0],
      A: [[1, 0]],
      b: [5],
      E: [[0, 1]],
      e: [1],
    }
    const f = quadprog(fixed, { method: 'interior-point' })
    expect(f.status).toBe('optimal')
    close(toFlat(f.x), [2, 1], 1e-7)
  })

  it('a positive semi-definite, singular Q still solves by interior point', () => {
    // Q = [[1, 1], [1, 1]] has eigenvalues 2 and 0: the objective is flat along (1, −1), and the constraints fix x.
    const flat = {
      Q: [
        [1, 1],
        [1, 1],
      ],
      c: [0, -3],
      A: [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
      ],
      b: [1, 1, 0, 0],
    }
    for (const method of ['interior-point', 'active-set'] as const) {
      const r = quadprog(flat, { method, x0: method === 'active-set' ? [0, 0] : undefined })
      expect(r.status).toBe('optimal')
      expect(r.report.stationarity).toBeLessThan(1e-6)
    }
    close(toFlat(quadprog(flat, { method: 'interior-point' }).x), [0, 1], 1e-6)
  })

  it('the QP algorithms satisfy the Algorithm protocol', () => {
    checkProtocol(activeSet(qp), { x0: [2, 0] }, { steps: 6 })
    checkProtocol(quadraticInteriorPoint(qp), {}, { steps: 8 })
    const box = {
      Q: [
        [2, 0.5],
        [0.5, 1],
      ],
      c: [-4, 1],
      lower: [0, 0],
      upper: [1, 1],
    }
    checkProtocol(boxQuadraticProgram(box), {}, { steps: 5 })
  })

  it('box QP projects and solves the free subspace', () => {
    const s = boxQuadprog({
      Q: [
        [2, 0.5],
        [0.5, 1],
      ],
      c: [-4, 1],
      lower: [0, 0],
      upper: [1, 1],
    })
    expect(s.converged).toBe(true)
    close(toFlat(s.x), [1, 0], 1e-9)
  })
})

describe('integer programming', () => {
  // max x + y s.t. −2x + 2y ≥ 1, −8x + 10y ≤ 13, x, y ≥ 0 integer: optimum (1, 2), value 3.
  const ip = {
    c: [-1, -1],
    A_ub: [
      [2, -2],
      [-8, 10],
    ],
    b_ub: [-1, 13],
  }
  it('branch and bound finds the integer optimum and keeps the tree', () => {
    for (const strategy of ['depth-first', 'best-bound', 'breadth-first'] as const) {
      const r = milp(ip, { strategy })
      expect(r.status).toBe('optimal')
      close(toFlat(r.x), [1, 2])
      expect(r.objective).toBe(-3)
      expect(r.tree[0].status).toBe('branched')
    }
    const t = trace(branchAndBound(ip), {}, 100)
    expect(t.meta.stopped).toBe('done')
    expect(t.final.status).toBe('optimal')
  })

  it('mixed-integer: continuous variables stay continuous', () => {
    const r = milp({ ...ip, integrality: [1, 0] })
    expect(r.objective).toBeCloseTo(-8.5, 9)
  })

  it('Gomory cuts reach the integer optimum and never cut off integer points', () => {
    const s = run(gomory(ip), {}, 50)
    expect(s.status).toBe('optimal')
    close(toFlat(s.x), [1, 2], 1e-7)
    for (const cut of s.cuts) {
      const a = toFlat(cut.a)
      expect(a[0] * 1 + a[1] * 2).toBeLessThanOrEqual(cut.b + 1e-9)
    }
  })
})

describe('integer programming protocol', () => {
  const ip = {
    c: [-1, -1],
    A_ub: [
      [2, -2],
      [-8, 10],
    ],
    b_ub: [-1, 13],
  }
  it('branch and bound and Gomory satisfy the Algorithm protocol', () => {
    for (const strategy of ['depth-first', 'best-bound', 'breadth-first'] as const)
      checkProtocol(branchAndBound(ip, { strategy }), {}, { steps: 6 })
    checkProtocol(gomory(ip), {}, { steps: 6 })
  })
})

describe('dynamic programming', () => {
  it('generic dp computes Fibonacci', () => {
    const { table } = dp({ shape: [11], cell: (i, _j, get) => (i < 2 ? i : get(i - 1, 0) + get(i - 2, 0)) })
    expect(toFlat(table)[10]).toBe(55)
  })

  it('dynamicProgram fills one row per step and ends at dp', () => {
    const edit = (a: string, b: string) => ({
      shape: [a.length + 1, b.length + 1] as [number, number],
      cell: (i: number, j: number, get: (i: number, j: number) => number) =>
        i === 0
          ? j
          : j === 0
            ? i
            : Math.min(get(i - 1, j) + 1, get(i, j - 1) + 1, get(i - 1, j - 1) + (a[i - 1] === b[j - 1] ? 0 : 1)),
    })
    const p = edit('kitten', 'sitting')
    expect(toFlat(dp(p).table).at(-1)).toBe(3)
    const s = run(dynamicProgram(p), {}, 100)
    expect(s.converged).toBe(true)
    expect(toFlat(s.table)).toEqual(toFlat(dp(p).table))
    checkProtocol(dynamicProgram(p), {}, { steps: 5 })
  })
})

describe('assignment', () => {
  it('Hungarian matches the brute-force optimum, square and rectangular', () => {
    const C = [
      [4, 1, 3],
      [2, 0, 5],
      [3, 2, 2],
    ]
    const r = hungarian(C)
    expect(r.cost).toBe(5)
    expect(toFlat(r.assignment)).toEqual([1, 0, 2])
    expect(hungarian(C, { maximize: true }).cost).toBe(11)
    const rect = hungarian([
      [4, 1],
      [2, 0],
      [3, 2],
    ])
    expect(rect.cost).toBe(3)
  })

  it('hungarianSteps satisfies the Algorithm protocol and ends at the optimum', () => {
    const C = [
      [4, 1, 3],
      [2, 0, 5],
      [3, 2, 2],
    ]
    checkProtocol(hungarianSteps(C), {}, { steps: 8 })
    expect(run(hungarianSteps(C), {}, 1000).converged).toBe(true)
  })
})
