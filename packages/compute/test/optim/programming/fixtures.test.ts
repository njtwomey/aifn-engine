import { describe, expect, it } from 'vitest'
import {
  activeSet,
  boxQuadraticProgram,
  branchAndBound,
  dualityReport,
  dynamicProgram,
  gomory,
  hungarianSteps,
  hungarian,
  linearInteriorPoint,
  linprog,
  milp,
  quadprog,
  quadraticInteriorPoint,
  simplex,
  type LinearProgram,
  type MixedIntegerProgram,
  type QuadraticProgram,
} from 'aifn-compute/optim/programming'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'

type LpCase = {
  problem: LinearProgram
  status: string
  x?: number[]
  objective?: number
  ineq?: number[]
  eq?: number[]
  lower?: number[]
  upper?: number[]
}
type Fixture = {
  lp: Record<string, LpCase>
  simplex: { problems: string[] }
  linearInteriorPoint: { problems: string[] }
  qp: Record<
    string,
    { problem: QuadraticProgram; x: number[]; objective: number; lambda: number[]; nu: number[]; start: number[] }
  >
  activeSet: { problems: string[] }
  quadraticInteriorPoint: { problems: string[] }
  boxQuadraticProgram: Record<
    string,
    { problem: { Q: number[][]; c: number[]; lower: number[]; upper: number[] }; x: number[]; objective: number }
  >
  milp: Record<string, { problem: MixedIntegerProgram; x: number[]; objective: number }>
  branchAndBound: { problems: string[] }
  gomory: { problems: string[] }
  hungarianSteps: Record<string, { cost: number[][]; maximize: boolean; rows: number[]; cols: number[]; total: number }>
  dynamicProgram: { weights: number[]; values: number[]; capacity: number; table: number[][] }
}
const F = fixture<Fixture>('optim/programming')

const near = (got: number, want: number, tol: number) =>
  expect(Math.abs(got - want), `${got} vs ${want}`).toBeLessThanOrEqual(tol * (1 + Math.abs(want)))
const nearAll = (got: ArrayLike<number>, want: number[], tol: number) => {
  expect(got.length).toBe(want.length)
  want.forEach((w, i) => near(got[i], w, tol))
}

describe('linear programs match scipy linprog (HiGHS)', () => {
  for (const method of ['simplex', 'interior-point'] as const) {
    const names = method === 'simplex' ? F.simplex.problems : F.linearInteriorPoint.problems
    for (const name of names)
      it(`${method}: ${name}`, () => {
        const ref = F.lp[name]
        const r = linprog(ref.problem, { method })
        expect(r.status).toBe(ref.status)
        if (ref.status !== 'optimal') return
        const tol = method === 'simplex' ? 1e-9 : 1e-6
        near(r.objective, ref.objective!, tol)
        nearAll(toFlat(r.x), ref.x!, tol)
        const d = r.report!.duals
        nearAll(toFlat(d.ineq), ref.ineq!, tol)
        nearAll(toFlat(d.eq), ref.eq!, tol)
        nearAll(toFlat(d.lower), ref.lower!, tol)
        nearAll(toFlat(d.upper), ref.upper!, tol)
      })
  }
  it('the step-through algorithms end where linprog does', () => {
    for (const name of F.linearInteriorPoint.problems) {
      const ref = F.lp[name]
      const s = run(simplex(ref.problem), {}, 10_000)
      near(s.objective, ref.objective!, 1e-9)
      const ip = run(linearInteriorPoint(ref.problem), {}, 200)
      expect(ip.status).toBe('optimal')
      near(ip.objective, ref.objective!, 1e-6)
      // scipy's point and marginals certify themselves through dualityReport.
      const duals = {
        ineq: tensor(ref.ineq!),
        eq: tensor(ref.eq!),
        lower: tensor(ref.lower!),
        upper: tensor(ref.upper!),
      }
      const d = dualityReport(ref.problem, ref.x!, duals)
      expect(Math.abs(d.dualityGap)).toBeLessThan(1e-9 * (1 + Math.abs(ref.objective!)))
      expect(d.primalInfeasibility).toBeLessThan(1e-9)
      expect(d.dualInfeasibility).toBeLessThan(1e-9)
      expect(d.complementarity).toBeLessThan(1e-9)
    }
  })
})

describe('quadratic programs match the active-set enumeration (and SLSQP)', () => {
  for (const name of F.activeSet.problems) {
    const ref = F.qp[name]
    it(`active set: ${name}`, () => {
      const r = quadprog(ref.problem)
      expect(r.status).toBe('optimal')
      near(r.objective, ref.objective, 1e-10)
      nearAll(toFlat(r.x), ref.x, 1e-8)
      nearAll(toFlat(r.report.lambda), ref.lambda, 1e-7)
      nearAll(toFlat(r.report.nu), ref.nu, 1e-7)
      const s = run(activeSet(ref.problem), { x0: ref.start }, 1000)
      expect(s.status).toBe('optimal')
      nearAll(toFlat(s.x), ref.x, 1e-8)
    })
    it(`interior point: ${name}`, () => {
      const s = run(quadraticInteriorPoint(ref.problem), {}, 200)
      expect(s.status).toBe('optimal')
      near(s.objective, ref.objective, 1e-7)
      nearAll(toFlat(s.x), ref.x, 1e-6)
      nearAll(toFlat(s.lambda), ref.lambda, 1e-5)
    })
  }
  for (const [name, ref] of Object.entries(F.boxQuadraticProgram))
    it(`box QP: ${name}`, () => {
      const s = run(boxQuadraticProgram(ref.problem), {}, 1000)
      expect(s.converged).toBe(true)
      near(s.objective, ref.objective, 1e-10)
      nearAll(toFlat(s.x), ref.x, 1e-8)
    })
})

describe('integer programs match scipy milp', () => {
  for (const name of F.branchAndBound.problems) {
    const ref = F.milp[name]
    it(`branch and bound: ${name}`, () => {
      for (const strategy of ['depth-first', 'best-bound', 'breadth-first'] as const) {
        const r = milp(ref.problem, { strategy })
        expect(r.status).toBe('optimal')
        near(r.objective, ref.objective, 1e-9)
      }
      const s = run(branchAndBound(ref.problem), {}, 100_000)
      expect(s.status).toBe('optimal')
      near(s.incumbentValue, ref.objective, 1e-9)
    })
  }
  for (const name of F.gomory.problems) {
    const ref = F.milp[name]
    it(`Gomory cuts: ${name}`, () => {
      const s = run(gomory(ref.problem), {}, 500)
      expect(s.status).toBe('optimal')
      near(s.objective, ref.objective, 1e-7)
    })
  }
})

describe('assignment and dynamic programming', () => {
  for (const [name, ref] of Object.entries(F.hungarianSteps))
    it(`Hungarian matches linear_sum_assignment: ${name}`, () => {
      const r = hungarian(ref.cost, { maximize: ref.maximize })
      near(r.cost, ref.total, 1e-12)
      const s = run(hungarianSteps(ref.cost, { maximize: ref.maximize }), {}, 1_000_000)
      expect(s.converged).toBe(true)
      // The assignment itself is unique for continuous costs.
      if (name !== 'ties') {
        const a = toFlat(r.assignment)
        ref.rows.forEach((row, k) => expect(a[row]).toBe(ref.cols[k]))
      }
    })
  it('dynamicProgram fills the 0/1 knapsack table', () => {
    const { weights, values, capacity, table } = F.dynamicProgram
    const s = run(
      dynamicProgram({
        shape: [weights.length + 1, capacity + 1],
        cell: (i, w, get) =>
          i === 0
            ? 0
            : Math.max(get(i - 1, w), weights[i - 1] <= w ? get(i - 1, w - weights[i - 1]) + values[i - 1] : -Infinity),
      }),
      {},
      100,
    )
    expect(s.converged).toBe(true)
    expect(toFlat(s.table)).toEqual(table.flat())
  })
})
