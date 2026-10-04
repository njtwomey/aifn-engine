import { describe, expect, it } from 'vitest'
import {
  bisection,
  brent,
  broyden,
  continuation,
  fixedPoint,
  minimizeScalar,
  newtonHomotopy,
  newtonRoot,
  newtonSystem,
  regulaFalsi,
  secant,
  type SystemWithJacobian,
} from 'aifn-compute/numerics/roots'
import { logGamma } from 'aifn-compute/numerics/special'
import { run } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'

type Fixture = {
  scalar: Record<
    string,
    { lo: number; hi: number; x0: number; root: number; brentqIterations: number; brentqCalls: number }
  >
  bisection: { functions: string[] }
  regulaFalsi: { functions: string[] }
  brent: { functions: string[] }
  secant: { functions: string[] }
  newtonRoot: { functions: string[] }
  systems: Record<string, { x0: number[]; root: number[] }>
  newtonSystem: { systems: string[] }
  broyden: { systems: string[] }
  continuation: { systems: string[] }
  fixedPoint: Record<string, { x0: number[]; point: number[] }>
  minimizeScalar: Record<
    string,
    { bracket: [number, number]; bounds: [number, number]; x: number; value: number; boundedX: number }
  >
}
const F = fixture<Fixture>('numerics/roots')

/** The scalar functions of the generator (gen/numerics/roots.py), with derivatives for Newton. */
const scalar: Record<string, { f: (x: number) => number; df: (x: number) => number }> = {
  cubic: { f: (x) => x ** 3 - 2 * x - 5, df: (x) => 3 * x * x - 2 },
  cosMinusX: { f: (x) => Math.cos(x) - x, df: (x) => -Math.sin(x) - 1 },
  kepler: { f: (x) => x - 0.9 * Math.sin(x) - 0.5, df: (x) => 1 - 0.9 * Math.cos(x) },
  lambert: { f: (x) => x * Math.exp(x) - 3, df: (x) => (1 + x) * Math.exp(x) },
  logShift: { f: (x) => Math.log(x) + x - 2, df: (x) => 1 / x + 1 },
  tanhShift: { f: (x) => Math.tanh(x - 0.3), df: (x) => 1 - Math.tanh(x - 0.3) ** 2 },
  wilkinson: {
    f: (x) => (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5),
    df: (x) => {
      let s = 0
      for (let k = 1; k <= 5; k++) {
        let p = 1
        for (let j = 1; j <= 5; j++) if (j !== k) p *= x - j
        s += p
      }
      return s
    },
  },
  steepExp: { f: (x) => Math.exp(10 * x) - 1e3, df: (x) => 10 * Math.exp(10 * x) },
}

type V = { data: ArrayLike<number> & Iterable<number> }
/** The systems of the generator with their Jacobians. */
const systems: Record<string, (x: V) => { value: number[]; jacobian: number[][] }> = {
  circleLine: ({ data: [x, y] }) => ({
    value: [x * x + y * y - 4, x - y - 0.5],
    jacobian: [
      [2 * x, 2 * y],
      [1, -1],
    ],
  }),
  rosenbrockGradient: ({ data: [x, y] }) => ({
    value: [10 * (y - x * x), 1 - x],
    jacobian: [
      [-20 * x, 10],
      [-1, 0],
    ],
  }),
  burdenFaires: ({ data: [x, y, z] }) => ({
    value: [
      3 * x - Math.cos(y * z) - 0.5,
      x * x - 81 * (y + 0.1) ** 2 + Math.sin(z) + 1.06,
      Math.exp(-x * y) + 20 * z + (10 * Math.PI - 3) / 3,
    ],
    jacobian: [
      [3, z * Math.sin(y * z), y * Math.sin(y * z)],
      [2 * x, -162 * (y + 0.1), Math.cos(z)],
      [-y * Math.exp(-x * y), -x * Math.exp(-x * y), 20],
    ],
  }),
  broydenTridiagonal: ({ data }) => {
    const n = data.length
    const value: number[] = []
    const jacobian = Array.from({ length: n }, () => new Array<number>(n).fill(0))
    for (let i = 0; i < n; i++) {
      const x = data[i]
      value.push((3 - 2 * x) * x - (i > 0 ? data[i - 1] : 0) - 2 * (i < n - 1 ? data[i + 1] : 0) + 1)
      jacobian[i][i] = 3 - 4 * x
      if (i > 0) jacobian[i][i - 1] = -1
      if (i < n - 1) jacobian[i][i + 1] = -2
    }
    return { value, jacobian }
  },
}
const fixedMaps: Record<string, (x: V) => number[]> = {
  dottie: ({ data: [x] }) => [Math.cos(x)],
  planar: ({ data: [x, y] }) => [0.5 * Math.cos(y), 0.5 * Math.sin(x) + 0.2],
}
const minima: Record<string, (x: number) => number> = {
  quartic: (x) => (x - 1.3) ** 4 + 0.5 * (x - 1.3) ** 2 - 2,
  sinPlus: (x) => Math.sin(x) + 0.1 * x,
  gammaLog: (x) => logGamma(x) as number,
}

const near = (got: number, want: number, tol: number) =>
  expect(Math.abs(got - want), `${got} vs ${want}`).toBeLessThanOrEqual(tol * (1 + Math.abs(want)))
const nearAll = (got: ArrayLike<number>, want: number[], tol: number) => want.forEach((w, i) => near(got[i], w, tol))

describe('scalar roots match scipy.optimize.brentq', () => {
  for (const name of F.brent.functions) {
    const ref = F.scalar[name]
    const { f, df } = scalar[name]
    const bracket = { lo: ref.lo, hi: ref.hi }
    it(`${name}: bracketing methods`, () => {
      for (const s of [
        run(bisection(f), bracket, 500),
        run(regulaFalsi(f), bracket, 500),
        run(brent(f), bracket, 500),
      ]) {
        expect(s.converged).toBe(true)
        near(s.x, ref.root, 1e-11)
      }
    })
    it(`${name}: Brent takes as many steps as brentq`, () => {
      const s = run(brent(f, { xtol: 1e-15 }), bracket, 500)
      near(s.x, ref.root, 1e-14)
      expect(Math.abs(s.evaluations - ref.brentqCalls)).toBeLessThanOrEqual(1)
    })
    it(`${name}: open methods from x0`, () => {
      const n = run(
        newtonRoot((x) => ({ value: f(x), derivative: df(x) }), { damped: true }),
        { x0: ref.x0 },
        200,
      )
      expect(n.converged).toBe(true)
      near(n.x, ref.root, 1e-11)
      const s = run(secant(f), { x0: ref.x0, x1: ref.x0 + 1e-3 * (ref.hi - ref.lo) }, 200)
      expect(s.converged).toBe(true)
      near(s.x, ref.root, 1e-11)
    })
  }
})

describe('systems match scipy.optimize.root (hybrd)', () => {
  for (const name of F.newtonSystem.systems) {
    const ref = F.systems[name]
    const sys = systems[name] as unknown as SystemWithJacobian
    it(`${name}: Newton, damped Newton, Broyden and continuation`, () => {
      for (const damped of [false, true]) {
        const s = run(newtonSystem(sys, { damped }), { x0: ref.x0 }, 200)
        expect(s.converged).toBe(true)
        nearAll(s.x.data, ref.root, 1e-10)
      }
      const b = run(
        broyden((x) => systems[name](x).value),
        { x0: ref.x0 },
        500,
      )
      expect(b.converged).toBe(true)
      nearAll(b.x.data, ref.root, 1e-9)
      const c = run(continuation(newtonHomotopy(sys, ref.x0), { tolerance: 1e-12 }), { x0: ref.x0 }, 500)
      expect(c.converged).toBe(true)
      nearAll(c.x.data, ref.root, 1e-9)
    })
  }
  for (const [name, ref] of Object.entries(F.fixedPoint))
    it(`fixed point ${name} matches scipy.optimize.fixed_point`, () => {
      const s = run(fixedPoint(fixedMaps[name], { ftol: 1e-15 }), { x0: ref.x0 }, 5000)
      expect(s.converged).toBe(true)
      nearAll(s.x.data, ref.point, 1e-13)
    })
})

describe('minimizeScalar matches scipy.optimize.minimize_scalar', () => {
  for (const [name, ref] of Object.entries(F.minimizeScalar))
    it(name, () => {
      const r = minimizeScalar(minima[name], { bracket: ref.bracket, tolerance: 1e-12 })
      expect(r.converged).toBe(true)
      near(r.value, ref.value, 1e-14)
      near(r.x, ref.x, 1e-6)
      const b = minimizeScalar(minima[name], { bounds: ref.bounds, tolerance: 1e-12 })
      near(b.x, ref.boundedX, 1e-6)
    })
})
