import { describe, expect, it } from 'vitest'
import {
  classifyLinear,
  curl,
  divergence,
  fixedPoints,
  flowMap,
  gradientField,
  hamiltonianField,
  invariantManifolds,
  jacobianAt,
  linearise,
  sampleScalar,
  gridAxes,
  lyapunovDerivative,
  streamline,
  transportDensity,
  transportDensityFrames,
  pushForwardDensityFrames,
  type VectorField,
} from 'aifn-compute/dynamics/fields'
import {
  add,
  cos,
  get,
  mul,
  neg,
  sin,
  square,
  stack,
  sub,
  sum,
  tensor,
  toFlat,
  toRows,
  type Tensor,
} from 'aifn-compute/foundation/tensor'

const pendulum: VectorField = (x) => stack([get(x, 1), neg(sin(get(x, 0)))])
const rotation: VectorField = (x) => stack([neg(get(x, 1)), get(x, 0)])

describe('calculus', () => {
  it('computes Jacobian, divergence and curl', () => {
    const f: VectorField = (x) => stack([mul(get(x, 0), get(x, 1)), add(square(get(x, 0)), mul(3, get(x, 1)))])
    expect(toRows(jacobianAt(f, [2, 5]))).toEqual([
      [5, 2],
      [4, 3],
    ])
    expect(divergence(f, [2, 5])).toBeCloseTo(8, 12)
    expect(curl(f, [2, 5])).toBeCloseTo(4 - 2, 12)
    expect(curl(rotation, [0.3, 0.4])).toBeCloseTo(2, 12)
    expect(divergence(rotation, [0.3, 0.4])).toBeCloseTo(0, 12)
  })

  it('gradient fields are curl-free and Hamiltonian fields divergence-free', () => {
    const V = (x: Tensor) => add(mul(0.5, sum(square(x))), mul(get(x, 0), get(x, 1)))
    const g = gradientField(V)
    expect(toFlat(g(tensor([1, 2])) as Tensor)).toEqual([-3, -3])
    expect(curl(g, [0.7, -0.2])).toBeCloseTo(0, 12)
    const H = hamiltonianField((x: Tensor) => sub(mul(0.5, square(get(x, 1))), cos(get(x, 0))))
    expect(divergence(H, [0.4, 1.1])).toBeCloseTo(0, 12)
    const hv = toFlat(H(tensor([0.4, 1.1])) as Tensor)
    expect(hv[0]).toBeCloseTo(1.1, 14)
    expect(hv[1]).toBeCloseTo(-Math.sin(0.4), 14)
  })
})

describe('grids', () => {
  it('samples a scalar field on a grid, rows along y', () => {
    const grid = { x: [-1, 1] as [number, number], y: [0, 2] as [number, number], nx: 3, ny: 2 }
    const { x, y } = gridAxes(grid)
    expect(toFlat(x)).toEqual([-1, 0, 1])
    expect(toFlat(y)).toEqual([0, 2])
    const s = sampleScalar((p) => add(get(p, 0), mul(10, get(p, 1))), grid)
    expect(toRows(s.values)).toEqual([
      [-1, 0, 1],
      [19, 20, 21],
    ])
  })
})

describe('fixed points', () => {
  it('classifies canonical linear systems', () => {
    expect(
      classifyLinear([
        [-1, 0],
        [0, -2],
      ]).kind,
    ).toBe('stable node')
    expect(
      classifyLinear([
        [1, 0],
        [0, -2],
      ]).kind,
    ).toBe('saddle')
    expect(
      classifyLinear([
        [-0.1, 1],
        [-1, -0.1],
      ]).kind,
    ).toBe('stable spiral')
    expect(
      classifyLinear([
        [0.1, 1],
        [-1, 0.1],
      ]).kind,
    ).toBe('unstable spiral')
    expect(
      classifyLinear([
        [0, 1],
        [-1, 0],
      ]).kind,
    ).toBe('centre')
    expect(
      classifyLinear([
        [-1, 0],
        [0, -1],
      ]).kind,
    ).toBe('star')
    expect(
      classifyLinear([
        [-1, 1],
        [0, -1],
      ]).kind,
    ).toBe('stable degenerate node')
    expect(
      classifyLinear([
        [0, 0],
        [0, -1],
      ]).kind,
    ).toBe('non-hyperbolic')
    const three = classifyLinear([
      [-1, 0, 0],
      [0, -2, 1],
      [0, -1, -2],
    ])
    expect(three.kind).toBe('stable spiral')
    expect(three.determinant).toBeCloseTo(-5, 10)
  })

  it('finds the pendulum’s equilibria', () => {
    const fps = fixedPoints(pendulum, [
      [-4, 4],
      [-2, 2],
    ])
    expect(fps.map((p) => toFlat(p.point)[0])).toEqual([
      expect.closeTo(-Math.PI, 10),
      expect.closeTo(0, 10),
      expect.closeTo(Math.PI, 10),
    ])
    expect(fps.map((p) => p.kind)).toEqual(['saddle', 'centre', 'saddle'])
  })

  it('traces a saddle’s manifolds along its eigenvectors', () => {
    const f: VectorField = (x) => stack([get(x, 0), neg(get(x, 1))])
    const m = invariantManifolds(f, linearise(f, [0, 0]), {
      t: 15,
      bounds: [
        [-2, 2],
        [-2, 2],
      ],
    })
    expect(m.unstable).toHaveLength(2)
    expect(m.stable).toHaveLength(2)
    m.unstable.forEach((c) => toRows(c).forEach(([, y]) => expect(Math.abs(y)).toBeLessThan(1e-12)))
    m.stable.forEach((c) => toRows(c).forEach(([x]) => expect(Math.abs(x)).toBeLessThan(1e-12)))
    // Stable branches run from the fixed point outwards.
    expect(toRows(m.stable[0])[0]).toEqual([0, 0])
    expect(Math.abs(toRows(m.stable[0]).at(-1)![1])).toBeGreaterThan(1)
  })

  it('the orbital derivative of a Lyapunov function', () => {
    const damped: VectorField = (x) => stack([get(x, 1), sub(neg(get(x, 0)), mul(0.5, get(x, 1)))])
    const V = (x: Tensor) => sum(square(x))
    // V̇ = 2x·y + 2y·(−x − y/2) = −y².
    expect(lyapunovDerivative(V, damped, [0.3, -0.7])).toBeCloseTo(-0.49, 12)
    expect(lyapunovDerivative(V, damped, [0.8, 0])).toBeCloseTo(0, 12)
  })
})

describe('flows', () => {
  it('flow maps and streamlines follow the rotation', () => {
    const x = toFlat(flowMap(rotation, [1, 0], Math.PI / 2))
    expect(x[0]).toBeCloseTo(0, 7)
    expect(x[1]).toBeCloseTo(1, 7)
    const s = streamline(rotation, [1, 0], { t: 2 * Math.PI, steps: 400 })
    toRows(s).forEach(([a, b]) => expect(Math.hypot(a, b)).toBeCloseTo(1, 6))
    const bounded = streamline((p) => p, [0.1, 0], {
      direction: 'forward',
      bounds: [
        [-1, 1],
        [-1, 1],
      ],
    })
    expect(toRows(bounded).at(-1)![0]).toBeGreaterThan(1)
    expect(toRows(bounded).at(-2)![0]).toBeLessThanOrEqual(1)
  })

  it('transports a density by Liouville’s equation', () => {
    const a = 0.7
    const f: VectorField = (x) => mul(a, x)
    const rho0 = (x: Tensor) => Math.exp(-0.5 * toFlat(x)[0] ** 2) / Math.sqrt(2 * Math.PI)
    const pts = [[-1], [0], [0.5], [2]]
    const t = 1.3
    const rho = toFlat(transportDensity(f, rho0, pts, t))
    pts.forEach(([x], k) => {
      const exact = rho0(tensor([x * Math.exp(-a * t)])) * Math.exp(-a * t)
      expect(rho[k]).toBeCloseTo(exact, 8)
    })
  })

  it('transports a density at every frame, matching a single-time transport', () => {
    const a = 0.7
    const f: VectorField = (x) => mul(a, x)
    const rho0 = (x: Tensor) => Math.exp(-0.5 * toFlat(x)[0] ** 2) / Math.sqrt(2 * Math.PI)
    const pts = [[-1], [0.5], [2]]
    const frames = toRows(transportDensityFrames(f, rho0, pts, 1.2, { frames: 4, steps: 10, divergence: () => a }))
    expect(frames.length).toBe(5)
    frames.forEach((row, j) => {
      const t = (1.2 * j) / 4
      pts.forEach(([x], k) => expect(row[k]).toBeCloseTo(rho0(tensor([x * Math.exp(-a * t)])) * Math.exp(-a * t), 7))
    })
  })

  it('pushes samples forward at every frame with their log-volume change', () => {
    const f: VectorField = (x) => mul(-0.5, x)
    const r = pushForwardDensityFrames(f, [[1, 2]], 2, { frames: 2, steps: 20, divergence: () => -1 })
    const x = toFlat(r.x)
    expect(x[4]).toBeCloseTo(Math.exp(-1), 8)
    expect(x[5]).toBeCloseTo(2 * Math.exp(-1), 8)
    expect(toFlat(r.logVolume)).toEqual([0, -1, -2].map((v) => expect.closeTo(v, 12)))
  })
})
