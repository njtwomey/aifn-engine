import { describe, expect, it } from 'vitest'
import {
  densityEvolution,
  fokkerPlanck,
  gridPoints,
  heatEquation,
  transportEquation,
  waveEquation,
} from 'aifn-methods/dynamics/pde'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'

const unit = { a: 0, b: 1, n: 51 }
const xs = toFlat(gridPoints(unit))
const maxError = (u: number[], exact: (x: number) => number) => Math.max(...u.map((v, i) => Math.abs(v - exact(xs[i]))))

describe('heat equation', () => {
  const D = 0.1
  const exact = (t: number) => (x: number) => Math.exp(-D * Math.PI ** 2 * t) * Math.sin(Math.PI * x)
  it.each(['explicit', 'implicit', 'crank-nicolson'] as const)('%s matches the analytic sine mode', (scheme) => {
    const dt = 0.001
    const s = run(
      heatEquation({
        diffusivity: D,
        grid: unit,
        boundary: { kind: 'dirichlet', left: 0, right: 0 },
        dt,
        scheme,
        tEnd: 0.5,
      }),
      { u0: (x) => Math.sin(Math.PI * x) },
      10_000,
    )
    expect(s.time).toBeCloseTo(0.5, 10)
    expect(s.t).toBe(500)
    expect(maxError(toFlat(s.u), exact(0.5))).toBeLessThan(scheme === 'crank-nicolson' ? 2e-4 : 1e-3)
    expect(s.stability.stable).toBe(true)
  })

  it('Crank–Nicolson is second order in time', () => {
    const err = (dt: number) =>
      maxError(
        toFlat(
          run(
            heatEquation({
              diffusivity: D,
              grid: { a: 0, b: 1, n: 201 },
              boundary: { kind: 'dirichlet' },
              dt,
              tEnd: 0.4,
            }),
            { u0: (x) => Math.sin(Math.PI * x) },
            10_000,
          ).u,
        ).filter((_, i) => i % 4 === 0),
        exact(0.4),
      )
    // Space error is small on the fine grid, so halving dt quarters the error.
    expect(Math.log2(err(0.1) / err(0.05))).toBeCloseTo(2, 0)
  })

  it('reports and shows the explicit instability past r = ½', () => {
    const alg = heatEquation({
      diffusivity: 1,
      grid: unit,
      boundary: { kind: 'dirichlet' },
      dt: 0.0003,
      scheme: 'explicit',
    })
    const tr = trace(alg, { u0: (x) => Math.sin(Math.PI * x) + 0.01 * Math.sin(49 * Math.PI * x) }, 2000)
    expect(tr.steps[0].stability.value).toBeCloseTo(0.75, 10)
    expect(tr.steps[0].stability.stable).toBe(false)
    expect(Math.max(...toFlat(tr.steps.at(-1)!.u).map(Math.abs))).toBeGreaterThan(10)
  })

  it('conserves mass with zero-flux ends', () => {
    const s0 = heatEquation({
      diffusivity: 0.5,
      grid: unit,
      boundary: { kind: 'neumann' },
      dt: 0.01,
      scheme: 'implicit',
    })
    const tr = trace(s0, { u0: (x) => Math.exp(-50 * (x - 0.3) ** 2) }, 400, { record: { mass: (s) => s.mass } })
    const mass = toFlat(tr.series.mass)
    mass.forEach((m) => expect(m).toBeCloseTo(mass[0], 12))
    // Tends to the uniform profile with the same mass.
    toFlat(tr.steps.at(-1)!.u).forEach((v) => expect(v).toBeCloseTo(mass[0], 3))
  })
})

describe('transport equation', () => {
  const bump = (x: number) => Math.exp(-100 * (x - 0.3) ** 2)
  it.each(['upwind', 'lax-friedrichs', 'lax-wendroff'] as const)('%s moves a bump at speed c', (scheme) => {
    const s = run(
      transportEquation({ velocity: 1, grid: { a: 0, b: 1, n: 201 }, dt: 0.004, scheme, tEnd: 0.4 }),
      { u0: bump },
      1000,
    )
    const u = toFlat(s.u)
    const peak = u.indexOf(Math.max(...u)) / 200
    expect(peak).toBeCloseTo(0.7, 1)
    expect(s.stability.value).toBeCloseTo(0.8, 10)
  })

  it('Lax–Wendroff is more accurate than upwind, and ν = 1 upwind is exact', () => {
    const grid = { a: 0, b: 1, n: 201 }
    const exact = (x: number) => bump((((x - 0.4) % 1) + 1) % 1)
    const pts = toFlat(gridPoints(grid))
    const err = (scheme: 'upwind' | 'lax-wendroff', dt: number) => {
      const u = toFlat(run(transportEquation({ velocity: 1, grid, dt, scheme, tEnd: 0.4 }), { u0: bump }, 1000).u)
      return Math.max(...u.map((v, i) => Math.abs(v - exact(pts[i]))))
    }
    expect(err('lax-wendroff', 0.004)).toBeLessThan(err('upwind', 0.004) / 3)
    expect(err('upwind', 0.005)).toBeLessThan(1e-12)
  })

  it('blows up past the CFL limit', () => {
    const tr = trace(
      transportEquation({ velocity: 1, grid: unit, dt: 0.03, scheme: 'lax-wendroff' }),
      { u0: bump },
      400,
    )
    expect(tr.steps[0].stability.stable).toBe(false)
    expect(Math.max(...toFlat(tr.steps.at(-1)!.u).map(Math.abs))).toBeGreaterThan(1e3)
  })
})

describe('wave equation', () => {
  it('is exact at ν = 1 and conserves energy', () => {
    const grid = { a: 0, b: 1, n: 101 }
    const pts = toFlat(gridPoints(grid))
    const u0 = (x: number) => Math.sin(Math.PI * x)
    const tr = trace(waveEquation({ speed: 1, grid, dt: 0.01 }), { u0 }, 100, { record: { e: (s) => s.energy } })
    // Standing wave: u(x, t) = sin(πx) cos(πt).
    const u = toFlat(tr.steps.at(-1)!.u)
    u.forEach((v, i) => expect(v).toBeCloseTo(Math.sin(Math.PI * pts[i]) * Math.cos(Math.PI * 1), 10))
    const e = toFlat(tr.series.e)
    e.forEach((v) => expect(Math.abs(v - e[0]) / e[0]).toBeLessThan(0.02))
    const bad = trace(
      waveEquation({ speed: 1, grid, dt: 0.0102 }),
      { u0: (x) => u0(x) + 1e-3 * Math.sin(99 * Math.PI * x) },
      3000,
    )
    expect(bad.steps[0].stability.stable).toBe(false)
    expect(bad.meta.stopped).toBe('diverged')
  })
})

describe('Fokker–Planck', () => {
  it('relaxes to the Ornstein–Uhlenbeck stationary density and conserves mass', () => {
    // dX = −θX dt + σ dW: stationary N(0, σ²/(2θ)).
    const [theta, sigma] = [1, 1]
    const grid = { a: -4, b: 4, n: 161 }
    const pts = toFlat(gridPoints(grid))
    const alg = fokkerPlanck({ drift: (x) => -theta * x, diffusion: () => sigma ** 2 / 2, grid, dt: 0.01 })
    const s0 = alg.init({ u0: (x) => Math.exp(-((x - 2) ** 2) / 0.02) / Math.sqrt(0.02 * Math.PI) }, stream(0))
    const s = run(alg, { u0: (x) => Math.exp(-((x - 2) ** 2) / 0.02) / Math.sqrt(0.02 * Math.PI) }, 1000)
    expect(s.mass).toBeCloseTo(s0.mass, 12)
    const v = sigma ** 2 / (2 * theta)
    const p = toFlat(s.u)
    p.forEach((pi, i) => expect(pi).toBeCloseTo(Math.exp(-(pts[i] ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v), 2))
    expect(Math.min(...p)).toBeGreaterThanOrEqual(0)
  })

  it('tracks the OU mean exactly enough at intermediate times', () => {
    const grid = { a: -3, b: 5, n: 321 }
    const pts = toFlat(gridPoints(grid))
    const alg = fokkerPlanck({
      drift: (x) => -x,
      diffusion: () => 0.1,
      grid,
      dt: 0.001,
      scheme: 'crank-nicolson',
      tEnd: 1,
    })
    const s = run(alg, { u0: (x) => Math.exp(-((x - 2) ** 2) / 0.02) / Math.sqrt(0.02 * Math.PI) }, 5000)
    const dx = pts[1] - pts[0]
    const mean = toFlat(s.u).reduce((acc, p, i) => acc + p * pts[i] * dx, 0)
    expect(mean).toBeCloseTo(2 * Math.exp(-1), 2)
  })
})

describe('trace protocol', () => {
  it('every solver follows the protocol', () => {
    const u0 = (x: number) => Math.cos(2 * Math.PI * x)
    for (const scheme of ['explicit', 'implicit', 'crank-nicolson'] as const)
      expectProtocol(
        heatEquation({ diffusivity: 0.1, grid: unit, boundary: { kind: 'periodic' }, dt: 0.001, scheme }),
        { u0 },
        { n: 30 },
      )
    expectProtocol(transportEquation({ velocity: 1, grid: unit, dt: 0.005 }), { u0 }, { n: 30 })
    expectProtocol(waveEquation({ speed: 1, grid: unit, dt: 0.005 }), { u0 }, { n: 30 })
    expectProtocol(
      fokkerPlanck({ drift: (x) => -x, diffusion: () => 0.5, grid: { a: -4, b: 4, n: 81 }, dt: 0.01 }),
      { u0: (x) => Math.exp(-x * x) },
      { n: 30 },
    )
  })
})

describe('densityEvolution', () => {
  it('densityEvolution relaxes to the OU stationary density', () => {
    const alg = densityEvolution({ drift: (x) => -x, sigma: () => 1, grid: { a: -4, b: 4, n: 161 }, dt: 0.01 })
    const s = run(alg, { u0: (x) => Math.exp(-((x - 1) ** 2) * 50) * Math.sqrt(50 / Math.PI) }, 800)
    expect(Math.max(...toFlat(s.u))).toBeCloseTo(1 / Math.sqrt(Math.PI), 2)
  })
})
