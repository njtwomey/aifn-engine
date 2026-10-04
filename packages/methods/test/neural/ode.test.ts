/**
 * The neural ODE family: a ResNet of N blocks with tied weights is N Euler steps of the NODE; the reflection and the
 * disc-in-ring data; the streamed runs finish with consistent records; a CNF's density integrates to one and
 * Hutchinson's estimate is unbiased; the latent ODE's trajectories respect the observation window.
 */
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { traceProbe } from 'aifn-compute/dynamics/ode'
import {
  cnf,
  cnfRun,
  discInRing,
  latentOdeRun,
  odeModel,
  odeRun,
  reflectionData,
  trajectories,
} from 'aifn-methods/neural/ode'

const flat = (v: unknown) => Array.from(toFlat(v as Tensor))
const x = fromData(Float64Array.from([0.3, -0.7, 1.1, 0.4, -0.5, 0.2]), [3, 2])

describe('odeModel', () => {
  it('a ResNet with every block equal to the field is Euler’s method on the NODE', () => {
    const node = odeModel({ kind: 'node', dim: 2, classes: 2, solver: { method: 'euler', stepSize: 0.1 } })
    const resnet = odeModel({ kind: 'resnet', dim: 2, classes: 2, depth: 10 })
    const p = node.init(stream(1))
    const tied = { field: Array.from({ length: 10 }, () => p.field), readout: p.readout }
    const a = flat(node.flow(p, x, [0, 1])[1])
    const b = flat(resnet.flow(tied as never, x, [0, 1])[1])
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 12))
  })

  it('pads ANODE states with zeros and SONODE states with zero velocity; keeps the readout shapes', () => {
    for (const [kind, stateDim] of [
      ['node', 2],
      ['anode', 4],
      ['sonode', 4],
    ] as const) {
      const m = odeModel({ kind, dim: 2, classes: 3, augment: 2 })
      expect(m.stateDim).toBe(stateDim)
      const p = m.init(stream(2))
      const z0 = flat(m.lift(x))
      expect(z0.filter((_, i) => i % stateDim >= 2).every((v) => v === 0)).toBe(true)
      const zT = m.flow(p, x, [0, 0.5, 1]).at(-1)!
      expect((m.readout(p, zT) as Tensor).shape).toEqual([3, 3])
    }
  })
})

describe('data', () => {
  it('reflection pairs x with −x; the disc and the ring have their radii', () => {
    const r = reflectionData(11)
    flat(r.x).forEach((v, i) => expect(flat(r.y)[i]).toBe(-v))
    const d = discInRing(stream(3), 300)
    const xs = flat(d.x)
    flat(d.y!).forEach((label, i) => {
      const radius = Math.hypot(xs[2 * i], xs[2 * i + 1])
      if (label === 0) expect(radius).toBeLessThanOrEqual(0.5)
      else expect(radius >= 1 && radius <= 1.5).toBe(true)
    })
  })
})

describe('streamed runs', () => {
  it('odeRun records one entry per iteration and agrees with backprop through the adjoint', () => {
    let last
    for (const s of odeRun(discInRing(stream(4), 120), { steps: 4, checkpoints: 2, batchSize: 60 })) last = s
    expect(last!.finished).toBe(true)
    expect(last!.loss.length).toBe(5)
    expect(Array.from(last!.nfeForward.slice(1))).toEqual([40, 40, 40, 40])
    const g = last!.checkpoints.at(-1)!.gradient!
    expect(g.relativeError).toBeLessThan(1e-2)
    expect(g.adjointBackward).toBe(40)
  })

  it('a NODE cannot fit g(x) = −x in 1-d; an ANODE can', () => {
    const mse = (kind: 'node' | 'anode') => {
      let last
      for (const s of odeRun(reflectionData(20), { kind, task: 'regression', steps: 120, checkpoints: 1 })) last = s
      return last!.checkpoints.at(-1)!.metric
    }
    expect(mse('node')).toBeGreaterThan(0.2)
    expect(mse('anode')).toBeLessThan(0.05)
  })

  it('latentOdeRun finishes; observations lie in the window', () => {
    const set = trajectories(stream(5), { n: 8, grid: 20 })
    const mask = flat(set.mask)
    for (let i = 0; i < 8; i++) {
      const row = mask.slice(i * 20, (i + 1) * 20)
      expect(row.slice(0, 10).some((v) => v === 1)).toBe(true)
      expect(row.slice(10).every((v) => v === 0)).toBe(true)
    }
    let last
    for (const s of latentOdeRun({ n: 16, steps: 3, checkpoints: 1 })) last = s
    expect(last!.finished).toBe(true)
    expect(last!.checkpoints.at(-1)!.predictions.length).toBe(4 * 40)
  })
})

describe('cnf', () => {
  const model = cnf({ hidden: 16 })
  const params = model.init(stream(6))
  it('p₁ integrates to one over the plane (the change of variables preserves mass)', () => {
    const g = 48
    const box = 6
    const h = (2 * box) / g
    const pts = new Float64Array(g * g * 2)
    for (let i = 0; i < g; i++)
      for (let j = 0; j < g; j++) {
        pts[2 * (i * g + j)] = -box + (j + 0.5) * h
        pts[2 * (i * g + j) + 1] = -box + (i + 0.5) * h
      }
    const logp = flat(model.logDensity(params, fromData(pts, [g * g, 2])).logDensity)
    expect(logp.reduce((s, v) => s + Math.exp(v) * h * h, 0)).toBeCloseTo(1, 2)
  })

  it('Hutchinson’s estimate of log p is unbiased', () => {
    const exact = flat(model.logDensity(params, x).logDensity)
    const sum = [0, 0, 0]
    const m = 200
    for (let k = 0; k < m; k++) {
      const probe = traceProbe(child(stream(7), k), [3, 2])
      flat(model.logDensity(params, x, { estimator: 'hutchinson', probe }).logDensity).forEach(
        (v, i) => (sum[i] += v / m),
      )
    }
    sum.forEach((v, i) => expect(v).toBeCloseTo(exact[i], 1))
  })

  it('cnfRun finishes with density frames on its grid', () => {
    let last
    const data = {
      x: fromData(
        Float64Array.from({ length: 200 }, (_, i) => Math.sin(i * 1.3)),
        [100, 2],
      ),
    }
    for (const s of cnfRun(data, { steps: 2, checkpoints: 1, grid: 8, frames: 3, samples: 10 })) last = s
    const c = last!.checkpoints.at(-1)!
    expect(c.density.length).toBe(3)
    expect(c.density[0].length).toBe(64)
    expect(Number.isFinite(c.nll)).toBe(true)
  })
})
