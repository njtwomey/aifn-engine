import { describe, expect, it } from 'vitest'
import { ceemdan, eemd, emd, vmd, vmdModes, vmdSteps } from 'aifn-compute/signal/decompositions'
import { signal } from 'aifn-compute/signal'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { close, F } from '../helpers'
import { checkProtocol } from '../../protocol'

const V = F.vmd

describe('variational mode decomposition', () => {
  it.each(Object.keys(V.cases))('matches the reference code (vmdpy port in numpy) sweep for sweep (%s)', (key) => {
    const c = V.cases[key]
    const alg = vmdSteps(V.x, { modes: c.K, alpha: c.alpha, tau: c.tau, dc: c.dc })
    const tr = trace(alg, undefined, 1000)
    expect(tr.final.converged).toBe(true)
    expect(tr.final.t).toBe(c.iterations)
    // The reference returns the iterate before its last sweep (u_hat_plus[n − 1]).
    const before = tr.steps[c.iterations - 1]
    close(before.omega, c.omega, 1e-9)
    vmdModes(before).forEach((m, k) => close(Array.from(m), c.u[k], 1e-9))
  })

  it('separates three tones and reports their centre frequencies in Hz', () => {
    const d = vmd(signal(V.x, { fs: 1000 }), { modes: 3 })
    expect(d.kind).toBe('decomposition')
    expect(d.method).toBe('vmd')
    const centres = d.components.map((c) => (c.meta as { centreFrequency: number }).centreFrequency)
    expect(centres[0]).toBeCloseTo(2, 1)
    expect(centres[1]).toBeCloseTo(24, 1)
    expect(centres[2]).toBeCloseTo(288, 0)
    // The slowest mode is the unit-amplitude 2 Hz cosine; the residual is the noise the narrow bands leave out.
    const t = Array.from({ length: 1000 }, (_, i) => i / 1000)
    close(
      d.components[0].values,
      t.map((s) => Math.cos(2 * Math.PI * 2 * s)),
      0.05,
    )
    const r = toFlat(d.residual!)
    expect(Math.sqrt(r.reduce((a, b) => a + b * b, 0) / r.length)).toBeLessThan(0.03)
  })

  it('is a traceable algorithm and keeps the DC mode at zero frequency', () => {
    checkProtocol(vmdSteps(V.x, { modes: 2 }), undefined, { steps: 6 })
    const s = run(vmdSteps(V.x, { modes: 3, dc: true }), undefined, 50)
    expect(toFlat(s.omega)[0]).toBe(0)
    const r = run(vmdSteps(V.x, { modes: 3, init: stream('vmd') }), undefined, 300)
    expect(r.converged).toBe(true)
  })
})

describe('CEEMDAN', () => {
  const n = 512
  const t = Array.from({ length: n }, (_, i) => i / n)
  // An intermittent fast burst on a slow tone: the classic mode-mixing example.
  const x = t.map(
    (s) => Math.sin(2 * Math.PI * 4 * s) + (s > 0.4 && s < 0.6 ? 0.3 * Math.sin(2 * Math.PI * 60 * s) : 0),
  )

  it('is complete: the IMFs and residue sum to the signal exactly (EEMD does not)', () => {
    const d = ceemdan(stream('ceemdan'), x, { trials: 20, maxImfs: 6 })
    expect(d.method).toBe('ceemdan')
    const sum = new Float64Array(n)
    for (const c of d.components) toFlat(c.values).forEach((v, i) => (sum[i] += v))
    toFlat(d.residual!).forEach((v, i) => (sum[i] += v))
    close(Array.from(sum), x, 1e-12)
    const e = eemd(stream('ceemdan'), x, { trials: 20, maxImfs: 6 })
    const esum = new Float64Array(n)
    for (const c of e.components) toFlat(c.values).forEach((v, i) => (esum[i] += v))
    toFlat(e.residual!).forEach((v, i) => (esum[i] += v))
    expect(Math.max(...Array.from(esum, (v, i) => Math.abs(v - x[i])))).toBeGreaterThan(1e-3)
  })

  it('reduces mode mixing: one mode follows the slow tone, where plain EMD splits it around the burst', () => {
    const slow = t.map((s) => Math.sin(2 * Math.PI * 4 * s))
    // The worst error of each mode against the 4 Hz tone, away from the ends.
    const best = (components: readonly { values: Parameters<typeof toFlat>[0] }[]) =>
      Math.min(
        ...components.map((c) => {
          const v = toFlat(c.values)
          let e = 0
          for (let i = 50; i < n - 50; i++) e = Math.max(e, Math.abs(v[i] - slow[i]))
          return e
        }),
      )
    // EMD's first mode jumps between the burst and the tone (intermittency), so no mode is the tone (error ≈ 1).
    expect(best(emd(x).components)).toBeGreaterThan(0.9)
    expect(best(ceemdan(stream('ceemdan'), x, { trials: 30 }).components)).toBeLessThan(0.45)
  })

  it('is reproducible from its stream and differs between streams', () => {
    const a = ceemdan(stream('a'), x, { trials: 5, maxImfs: 2 })
    const b = ceemdan(stream('a'), x, { trials: 5, maxImfs: 2 })
    const c = ceemdan(stream('b'), x, { trials: 5, maxImfs: 2 })
    expect(toFlat(a.components[0].values)).toEqual(toFlat(b.components[0].values))
    expect(toFlat(a.components[0].values)).not.toEqual(toFlat(c.components[0].values))
  })
})
