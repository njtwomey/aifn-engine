import { describe, expect, it } from 'vitest'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { treeLeaves } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, norm, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { barsAndStripes, moons } from 'aifn-methods/data/synthetic'
import {
  autoencoder,
  autoencoderLoss,
  autoencoderRun,
  initAutoencoder,
  type AutoencoderParams,
} from 'aifn-methods/generative/autoencoders'
import {
  flowForward,
  flowLogDensityValues,
  flowSample,
  initRealNvp,
  realNvp,
  realNvpRun,
} from 'aifn-methods/generative/flows'
import {
  capacityCurve,
  corruptPattern,
  freeEnergy,
  hebbianWeights,
  hopfieldRecall,
  logPartition,
  modernHopfieldUpdate,
  rbm,
  rbmRun,
} from 'aifn-methods/generative/boltzmann'

const data = moons(stream(1), { n: 300, noise: 0.08 })

describe('autoencoders', () => {
  it('a VAE lowers its negative ELBO, and the VQ-VAE passes gradients to the encoder straight through', () => {
    let first: number | undefined
    let last: { history: { total: number[] } } | undefined
    for (const r of autoencoderRun(data, { kind: 'vae', steps: 300, checkpoints: 3 })) {
      first ??= r.history.total[0]
      last = r
    }
    expect(last!.history.total.at(-1)!).toBeLessThan(first! / 2)
    const vq = autoencoder({ kind: 'vqvae', inputs: 2, latent: 2, codes: 8 })
    const p = initAutoencoder(vq, stream(2))
    const result: { grad: unknown } = valueAndGrad(
      (q: AutoencoderParams) => autoencoderLoss(vq, q, data.x, stream(3)).loss,
      {},
    )(p)
    const g = result.grad as AutoencoderParams
    const encoderNorm = Math.hypot(...treeLeaves(g.encoder).map((l) => norm(l.value as Tensor)))
    expect(encoderNorm).toBeGreaterThan(0)
    expect(norm(g.codebook)).toBeGreaterThan(0)
  })
})

describe('RealNVP', () => {
  it('starts as the base density, integrates to 1 and inverts its samples', () => {
    const flow = realNvp({ layers: 4 })
    const p = initRealNvp(flow, stream(4))
    const g = 121
    const h = 12 / (g - 1)
    const pts = new Float64Array(g * g * 2)
    for (let i = 0; i < g; i++)
      for (let j = 0; j < g; j++) {
        pts[2 * (i * g + j)] = -6 + j * h
        pts[2 * (i * g + j) + 1] = -6 + i * h
      }
    const mass = (q: typeof p) =>
      flowLogDensityValues(flow, q, fromData(pts, [g * g, 2])).reduce((a, v) => a + Math.exp(v) * h * h, 0)
    expect(mass(p)).toBeCloseTo(1, 3)
    let last: { nll: { value: number[] } } | undefined
    for (const r of realNvpRun(data, { layers: 4, steps: 150, checkpoints: 1 })) last = r
    expect(last!.nll.value.at(-1)!).toBeLessThan(last!.nll.value[0])
    const z = fromData(Float64Array.from([0.3, -1.2, 1.5, 0.4]), [2, 2])
    const back = flowForward(flow, p, flowSample(flow, p, z)).at(-1)!
    toFlat(back).forEach((v, i) => expect(v).toBeCloseTo(toFlat(z)[i], 10))
  })
})

describe('Boltzmann machines and Hopfield networks', () => {
  it('sums the RBM partition function exactly', () => {
    const m = rbm(stream(5), 3, 2)
    const W = m.W.map((_, q) => 0.4 * Math.sin(q + 1))
    const model = { ...m, W, a: Float64Array.of(0.1, -0.3, 0.2), b: Float64Array.of(0.5, -0.2) }
    let Z = 0
    for (let v = 0; v < 8; v++) {
      const vis = [v & 1, (v >> 1) & 1, (v >> 2) & 1]
      Z += Math.exp(-freeEnergy(model, vis))
    }
    expect(logPartition(model)).toBeCloseTo(Math.log(Z), 12)
  })
  it('learns bars and stripes by CD-1', () => {
    let last: { logLikelihood: number[] } | undefined
    for (const r of rbmRun({ x: barsAndStripes({ size: 3 }) }, { hidden: 8, epochs: 400, seed: 1 })) last = r
    const ll = last!.logLikelihood
    expect(ll.at(-1)!).toBeGreaterThan(ll[0] + 1.5)
  })
  it('recalls stored patterns, never raising the energy; the modern update recalls past classical capacity', () => {
    const u = Array.from({ length: 3 * 40 }, (_, i) => (Math.sin(17.3 * i + 1) > 0 ? 1 : -1))
    const X = fromData(Float64Array.from(u), [3, 40])
    const W = hebbianWeights(X)
    const cue = corruptPattern(u.slice(0, 40), stream(6), 0.15)
    const r = hopfieldRecall(W, cue, stream(7))
    for (let i = 1; i < r.energies.length; i++) expect(r.energies[i]).toBeLessThanOrEqual(r.energies[i - 1] + 1e-12)
    expect(Array.from(r.states.at(-1)!)).toEqual(u.slice(0, 40))
    const m = modernHopfieldUpdate(X, cue, 1)
    expect(m.weights[0]).toBeGreaterThan(0.99)
    const curve = capacityCurve(child(stream(8), 'c'), { units: 40, maxPatterns: 16, trials: 4 })
    expect(curve.modern.at(-1)!).toBeGreaterThan(curve.classical.at(-1)!)
  })
})
