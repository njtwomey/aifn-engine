/**
 * The RQ-VAE: its spec and codebooks; quantisation against residual quantisation done by hand; the loss's parts
 * against their formulas and its gradients through the straight-through estimator; a training run that learns its
 * stages and records the hierarchy.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { normal, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  autoencoder,
  autoencoderEncode,
  autoencoderLoss,
  autoencoderQuantise,
  autoencoderRun,
  initAutoencoder,
  initQuantiserCodebook,
  latentOfCodes,
  type AutoencoderParams,
} from 'aifn-methods/generative/autoencoders'

const x = normal(stream(21), 0, 1, { shape: [120, 2] })
const model = autoencoder({ kind: 'rqvae', inputs: 2, latent: 2, codes: 4, depth: 3, observationSd: 1 })
const params = initQuantiserCodebook(model, initAutoencoder(model, stream(22)), x, stream(23))

describe('the RQ-VAE model', () => {
  it('defaults to 4 stages with a codebook each and β = 0.25; a shared codebook is one block', () => {
    const d = autoencoder({ kind: 'rqvae', inputs: 2 })
    expect([d.spec.depth, d.spec.sharedCodebook, d.spec.beta]).toEqual([4, false, 0.25])
    expect(params.codebook.shape).toEqual([12, 2])
    const shared = autoencoder({ kind: 'rqvae', inputs: 2, codes: 4, depth: 3, sharedCodebook: true })
    expect(initAutoencoder(shared, stream(1)).codebook.shape).toEqual([4, 2])
    expect(autoencoder({ kind: 'vqvae', inputs: 2 }).spec.depth).toBe(1)
  })

  it('quantises stage by stage: each code is the nearest codeword of its stage to the residual so far', () => {
    const q = autoencoderQuantise(model, params, x)
    expect(q.codes.shape).toEqual([120, 3])
    const ze = toRows(unwrap(autoencoderEncode(model, params, x).mean) as Tensor)
    const C = toRows(params.codebook)
    const codes = toRows(q.codes)
    ze.forEach((z, i) => {
      let r = z.slice()
      for (let d = 0; d < 3; d++) {
        const dist = (k: number) => (r[0] - C[d * 4 + k][0]) ** 2 + (r[1] - C[d * 4 + k][1]) ** 2
        const best = [0, 1, 2, 3].reduce((a, k) => (dist(k) < dist(a) ? k : a), 0)
        expect(codes[i][d]).toBe(best)
        r = r.map((v, j) => v - C[d * 4 + best][j])
      }
    })
    // The quantised latent is the sum of the codewords, and the last partial sum.
    expect(Array.from(toFlat(q.quantised))).toEqual(Array.from(toFlat(latentOfCodes(model, params, q.codes))))
    expect(Array.from(toFlat(q.partials[2]))).toEqual(Array.from(toFlat(q.quantised)))
    expect(Array.from(toFlat(latentOfCodes(model, params, q.codes, 0)))).toEqual(new Array(240).fill(0))
  })

  it('has a regulariser of codebook terms plus β times the commitments to every partial sum', () => {
    const parts = autoencoderLoss(model, params, x, stream(1))
    const ze = toRows(unwrap(autoencoderEncode(model, params, x).mean) as Tensor)
    const q = autoencoderQuantise(model, params, x)
    const partials = q.partials.map((p) => toRows(p))
    const meanSq = (f: (i: number) => number[]) =>
      ze.reduce((a, _, i) => a + f(i).reduce((b, v) => b + v * v, 0), 0) / ze.length
    let codebook = 0
    let commitment = 0
    for (let d = 0; d < 3; d++) {
      const before = (i: number) => (d === 0 ? [0, 0] : partials[d - 1][i])
      // codeword_d = partial_d − partial_{d−1}; residual_{d−1} = z_e − partial_{d−1}.
      codebook += meanSq((i) => ze[i].map((v, j) => v - before(i)[j] - (partials[d][i][j] - before(i)[j])))
      commitment += meanSq((i) => ze[i].map((v, j) => v - partials[d][i][j]))
    }
    expect(Number(unwrap(parts.regulariser))).toBeCloseTo(codebook + 0.25 * commitment, 9)
  })

  it('sends gradients to the codebooks and, through the straight-through estimator, to the encoder', () => {
    // The pytree's gradient type is too deep to infer here; the gradient has the parameters' structure.
    const gradOf = grad as unknown as (
      f: (p: AutoencoderParams) => unknown,
    ) => (p: AutoencoderParams) => AutoencoderParams
    const g = gradOf((p) => autoencoderLoss(model, p, x, stream(1)).loss)(params)
    expect(toFlat(g.codebook).some((v) => v !== 0)).toBe(true)
    const encoderGrads = (g.encoder as unknown as Record<string, Tensor>[]).flatMap((layer) =>
      Object.values(layer).flatMap((t) => Array.from(toFlat(t))),
    )
    expect(encoderGrads.some((v) => v !== 0)).toBe(true)
  })
})

describe('an RQ-VAE run', () => {
  let run: ReturnType<typeof runOnce>
  function runOnce() {
    let last
    for (const r of autoencoderRun(
      { x },
      { kind: 'rqvae', latent: 2, codes: 4, depth: 3, observationSd: 1, steps: 120, seed: 3, checkpoints: 4 },
    ))
      last = r
    return last!
  }
  run = runOnce()

  it('records each stage’s error, and each stage leaves less than the one before by the end', () => {
    const last = run.history.stageError.at(-1)!
    expect(last).toHaveLength(3)
    expect(last[1]).toBeLessThan(last[0])
    expect(last[2]).toBeLessThan(last[1])
    expect(run.history.stageError).toHaveLength(run.history.step.length)
    expect([run.depth, run.codewords]).toEqual([3, 4])
  })

  it('checkpoints the codes, the coarse-to-fine reconstructions and the code tree', () => {
    const c = run.checkpoints.at(-1)!
    expect(c.stageCodes).toHaveLength(120 * 3)
    expect(c.partialReconstructions).toHaveLength(3)
    expect(c.usedByStage).toHaveLength(3)
    expect(c.used).toBe(c.usedByStage!.reduce((a, b) => a + b, 0))
    const tree = c.codeTree!
    expect(tree[0].count).toBe(120)
    expect(tree.filter((n) => n.depth === 3).length).toBe(c.uniqueCodes)
    expect(c.uniqueCodes).toBeLessThanOrEqual(64)
    expect(c.codebook).toHaveLength(12 * 2)
  })

  it('is deterministic in its seed', () => {
    expect(Array.from(runOnce().checkpoints.at(-1)!.stageCodes!)).toEqual(
      Array.from(run.checkpoints.at(-1)!.stageCodes!),
    )
  })

  it('leaves the other kinds’ checkpoints without the hierarchy', () => {
    let last
    for (const r of autoencoderRun({ x }, { kind: 'vqvae', latent: 2, codes: 4, steps: 20, checkpoints: 1 })) last = r
    const c = last!.checkpoints.at(-1)!
    expect([c.stageCodes, c.codeTree, c.uniqueCodes]).toEqual([null, null, 0])
    expect(last!.history.stageError).toEqual([])
  })
})
