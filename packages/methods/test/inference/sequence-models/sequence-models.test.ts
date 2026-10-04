import { describe, expect, it } from 'vitest'
import {
  crfGradient,
  crfLogLikelihood,
  crfMarginals,
  crfScore,
  crfViterbi,
  dishonestCasino,
  hmmChain,
  hmmModel,
  linearChainCrf,
  sampleHmm,
} from 'aifn-methods/inference/sequence-models'
import { forwardBackward, viterbi } from 'aifn-compute/inference/exact'
import { infer } from 'aifn-compute/inference/engines'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'

describe('the dishonest casino as an HMM', () => {
  const h = dishonestCasino()
  const obs = [5, 5, 0, 5, 2, 5, 5]
  it('hmmChain folds π into ψ₀; its forward–backward is log p(x), equal to the model description', () => {
    const fb = forwardBackward(hmmChain(h, obs))
    const K = 2
    // p(x) by enumerating every hidden path.
    const A = toFlat(h.transition)
    const B = toFlat(h.emission)
    const pi = toFlat(h.initial)
    let total = 0
    for (let code = 0; code < K ** obs.length; code++) {
      const y = obs.map((_, n) => Math.floor(code / K ** (obs.length - 1 - n)) % K)
      let p = pi[y[0]] * B[y[0] * 6 + obs[0]]
      for (let n = 1; n < obs.length; n++) p *= A[y[n - 1] * K + y[n]] * B[y[n] * 6 + obs[n]]
      total += p
    }
    expect(fb.logLikelihood).toBeCloseTo(Math.log(total), 12)
    const inf = infer(hmmModel(h), { sizes: { T: obs.length }, data: { x: obs } })
    expect(inf.engine).toBe('forward-backward')
    const s = run(inf.algorithm, undefined, 100) as unknown as { marginals: Tensor[]; logZ: number }
    s.marginals.forEach((m, n) => expect(toFlat(m)[1]).toBeCloseTo(toFlat(fb.marginals)[n * K + 1], 12))
    const v = viterbi(hmmChain(h, obs))
    expect(v.path.shape).toEqual([obs.length])
    expect(toFlat(v.path)[3]).toBe(1) // three sixes in four rolls: the loaded die
  })

  it('sampling is reproducible and draws valid symbols', () => {
    const a = sampleHmm(stream(4), h, 50)
    const b = sampleHmm(stream(4), h, 50)
    expect(toFlat(a.observations)).toEqual(toFlat(b.observations))
    expect(toFlat(a.states).every((k) => k === 0 || k === 1)).toBe(true)
    expect(toFlat(a.observations).every((x) => x >= 0 && x < 6)).toBe(true)
  })
})

describe('linear-chain CRF', () => {
  const s = stream(21)
  const draw = (k: string, i: number) => uniform(child(s, k, i)) as number
  const K = 3
  const F = 2
  const N = 5
  const rnd = (k: string, n: number) => Array.from({ length: n }, (_, i) => draw(k, i) * 2 - 1)
  const crf = linearChainCrf(
    [rnd('w0', F), rnd('w1', F), rnd('w2', F)],
    [rnd('t0', K), rnd('t1', K), rnd('t2', K)],
    rnd('s', K),
  )
  const x = Array.from({ length: N }, (_, n) => rnd(`x${n}`, F))
  const y = [0, 2, 1, 1, 0]
  const all = () => {
    const out: number[][] = []
    for (let c = 0; c < K ** N; c++) out.push(Array.from({ length: N }, (_, n) => Math.floor(c / K ** (N - 1 - n)) % K))
    return out
  }
  it('log Z, marginals and Viterbi against enumeration', () => {
    const scores = all().map((lab) => ({ lab, s: crfScore(crf, x, lab) }))
    const logZ = Math.log(scores.reduce((t, { s }) => t + Math.exp(s), 0))
    const m = crfMarginals(crf, x)
    expect(m.logZ).toBeCloseTo(logZ, 12)
    const p0 = scores.filter(({ lab }) => lab[2] === 1).reduce((t, { s }) => t + Math.exp(s - logZ), 0)
    expect(m.marginals.data[2 * K + 1]).toBeCloseTo(p0, 12)
    const best = scores.reduce((a, b) => (b.s > a.s ? b : a))
    expect(Array.from(crfViterbi(crf, x).path.data)).toEqual(best.lab)
  })
  it('the gradient matches finite differences', () => {
    const g = crfGradient(crf, x, y)
    expect(g.logLikelihood).toBeCloseTo(crfLogLikelihood(crf, x, y), 12)
    const h = 1e-6
    const perturb = (block: 'weights' | 'transitions' | 'start', i: number, d: number) => {
      const copy = { weights: crf.weights, transitions: crf.transitions, start: crf.start }
      const data = Float64Array.from(copy[block].data)
      data[i] += d
      return { ...copy, [block]: fromData(data, copy[block].shape) }
    }
    for (const block of ['weights', 'transitions', 'start'] as const)
      for (let i = 0; i < crf[block].data.length; i++) {
        const fd =
          (crfLogLikelihood(perturb(block, i, h), x, y) - crfLogLikelihood(perturb(block, i, -h), x, y)) / (2 * h)
        expect(Math.abs(g[block].data[i] - fd)).toBeLessThan(1e-6)
      }
  })
})
