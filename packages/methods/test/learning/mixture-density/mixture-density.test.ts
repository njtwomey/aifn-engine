/**
 * Mixture density networks: on Bishop's inverse problem a many-component MDN beats the unimodal (squared-error)
 * network's log-likelihood and finds the three branches in the fold; the run streams both networks from step 0; the
 * estimator's predictive is the compute mixture and its decision a mode.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset as asDataset } from 'aifn-compute/learning/estimators'
import type { InverseTruth } from 'aifn-methods/data'
import { bishopInverse } from 'aifn-methods/data/synthetic'
import {
  mdnModel,
  mdnPredict,
  mixtureDensityNetwork,
  mixtureDensityRun,
  type MdnSnapshot,
} from 'aifn-methods/learning/mixture-density'

const dataset = bishopInverse(stream('mdn-test'), { n: 300 })
const data = { x: dataset.x, y: dataset.y! }
const truth = dataset.meta.truth as InverseTruth

describe('mixtureDensityRun', () => {
  let last: MdnSnapshot | undefined
  const snapshots: MdnSnapshot[] = []
  for (const s of mixtureDensityRun({ data, components: 5, hidden: [20], steps: 1200, every: 400, seed: 3 })) {
    snapshots.push(s)
    last = s
  }

  it('streams from step 0, with checkpoints of both networks', () => {
    expect(snapshots[0].step).toBe(0)
    expect(last!.done).toBe(true)
    expect(last!.checkpoints.map((c) => c.step)).toEqual([0, 400, 800, 1200])
    expect(last!.spec.objective).toBe('mixture')
    expect(last!.meanSpec.objective).toBe('squared-error')
  })

  it('at large K the MDN log-likelihood beats the unimodal network', () => {
    const h = last!.history
    const end = h.step.length - 1
    expect(h.nll[end]).toBeLessThan(h.nll[0])
    expect(h.nll[end]).toBeLessThan(h.meanNll[end] - 0.5)
  })

  it('finds the three branches in the fold, and the squared-error network predicts the mean between them', () => {
    const final = last!.checkpoints.at(-1)!
    const x = tensor([[0.5], [0.42]])
    const p = mdnPredict(mdnModel(last!.spec), final.mixture, x)
    // The three most probable modes (a narrow spurious component may add a fourth, lower one).
    const modes = p.mixture!.modes(0).slice(0, 3)
    const branches = truth.solutions([0.5]).map((s) => s.value[0])
    for (const b of branches) expect(Math.min(...modes.map((m) => Math.abs(m.value[0] - b)))).toBeLessThan(0.05)
    const mean = mdnPredict(mdnModel(last!.meanSpec), final.mean, x).mean[1]
    const sol = truth.solutions([0.42]).map((s) => s.value[0])
    // It fits the true conditional mean, which lies on no branch.
    expect(mean).toBeCloseTo(toFlat(truth.mean(tensor([[0.42]])))[0], 1)
    expect(Math.min(...sol.map((s) => Math.abs(s - mean)))).toBeGreaterThan(0.03)
  })
})

describe('mixtureDensityNetwork', () => {
  it('fits; its predictive is the mixture and its decision the most probable mode', () => {
    const model = mixtureDensityNetwork({ components: 3, steps: 1000 }).fit(asDataset(data.x, data.y), {
      stream: stream('fit'),
    })
    const x = tensor([[0.1], [0.9]])
    const law = model.predictive(x)
    expect(law.batchShape).toEqual([2])
    const decided = toFlat(model.decide(x))
    const m = model.mixture(x)
    expect(decided[0]).toBeCloseTo(m.modes(0)[0].value[0], 12)
    const lp = toFlat(unwrap(law.logProb(tensor(Array.from(decided)))) as Tensor)
    expect(lp[0]).toBeCloseTo(m.logDensity(0, [decided[0]]), 10)
    // At the ends of x there is one branch, and the decision is near it.
    expect(decided[0]).toBeCloseTo(truth.solutions([0.1])[0].value[0], 1)
  })
})

describe('mixtureDensityRun by full-batch L-BFGS', () => {
  it('lowers both networks’ objectives in a few dozen iterations', () => {
    const snaps = [
      ...mixtureDensityRun({
        data,
        components: 3,
        hidden: [10],
        steps: 40,
        every: 10,
        seed: 4,
        method: { method: 'lbfgs' },
      }),
    ]
    const h = snaps.at(-1)!.history
    expect(snaps.at(-1)!.done).toBe(true)
    expect(h.nll.at(-1)!).toBeLessThan(h.nll[0] - 0.5)
    expect(h.meanMse.at(-1)!).toBeLessThan(h.meanMse[0])
  })
})
