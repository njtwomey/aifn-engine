/**
 * aifn-compute/probability/information. References: scipy.stats.entropy, scipy.spatial.distance.jensenshannon and
 * sklearn.metrics.mutual_info_score (values hard-coded), and identities between the measures. Channels, codes and
 * KL projections are applications (applications/test/information).
 */

import { describe, expect, it } from 'vitest'
import * as I from 'aifn-compute/probability/information'
import { Normal } from 'aifn-compute/probability/distributions'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { fromRows, tensor, toFlat, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import { checkGradient } from '../../foundation/tensor/check-gradient'

const n = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}
const close = (a: number, b: number, tol = 1e-12) =>
  expect(Math.abs(a - b) / Math.max(1, Math.abs(b))).toBeLessThan(tol)
const P = [0.1, 0.2, 0.3, 0.4]
const Q = [0.25, 0.25, 0.4, 0.1]

describe('info: measures', () => {
  it('against scipy and sklearn', () => {
    close(n(I.entropy(P, { base: 2 })), 1.8464393446710157)
    close(n(I.entropy([1, 2, 3, 4])), n(I.entropy(P)))
    close(n(I.klDivergence(P, Q)), 0.3319553392621645)
    close(n(I.jensenShannonDistance(P, Q, { base: 2 })), 0.317253708730743)
    close(
      n(
        I.mutualInformation([
          [10, 2, 3],
          [1, 8, 4],
        ]),
      ),
      0.22147334222306372,
    )
    expect(n(I.klDivergence([0.5, 0.5], [1, 0]))).toBe(Infinity)
  })

  it('identities between the measures', () => {
    const joint = [
      [10, 2, 3],
      [1, 8, 4],
    ]
    close(n(I.conditionalEntropy(joint)), n(I.jointEntropy(joint)) - n(I.entropy([15, 13])))
    close(n(I.crossEntropy(P, Q)), n(I.entropy(P)) + n(I.klDivergence(P, Q)))
    close(I.fDivergence(P, Q, I.fGenerators.kl), n(I.klDivergence(P, Q)))
    close(I.fDivergence(P, Q, I.fGenerators.reverseKl), n(I.klDivergence(Q, P)))
    close(I.fDivergence(P, Q, I.fGenerators.jensenShannon), n(I.jensenShannonDivergence(P, Q)))
    close(I.fDivergence(P, Q, I.fGenerators.totalVariation), n(I.totalVariation(P, Q)))
    close(Math.sqrt(I.fDivergence(P, Q, I.fGenerators.squaredHellinger)), n(I.hellingerDistance(P, Q)))
    const pmi = toFlat(I.pointwiseMutualInformation(joint) as never)
    close(pmi[0], Math.log(10 / 28 / ((15 / 28) * (11 / 28))))
    close(n(I.differentialEntropy(Normal(0, 2), { base: 2 })), n(Normal(0, 2).entropy()) / Math.LN2)
    close(
      n(
        I.gaussianMutualInformation(
          fromRows([
            [1, 0.6],
            [0.6, 1],
          ]),
          [0],
          [1],
        ),
      ),
      -0.5 * Math.log(1 - 0.36),
    )
  })

  it('are differentiable', () => {
    checkGradient((p, q) => I.klDivergence(p, q), [tensor(P), tensor(Q)], { tol: 1e-6 })
    checkGradient((p) => I.entropy(p), [tensor(P)])
    checkGradient((p, q) => I.jensenShannonDivergence(p, q), [tensor(P), tensor(Q)], { tol: 1e-6 })
  })

  it('KSG estimate is close to the Gaussian closed form', () => {
    const s = stream('ksg')
    const a = toFlat(normals(child(s, 'a'), 1000))
    const e = toFlat(normals(child(s, 'e'), 1000))
    const b = a.map((v, i) => 0.6 * v + 0.8 * e[i])
    expect(Math.abs(I.ksgMutualInformation(a, b) - -0.5 * Math.log(1 - 0.36))).toBeLessThan(0.03)
  })
})
