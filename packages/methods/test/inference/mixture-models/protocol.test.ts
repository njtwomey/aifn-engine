import { describe, it } from 'vitest'
import { caviGaussianMixture } from 'aifn-methods/inference/mixture-models'
import { expectProtocol } from '../../protocol'

describe('caviGaussianMixture', () => {
  it('follows the trace protocol (the means start at data points drawn at init)', () => {
    const x = [-3.1, -2.8, -3.3, -2.9, 0.1, 0.3, -0.2, 3.2, 2.9, 3.1, 2.7]
    expectProtocol(caviGaussianMixture(x, 3), undefined, { n: 8, record: { elbo: (s) => s.elbo } })
  })
})
