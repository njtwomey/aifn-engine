import { describe, expect, it } from 'vitest'
import { getWindow, windowValues, type WindowSpec } from 'aifn-compute/signal/windows'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { close, F } from '../helpers'

describe('windows against scipy', () => {
  const specs: Record<string, WindowSpec> = {
    hann: 'hann',
    hamming: 'hamming',
    blackman: 'blackman',
    blackmanharris: 'blackmanharris',
    nuttall: 'nuttall',
    flattop: 'flattop',
    bartlett: 'bartlett',
    triangular: 'triangular',
    boxcar: 'boxcar',
    cosine: { name: 'cosine' },
    kaiser: { name: 'kaiser', beta: 8 },
    gaussian: { name: 'gaussian', std: 2.5 },
    tukey: { name: 'tukey', alpha: 0.5 },
  }
  it.each(Object.keys(specs))('%s, symmetric and periodic', (name) => {
    for (const n of [10, 11])
      for (const kind of ['sym', 'periodic']) {
        const w = getWindow(specs[name], n, { periodic: kind === 'periodic' })
        close(w, F.windows[`${name}-${n}-${kind}`], 1e-12)
      }
  })

  it('windowValues is the raw form of getWindow', () => {
    expect(Array.from(windowValues('hann', 8, true))).toEqual(toFlat(getWindow('hann', 8, { periodic: true })))
  })
})
