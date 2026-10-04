import { describe, expect, it } from 'vitest'
import {
  blobsDog,
  blobsLog,
  canny,
  closing,
  detectCorners,
  dilate,
  discElement,
  erode,
  gaussianBlur,
  gaussianLaplace,
  gaussianPyramid,
  gradients,
  harrisResponse,
  houghCirclePeaks,
  houghCircles,
  houghLinePeaks,
  houghLines,
  laplacianPyramid,
  opening,
  pyramidReduce,
  reconstructLaplacian,
  shiTomasiResponse,
  sobel,
  squareElement,
  structureTensor,
} from 'aifn-compute/signal/image'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Img = number[][]
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const S = fixture<any>('signal').image
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('signal/image')

const tensor = (rows: Img) => fromData(Float64Array.from(rows.flat()), [rows.length, rows[0].length])
const close = (a: Tensor, e: Img, tol: number) => {
  const x = toFlat(a)
  const y = e.flat()
  expect(x.length).toBe(y.length)
  const scale = Math.max(1, ...y.map(Math.abs))
  x.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(tol * scale))
}
const img = tensor(F.image)

describe('linear filters against scipy.ndimage and scikit-image', () => {
  it('Gaussian blur, Sobel, Prewitt and Scharr', () => {
    const small = tensor(S.image)
    close(gaussianBlur(small, 1.5), S.gaussian, 1e-12)
    const s = sobel(small)
    close(s.gx, S.sobelX, 1e-12)
    close(s.gy, S.sobelY, 1e-12)
    const p = gradients(img, { operator: 'prewitt' })
    close(p.gx, F.prewitt.x, 1e-12)
    close(p.gy, F.prewitt.y, 1e-12)
    const c = gradients(img, { operator: 'scharr' })
    close(c.gx, F.scharr.x, 1e-12)
    close(c.gy, F.scharr.y, 1e-12)
  })

  it('structure tensor, Harris and Shi–Tomasi responses', () => {
    const M = structureTensor(img, { sigma: 1 })
    close(M.rr, F.structureTensor.rr, 1e-10)
    close(M.rc, F.structureTensor.rc, 1e-10)
    close(M.cc, F.structureTensor.cc, 1e-10)
    close(harrisResponse(img, { k: 0.05, sigma: 1 }), F.harrisResponse, 1e-10)
    close(shiTomasiResponse(img, { sigma: 1 }), F.shiTomasiResponse, 1e-10)
  })

  it('the Laplacian of Gaussian', () => {
    close(gaussianLaplace(img, 2), F.gaussianLaplace, 1e-12)
  })
})

describe('features', () => {
  it('finds the four corners of the square among the Harris peaks', () => {
    const found = detectCorners(img, { minDistance: 2, relative: 0.2 })
    for (const [r, c] of [
      [8, 6],
      [8, 17],
      [19, 6],
      [19, 17],
    ])
      expect(found.some((p) => Math.abs(p.row - r) <= 1.5 && Math.abs(p.col - c) <= 1.5)).toBe(true)
  })

  it('LoG blobs match scikit-image', () => {
    const b = blobsLog(tensor(F.blobsLog.image), { minSigma: 1, maxSigma: 8, levels: 10, threshold: 0.1 })
    const want = (F.blobsLog.blobs as number[][]).map(([r, c, s]) => ({ r, c, s }))
    expect(b.length).toBe(want.length)
    for (const w of want)
      expect(b.some((x) => x.row === w.r && x.col === w.c && Math.abs(x.sigma - w.s) < 1e-9)).toBe(true)
    // The difference of Gaussians finds the same centres.
    const d = blobsDog(tensor(F.blobsLog.image), { minSigma: 1, maxSigma: 8, threshold: 0.1 })
    for (const w of want) expect(d.some((x) => Math.hypot(x.row - w.r, x.col - w.c) <= 1)).toBe(true)
  })

  it('Canny edges agree with scikit-image, and are thin', () => {
    const c = canny(img, { sigma: 1.5, quantile: false, low: 0.1 * 4, high: 0.2 * 4 })
    const ours = toFlat(c.edges)
    const theirs = (F.canny.edges as Img).flat()
    // Agreement within one pixel: each of their edges has one of ours in its 3 × 3 neighbourhood, and vice versa.
    const near = (a: number[], b: number[]) => {
      let hit = 0
      let total = 0
      for (let r = 0; r < 40; r++)
        for (let k = 0; k < 48; k++) {
          if (!a[r * 48 + k]) continue
          total++
          let ok = false
          for (let dr = -1; dr <= 1 && !ok; dr++)
            for (let dk = -1; dk <= 1; dk++) if (b[(r + dr) * 48 + k + dk]) ok = true
          if (ok) hit++
        }
      return hit / total
    }
    expect(near(theirs, ours)).toBeGreaterThan(0.9)
    expect(near(ours, theirs)).toBeGreaterThan(0.9)
    // Stages are consistent: every edge is strong or weak, and strong pixels are edges.
    const strong = toFlat(c.strong)
    const weak = toFlat(c.weak)
    ours.forEach((e, i) => {
      if (e) expect(strong[i] + weak[i]).toBe(1)
      if (strong[i]) expect(e).toBe(1)
    })
  })

  it('the Hough line accumulator equals scikit-image and its peaks find both lines', () => {
    const acc = houghLines(tensor(F.houghLines.edges))
    close(acc.votes, F.houghLines.votes, 1e-12)
    const lines = houghLinePeaks(acc, { count: 2 })
    const got = lines.map((l) => [l.angle, l.distance]).sort((a, b) => a[0] - b[0])
    const want = (F.houghLines.angles as number[])
      .map((a, i) => [a, F.houghLines.distances[i]])
      .sort((a, b) => a[0] - b[0])
    got.forEach((g, i) => {
      expect(g[0]).toBeCloseTo(want[i][0], 9)
      expect(g[1]).toBeCloseTo(want[i][1], 9)
    })
  })

  it('the Hough circle transform finds a drawn circle', () => {
    const h = 50
    const w = 60
    const e = new Float64Array(h * w)
    for (let k = 0; k < 400; k++) {
      const a = (2 * Math.PI * k) / 400
      e[Math.round(22 + 11 * Math.sin(a)) * w + Math.round(31 + 11 * Math.cos(a))] = 1
    }
    const radii = [8, 9, 10, 11, 12, 13]
    const found = houghCirclePeaks(houghCircles(fromData(e, [h, w]), radii), radii, { count: 1 })
    expect(found[0].row).toBe(22)
    expect(found[0].col).toBe(31)
    expect(found[0].radius).toBe(11)
    expect(found[0].score).toBeGreaterThan(0.9)
  })
})

describe('morphology against scipy.ndimage', () => {
  it('erosion, dilation, opening and closing', () => {
    const cross = discElement(1)
    close(erode(img, cross), F.morphology.erode, 1e-15)
    close(dilate(img, cross), F.morphology.dilate, 1e-15)
    close(erode(img, squareElement(5)), F.morphology.erode5, 1e-15)
    close(opening(img, squareElement(5)), F.morphology.opening, 1e-15)
    close(closing(img, squareElement(5)), F.morphology.closing, 1e-15)
  })

  it('laws: opening is anti-extensive and idempotent, closing extensive', () => {
    const e = squareElement(3)
    const o = toFlat(opening(img, e))
    const c = toFlat(closing(img, e))
    const v = toFlat(img)
    v.forEach((x, i) => {
      expect(o[i]).toBeLessThanOrEqual(x + 1e-15)
      expect(c[i]).toBeGreaterThanOrEqual(x - 1e-15)
    })
    close(opening(opening(img, e), e), [Array.from(o)], 1e-15)
  })
})

describe('pyramids', () => {
  it('reduce matches the five-tap kernel with mirrored borders', () => {
    close(pyramidReduce(img), F.pyramidReduce, 1e-12)
  })

  it('the Laplacian pyramid reconstructs the image exactly', () => {
    const L = laplacianPyramid(img)
    expect(L.length).toBe(gaussianPyramid(img).length)
    close(reconstructLaplacian(L), F.image, 1e-12)
  })
})
