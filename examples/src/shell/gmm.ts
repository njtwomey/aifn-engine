/**
 * Expectation–maximisation for a mixture of Gaussians in the plane, for the landing page's figure: the data and
 * every iteration of the fit, computed once. Dempster, Laird and Rubin (1977); Bishop (2006), §9.2.
 */
import { normal, stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import { grid } from '@examples/data'

export type P = number[]
/** A 2×2 covariance as [a, b, d]: [[a, b], [b, d]]. */
export type Cov = [number, number, number]
export type Mixture = { weights: number[]; means: P[]; covs: Cov[] }

export const K = 5
/** Five clusters of different sizes, shapes and orientations: each a centre and a factor A, the cluster A z + c. */
const CLUSTERS: { centre: P; A: [number, number, number, number]; n: number }[] = [
  { centre: [-2.6, 2.3], A: [0.9, 0.5, 0, 0.35], n: 60 },
  { centre: [2.5, 2.6], A: [0.35, 0, 0, 0.8], n: 50 },
  { centre: [0, -0.1], A: [0.45, 0, 0, 0.45], n: 50 },
  { centre: [-2.4, -2.6], A: [0.7, -0.55, 0, 0.3], n: 50 },
  { centre: [2.8, -2.3], A: [0.9, 0, 0, 0.3], n: 40 },
]
export const DATA: P[] = CLUSTERS.flatMap(({ centre, A, n }, c) =>
  toRows(normal(stream(`home/gmm/${c}`), 0, 1, { shape: [n, 2] })).map(([u, v]) => [
    centre[0] + A[0] * u + A[1] * v,
    centre[1] + A[2] * u + A[3] * v,
  ]),
)

/** The first means, all five bunched in the bottom-left corner, with unit covariances and equal weights. */
const START: Mixture = {
  weights: Array.from({ length: K }, () => 1 / K),
  means: grid(0, 1, K).map((t) => [-3.9 + 0.8 * t, -3.9 + 0.8 * (1 - t) ** 2]),
  covs: Array.from({ length: K }, () => [1, 0, 1] as Cov),
}

const LOG_2PI = Math.log(2 * Math.PI)
/** log N(x | m, S) for a 2×2 covariance. */
function logDensity(x: P, m: P, [a, b, d]: Cov) {
  const det = a * d - b * b
  const u = x[0] - m[0]
  const v = x[1] - m[1]
  return -LOG_2PI - 0.5 * Math.log(det) - (0.5 * (d * u * u - 2 * b * u * v + a * v * v)) / det
}

/** The E-step: each point's responsibilities [n, K] and the log-likelihood of the data. */
export function responsibilities(m: Mixture) {
  let logLik = 0
  const R = DATA.map((x) => {
    const l = m.means.map((mu, k) => Math.log(m.weights[k]) + logDensity(x, mu, m.covs[k]))
    const top = Math.max(...l)
    const s = l.reduce((acc, v) => acc + Math.exp(v - top), 0)
    logLik += top + Math.log(s)
    return l.map((v) => Math.exp(v - top) / s)
  })
  return { R, logLik }
}

/** The M-step: weights, means and covariances from the responsibilities (a small ridge keeps each one invertible). */
function maximise(R: number[][]): Mixture {
  const n = DATA.length
  const weights: number[] = []
  const means: P[] = []
  const covs: Cov[] = []
  for (let k = 0; k < K; k++) {
    const Nk = R.reduce((acc, r) => acc + r[k], 0) + 1e-9
    const mx = DATA.reduce((acc, x, i) => acc + R[i][k] * x[0], 0) / Nk
    const my = DATA.reduce((acc, x, i) => acc + R[i][k] * x[1], 0) / Nk
    let [a, b, d] = [0, 0, 0]
    DATA.forEach((x, i) => {
      const u = x[0] - mx
      const v = x[1] - my
      a += R[i][k] * u * u
      b += R[i][k] * u * v
      d += R[i][k] * v * v
    })
    weights.push(Nk / n)
    means.push([mx, my])
    covs.push([a / Nk + 1e-3, b / Nk, d / Nk + 1e-3])
  }
  return { weights, means, covs }
}

export type Phase = 'start' | 'iterate' | 'done'
export type Frame = { iter: number; phase: Phase; mixture: Mixture; labels: number[]; logLik: number }

/** EM from the corner start to convergence (the log-likelihood rising by less than 10⁻⁶ per point), a frame an iteration. */
export function em(): Frame[] {
  const labels = (R: number[][]) => R.map((r) => r.indexOf(Math.max(...r)))
  let m = START
  let e = responsibilities(m)
  const frames: Frame[] = [{ iter: 0, phase: 'start', mixture: m, labels: labels(e.R), logLik: e.logLik }]
  for (let iter = 1; iter <= 500; iter++) {
    m = maximise(e.R)
    const next = responsibilities(m)
    const gain = next.logLik - e.logLik
    e = next
    frames.push({ iter, phase: 'iterate', mixture: m, labels: labels(e.R), logLik: e.logLik })
    if (gain < 1e-6 * DATA.length) break
  }
  frames.push({ ...frames[frames.length - 1], phase: 'done' })
  return frames
}

/** The ellipse of a 2×2 covariance scaled by `s` standard deviations around `m`, as a closed polyline. */
export function ellipse(m: P, [a, b, d]: Cov, s: number, points = 72): P[] {
  const half = (a + d) / 2
  const r = Math.sqrt(((a - d) / 2) ** 2 + b * b)
  const [l1, l2] = [half + r, Math.max(half - r, 0)]
  const theta = 0.5 * Math.atan2(2 * b, a - d)
  const [c, sn] = [Math.cos(theta), Math.sin(theta)]
  return grid(0, 2 * Math.PI, points).map((t) => {
    const u = s * Math.sqrt(l1) * Math.cos(t)
    const v = s * Math.sqrt(l2) * Math.sin(t)
    return [m[0] + c * u - sn * v, m[1] + sn * u + c * v]
  })
}
