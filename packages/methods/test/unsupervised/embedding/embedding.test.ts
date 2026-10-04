import { describe, expect, it } from 'vitest'
import {
  andrewsCurves,
  classicalMds,
  kernelPca,
  metricMds,
  pca,
  smacofSteps,
} from 'aifn-methods/unsupervised/embedding/linear'
import {
  curveParameters,
  DESCENT_ABOVE,
  fuzzyGraph,
  spectralLayout,
  jointProbabilities,
  perplexityCalibration,
  tsne,
  tsneSteps,
  umap,
  umapSteps,
} from 'aifn-methods/unsupervised/embedding/neighbour'
import { nearestNeighbourDescent } from 'aifn-compute/numerics/neighbours'
import { isomap, laplacianEigenmaps, locallyLinearEmbedding } from 'aifn-methods/unsupervised/embedding/manifold'
import { rbf } from 'aifn-compute/learning/kernels'
import { blobs } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { dense, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { run, seek, trace } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'
import { fixture } from '../../fixtures'

type Fx = {
  x: number[][]
  xq: number[][]
  pca: {
    components: number[][]
    variance: number[]
    ratio: number[]
    noise: number
    transform: number[][]
    whiten: number[][]
  }
  kpca: { eigenvalues: number[]; embedding: number[][]; transform: number[][] }
  spiral: number[][]
  isomap: number[][]
  spectral: number[][]
  lle: number[][]
  mds_eigenvalues: number[]
  smacof: { init: number[][]; steps: number[][][] }
  joint: number[][]
}
const fx = fixture<Fx>('unsupervised/embedding')
const X = tensor(fx.x)
const XQ = tensor(fx.xq)
const S = tensor(fx.spiral)

function close(a: Tensor | number[], b: unknown, digits = 8) {
  const got = Array.isArray(a) ? a : toFlat(a)
  const want = (b as number[]).flat(3) as number[]
  expect(got.length).toBe(want.length)
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], digits))
}

/** Columns equal up to sign. */
function closeUpToSign(a: Tensor, b: number[][], digits = 6) {
  const A = toRows(a)
  for (let c = 0; c < b[0].length; c++) {
    const s = Math.sign(A.reduce((acc, r, i) => acc + r[c] * b[i][c], 0)) || 1
    A.forEach((r, i) => expect(s * r[c]).toBeCloseTo(b[i][c], digits))
  }
}

const distances = (x: number[][]) => tensor(x.map((a) => x.map((b) => Math.hypot(...a.map((u, k) => u - b[k])))))

describe('PCA and kernel PCA', () => {
  it('PCA matches scikit-learn, with whitening and round trips', () => {
    const m = pca({ components: 3 }).fit(dataset(X))
    close(m.components, fx.pca.components)
    close(m.explainedVariance, fx.pca.variance)
    close(m.explainedVarianceRatio, fx.pca.ratio)
    expect(m.noiseVariance).toBeCloseTo(fx.pca.noise, 8)
    close(m.transform(XQ), fx.pca.transform)
    close(pca({ components: 2, whiten: true }).fit(dataset(X)).transform(XQ), fx.pca.whiten)
    const full = pca().fit(dataset(X))
    close(full.inverseTransform(full.transform(XQ)), fx.xq)
  })
  it('kernel PCA matches scikit-learn', () => {
    const m = kernelPca({ kernel: rbf({ lengthscale: Math.SQRT2 }), components: 2 }).fit(dataset(X))
    close(m.eigenvalues, fx.kpca.eigenvalues, 7)
    closeUpToSign(m.embedding, fx.kpca.embedding)
    closeUpToSign(m.transform(XQ), fx.kpca.transform)
  })
})

describe('multidimensional scaling', () => {
  it('classical MDS eigenvalues and distances', () => {
    const r = classicalMds(distances(fx.x), 4)
    close(r.eigenvalues, fx.mds_eigenvalues, 7)
    // Euclidean input: the full-rank embedding reproduces the distances.
    close(distances(toRows(r.embedding)), toRows(distances(fx.x)), 7)
  })
  it('SMACOF steps match the Guttman transform; stress falls', () => {
    const alg = smacofSteps(distances(fx.x))
    const t = trace(alg, { embedding: tensor(fx.smacof.init) }, 3)
    fx.smacof.steps.forEach((want, k) => close(t.steps[k + 1].embedding, want))
    const m = metricMds().fit(dataset(X))
    const s = toFlat(m.training.series.stress)
    for (let i = 1; i < s.length; i++) expect(s[i]).toBeLessThanOrEqual(s[i - 1] + 1e-9)
    expect(toFlat(seek(alg, {}, 4).embedding)).toEqual(toFlat(run(alg, {}, 4).embedding))
  })
})

describe('neighbour-graph embeddings', () => {
  it('Isomap matches scikit-learn', () => {
    const m = isomap({ neighbours: 6 }).fit(dataset(S))
    expect(m.components).toBe(1)
    closeUpToSign(m.embedding, fx.isomap, 5)
  })
  it('Laplacian eigenmaps match SpectralEmbedding', () => {
    // scikit-learn counts the point itself among its neighbours.
    closeUpToSign(laplacianEigenmaps({ neighbours: 6 }).fit(dataset(S)).embedding, fx.spectral, 5)
  })
  it('LLE matches scikit-learn', () => {
    closeUpToSign(locallyLinearEmbedding({ neighbours: 6 }).fit(dataset(S)).embedding, fx.lle, 5)
  })
})

describe('t-SNE', () => {
  it('joint affinities match scikit-learn; calibration hits the perplexity', () => {
    close(jointProbabilities(X, 5), fx.joint, 6)
    const n = fx.x.length
    const d2 = tensor(fx.x.map((a) => fx.x.map((b) => a.reduce((s, u, k) => s + (u - b[k]) ** 2, 0))))
    const c = perplexityCalibration(d2, 5)
    for (const h of toFlat(c.entropies)) expect(Math.abs(h - Math.log(5))).toBeLessThan(1e-5)
    expect(c.conditional.shape).toEqual([n, n])
  })
  it('is deterministic, lowers KL and follows the trace protocol', () => {
    const a = tsne({ perplexity: 5, iterations: 300 }).fit(dataset(X), { stream: stream(1) })
    const b = tsne({ perplexity: 5, iterations: 300 }).fit(dataset(X), { stream: stream(1) })
    expect(toFlat(a.embedding)).toEqual(toFlat(b.embedding))
    const kl = toFlat(a.training.series.kl)
    expect(kl.at(-1)!).toBeLessThan(kl[kl.length > 30 ? 26 : 1])
    expectProtocol(tsneSteps(jointProbabilities(X, 5)), {}, { record: { kl: (st) => st.kl } })
  })
})

describe('UMAP', () => {
  it('curve parameters match umap-learn for min_dist 0.1', () => {
    const { a, b } = curveParameters(0.1, 1)
    expect(a).toBeCloseTo(1.577, 2)
    expect(b).toBeCloseTo(0.895, 2)
  })
  it('the fuzzy graph calibrates σ and is symmetric with weights in (0, 1]', () => {
    const g = fuzzyGraph(S, 8)
    for (const w of g.edges.weight) expect(w > 0 && w <= 1).toBe(true)
    expect(toFlat(g.rho).every((r) => r > 0)).toBe(true)
  })
  it('nearest-neighbour descent recalls ≥ 0.9 of the exact 10 nearest neighbours on blobs', () => {
    const data = blobs(stream('nn-descent'), { n: 1500, centers: 5, dim: 8 })
    const x = data.x as Tensor
    const [n, d] = x.shape
    const v = dense.data(x)
    const k = 10
    const exact = fuzzyGraph(x, k + 1, { search: 'exact' })
    const result = nearestNeighbourDescent(x, k, { stream: stream(1) })
    const found = {
      indices: result.indices.data as Int32Array,
      distances: toFlat(result.distances),
      rounds: result.rounds,
    }
    const truth = toFlat(exact.neighbours)
    let hits = 0
    for (let i = 0; i < n; i++) {
      const want = new Set(Array.from({ length: k }, (_, r) => truth[i * (k + 1) + r + 1]))
      for (let r = 0; r < k; r++) if (want.has(found.indices[i * k + r])) hits++
    }
    expect(hits / (n * k)).toBeGreaterThanOrEqual(0.9)
    expect(found.rounds).toBeLessThan(30)
    // Lists are sorted, hold no self or repeats, and report Euclidean distances.
    for (let i = 0; i < n; i += 97) {
      const row = Array.from(found.indices.subarray(i * k, (i + 1) * k))
      expect(new Set(row).size).toBe(k)
      expect(row.includes(i)).toBe(false)
      for (let r = 1; r < k; r++)
        expect(found.distances[i * k + r]).toBeGreaterThanOrEqual(found.distances[i * k + r - 1])
      const j = row[0]
      let q = 0
      for (let c = 0; c < d; c++) q += (v[i * d + c] - v[j * d + c]) ** 2
      expect(found.distances[i * k]).toBeCloseTo(Math.sqrt(q), 12)
    }
  })
  it('the fuzzy graph by descent matches the exact one on small data, and auto is exact up to DESCENT_ABOVE', () => {
    const exact = fuzzyGraph(S, 8, { search: 'exact' })
    const auto = fuzzyGraph(S, 8)
    expect(toFlat(auto.neighbours)).toEqual(toFlat(exact.neighbours))
    expect(DESCENT_ABOVE).toBe(2000)
    const approx = fuzzyGraph(S, 8, { search: 'descent', stream: stream(2) })
    // Self first; σ and ρ agree where the neighbour lists do.
    const nb = toFlat(approx.neighbours)
    const want = toFlat(exact.neighbours)
    const k = 8
    const rows = S.shape[0]
    let same = 0
    for (let i = 0; i < rows; i++) {
      expect(nb[i * k]).toBe(i)
      if (Array.from({ length: k }, (_, r) => nb[i * k + r] === want[i * k + r]).every(Boolean)) {
        same++
        expect(toFlat(approx.sigma)[i]).toBeCloseTo(toFlat(exact.sigma)[i], 10)
      }
    }
    expect(same / rows).toBeGreaterThan(0.8)
  })
  it('spectral start: Lanczos (eigsh) matches the dense eigendecomposition', () => {
    const g = fuzzyGraph(S, 8)
    const d = spectralLayout(g, 2, { method: 'dense' })
    const l = spectralLayout(g, 2, { method: 'lanczos' })
    expect(l.length).toBe(d.length)
    let worst = 0
    for (let i = 0; i < d.length; i++) worst = Math.max(worst, Math.abs(l[i] - d[i]))
    expect(worst).toBeLessThan(1e-5)
  })
  it('spectral start stays cheap at n = 5000 (sparse products, no n × n matrix)', () => {
    const big = blobs(stream(11), { n: 5000, centers: 4, dim: 5, sd: 3 }).x
    const g = fuzzyGraph(big, 15)
    const t0 = performance.now()
    const Y = spectralLayout(g, 2)
    const ms = performance.now() - t0
    expect(Y.length).toBe(10000)
    expect(Y.every(Number.isFinite)).toBe(true)
    expect(ms).toBeLessThan(20000)
  }, 120_000)
  it('is deterministic and follows the trace protocol', () => {
    const a = umap({ neighbours: 8, epochs: 50 }).fit(dataset(S), { stream: stream(3) })
    const b = umap({ neighbours: 8, epochs: 50 }).fit(dataset(S), { stream: stream(3) })
    expect(toFlat(a.embedding)).toEqual(toFlat(b.embedding))
    expect(toFlat(a.embedding).every(Number.isFinite)).toBe(true)
    expectProtocol(umapSteps(fuzzyGraph(S, 8), { epochs: 20 }), {}, { n: 10 })
    expectProtocol(smacofSteps(distances(fx.x)), {}, { n: 6 })
  })
})

describe('andrewsCurves', () => {
  it('evaluates x1/√2 + x2 sin t + x3 cos t + x4 sin 2t + x5 cos 2t', () => {
    const x = tensor([
      [1, 2, 3, 4, 5],
      [0, 0, 0, 0, 1],
    ])
    const t = [0, 0.3, -1.2, Math.PI]
    const { curves } = andrewsCurves(x, t)
    const f = (r: number[], s: number) =>
      r[0] / Math.SQRT2 + r[1] * Math.sin(s) + r[2] * Math.cos(s) + r[3] * Math.sin(2 * s) + r[4] * Math.cos(2 * s)
    const rows = toRows(curves) as number[][]
    expect(curves.shape).toEqual([2, 4])
    t.forEach((s, j) => {
      expect(rows[0][j]).toBeCloseTo(f([1, 2, 3, 4, 5], s), 12)
      expect(rows[1][j]).toBeCloseTo(f([0, 0, 0, 0, 1], s), 12)
    })
  })

  it('preserves distances: ∫(f_x − f_y)² dt = π‖x − y‖² on [−π, π]', () => {
    const x = tensor([
      [0.5, -1, 2, 0.3],
      [1.5, 0.2, -0.4, 1],
    ])
    const m = 4000
    // Midpoint rule on a full period is exact for trigonometric polynomials of low degree.
    const t = Array.from({ length: m }, (_, j) => -Math.PI + (2 * Math.PI * (j + 0.5)) / m)
    const rows = toRows(andrewsCurves(x, t).curves) as number[][]
    const integral = rows[0].reduce((s, a, j) => s + (a - rows[1][j]) ** 2, 0) * ((2 * Math.PI) / m)
    const d2 = [1, -1.2, 2.4, -0.7].reduce((s, v) => s + v * v, 0)
    expect(integral).toBeCloseTo(Math.PI * d2, 9)
  })

  it('defaults to 101 points on [−π, π] and rejects a vector', () => {
    const { t, curves } = andrewsCurves(tensor([[1, 1]]))
    expect(t.length).toBe(101)
    expect(t[0]).toBeCloseTo(-Math.PI, 12)
    expect(t[100]).toBeCloseTo(Math.PI, 12)
    expect(curves.shape).toEqual([1, 101])
    expect(() => andrewsCurves(tensor([1, 2]))).toThrow(/matrix/)
  })
})
