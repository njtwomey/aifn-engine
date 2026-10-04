import { describe, expect, it } from 'vitest'
import {
  agglomerative,
  agglomerativeSteps,
  canonicalLabels,
  cutTree,
  dbscan,
  dendrogram,
  gaussianMixture,
  gaussianMixtureSteps,
  kMedoids,
  kmeans,
  kmeansSteps,
  kMedoidsSteps,
  meanShiftSteps,
  miniBatchKMeansSteps,
  linkage,
  meanShift,
  mergeTree,
  miniBatchKMeans,
  optics,
  spectralClustering,
  CORE,
} from 'aifn-methods/unsupervised/clustering'
import { leaves } from 'aifn-compute/graph'
import { pairwiseDistances } from 'aifn-compute/numerics/linalg'
import { kmeansPlusPlus } from 'aifn-compute/numerics/neighbours'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { run, seek, trace } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'
import { fixture } from '../../fixtures'

type Step = { weights: number[]; means: number[][]; covariances: number[][][]; lower_bound: number }
type Fx = {
  x: number[][]
  init: number[][]
  kmeans: { centroids: number[][]; labels: number[]; inertia: number }
  gmm: Record<'full' | 'diagonal' | 'spherical', { weights: number[]; covariances: number[][][]; steps: Step[] }>
  linkage: Record<'single' | 'complete' | 'average' | 'ward', number[][]>
  leaves: number[]
  dbscan: { labels: number[]; core: number[] }
  optics: { ordering: number[]; reachability: number[]; core: number[]; predecessor: number[] }
  xs: number[][]
  spectral: number[]
  meanshift: { centres: number[][]; labels: number[] }
}
const fx = fixture<Fx>('unsupervised/clustering')
const X = tensor(fx.x)

function close(a: Tensor | number[], b: unknown, digits = 8) {
  const got = Array.isArray(a) ? a : toFlat(a)
  const want = (b as number[]).flat(3) as number[]
  expect(got.length).toBe(want.length)
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], digits))
}

const partition = (labels: Tensor | number[]) =>
  Array.from(canonicalLabels(Array.isArray(labels) ? labels : toFlat(labels)))

describe('k-means', () => {
  it('Lloyd from fixed centroids matches scikit-learn', () => {
    const m = kmeans({ k: 3, centroids: tensor(fx.init) }).fit(dataset(X))
    close(m.centroids, fx.kmeans.centroids)
    expect(toFlat(m.decide(X))).toEqual(fx.kmeans.labels)
    expect(m.inertia).toBeCloseTo(fx.kmeans.inertia, 8)
    expect(m.converged).toBe(true)
  })
  it('inertia never rises and the trace protocol holds', () => {
    const alg = kmeansSteps(X, { k: 3 })
    const t = trace(alg, {}, 50, { stream: stream(2), record: { inertia: (s) => s.inertia } })
    const inertia = toFlat(t.series.inertia)
    for (let i = 1; i < inertia.length; i++) expect(inertia[i]).toBeLessThanOrEqual(inertia[i - 1] + 1e-12)
    expectProtocol(alg, {}, { record: { inertia: (s) => s.inertia } })
    expectProtocol(
      alg,
      {
        centroids: tensor([
          [0, 0],
          [1, 1],
          [2, 2],
        ]),
      },
      { n: 6 },
    )
  })
  it('k-means++ is reproducible and its first pick is uniform', () => {
    const a = kmeansPlusPlus(stream(4), X, 3)
    expect(toFlat(kmeansPlusPlus(stream(4), X, 3).indices)).toEqual(toFlat(a.indices))
    const p = toRows(a.probabilities)
    expect(p[0].every((u) => Math.abs(u - 1 / 45) < 1e-15)).toBe(true)
    expect(p[1][toFlat(a.indices)[0]]).toBe(0)
    for (const row of p) expect(row.reduce((u, v) => u + v, 0)).toBeCloseTo(1, 12)
  })
  it('mini-batch k-means and k-medoids find the three groups', () => {
    const mb = miniBatchKMeans({ k: 3, steps: 60, batchSize: 10 }).fit(dataset(X), { stream: stream(1) })
    expect(partition(mb.decide(X))).toEqual(partition(fx.kmeans.labels))
    const pam = kMedoids({ k: 3 }).fit(dataset(X))
    expect(partition(pam.decide(X))).toEqual(partition(fx.kmeans.labels))
    const costs = toFlat(pam.training.series.cost)
    for (let i = 1; i < costs.length; i++) expect(costs[i]).toBeLessThanOrEqual(costs[i - 1])
  })
})

describe('mini-batch k-means and k-medoids, exactly', () => {
  it('one mini-batch step moves each centroid to the mean of its batch rows (per-centre rate 1/count)', () => {
    const alg = miniBatchKMeansSteps(X, { k: 3, batchSize: 16 })
    const s0 = alg.init({ centroids: tensor(fx.init) }, stream(5))
    const s1 = alg.step(s0, { t: 0, stream: stream(6) } as never)
    const rows = toRows(X)
    const c0 = toRows(s0.centroids)
    const c1 = toRows(s1.centroids)
    const batch = Array.from(toFlat(s1.batch))
    const nearestOf = (r: number[]) =>
      c0.map((c) => (c[0] - r[0]) ** 2 + (c[1] - r[1]) ** 2).reduce((b, v, j, a) => (v < a[b] ? j : b), 0)
    const counts = toFlat(s1.counts)
    expect(counts.reduce((a, b) => a + b, 0)).toBe(16)
    for (let j = 0; j < 3; j++) {
      const mine = batch.filter((i) => nearestOf(rows[i]) === j).map((i) => rows[i])
      expect(counts[j]).toBe(mine.length)
      if (mine.length === 0) expect(c1[j]).toEqual(c0[j])
      else
        for (let a = 0; a < 2; a++) expect(c1[j][a]).toBeCloseTo(mine.reduce((u, r) => u + r[a], 0) / mine.length, 12)
    }
  })

  it('PAM stops where no single swap lowers the cost, and BUILD starts at the 1-medoid', () => {
    const pts = toRows(X).slice(0, 12)
    const D = toRows(pairwiseDistances(tensor(pts), tensor(pts)))
    const cost = (m: number[]) => D.reduce((s, row) => s + Math.min(...m.map((j) => row[j])), 0)
    let best = Infinity
    for (let a = 0; a < 12; a++)
      for (let b = a + 1; b < 12; b++) for (let c = b + 1; c < 12; c++) best = Math.min(best, cost([a, b, c]))
    const final = run(kMedoidsSteps(pairwiseDistances(tensor(pts), tensor(pts)), { k: 3 }), {}, 100)
    expect(final.converged).toBe(true)
    const medoids = Array.from(toFlat(final.medoids))
    expect(final.cost).toBeCloseTo(cost(medoids), 10)
    // SWAP's fixed point: every (medoid, non-medoid) exchange costs at least as much. PAM is a local search, so the
    // exhaustive optimum bounds it from below.
    for (let m = 0; m < 3; m++)
      for (let o = 0; o < 12; o++)
        if (!medoids.includes(o))
          expect(cost(medoids.map((u, j) => (j === m ? o : u)))).toBeGreaterThanOrEqual(final.cost - 1e-12)
    expect(final.cost).toBeGreaterThanOrEqual(best - 1e-12)
    // BUILD's first medoid minimises the total distance to every point.
    const first = toFlat(
      kMedoidsSteps(pairwiseDistances(tensor(pts), tensor(pts)), { k: 1 }).init({}, stream(0)).medoids,
    )
    const sums = D.map((row) => row.reduce((u, v) => u + v, 0))
    expect(sums[first[0]]).toBe(Math.min(...sums))
  })
})

describe('Gaussian mixtures', () => {
  for (const kind of ['full', 'diagonal', 'spherical'] as const) {
    it(`EM from a fixed start matches scikit-learn (${kind})`, () => {
      const g = fx.gmm[kind]
      const alg = gaussianMixtureSteps(X, { k: 3, covariance: kind, tolerance: -1 })
      const init = { weights: tensor(g.weights), means: tensor(fx.init), covariances: tensor(g.covariances) }
      const t = trace(alg, init, 3)
      ;[1, 3].forEach((it, j) => {
        const s = t.steps[it]
        close(s.weights, g.steps[j].weights, 7)
        close(s.means, g.steps[j].means, 7)
        close(s.covariances, g.steps[j].covariances, 7)
        expect(t.steps[it - 1].logLikelihood).toBeCloseTo(g.steps[j].lower_bound, 7)
      })
    })
  }
  it('the log-likelihood rises and responsibilities sum to one', () => {
    const m = gaussianMixture({ k: 3 }).fit(dataset(X), { stream: stream(3) })
    const ll = toFlat(m.training.series.logLikelihood)
    for (let i = 1; i < ll.length; i++) expect(ll[i]).toBeGreaterThanOrEqual(ll[i - 1] - 1e-10)
    for (const row of toRows(m.responsibilities(X))) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const draws = m.sampleMixture(stream(9), 500)
    expect(draws.x.shape).toEqual([500, 2])
  })
})

describe('agglomerative clustering', () => {
  for (const method of ['single', 'complete', 'average', 'ward'] as const) {
    it(`${method} linkage matches SciPy`, () => close(linkage(X, method), fx.linkage[method]))
  }
  it('dendrogram leaves match SciPy; cuts give the expected partitions', () => {
    const Z = linkage(X, 'ward')
    expect(toFlat(dendrogram(Z).order)).toEqual(fx.leaves)
    const three = cutTree(Z, { clusters: 3 })
    expect(new Set(toFlat(three)).size).toBe(3)
    const h = toRows(Z)[41][2]
    expect(toFlat(cutTree(Z, { height: h }))).toEqual(toFlat(three))
    const tree = mergeTree(Z)
    expect(leaves(tree).sort((a, b) => a - b)).toEqual(Array.from({ length: 45 }, (_, i) => i))
    expect(tree.nodes[tree.root].size).toBe(45)
    const m = agglomerative({ linkage: 'ward', clusters: 3 }).fit(dataset(X))
    expect(toFlat(m.labels)).toEqual(toFlat(three))
    const alg = agglomerativeSteps(X, { linkage: 'average' })
    expect(toFlat(seek(alg, undefined, 10).labels)).toEqual(toFlat(run(alg, undefined, 10).labels))
  })
})

describe('density clustering', () => {
  it('DBSCAN matches scikit-learn', () => {
    const m = dbscan({ eps: 0.6, minSamples: 4 }).fit(dataset(X))
    expect(toFlat(m.labels)).toEqual(fx.dbscan.labels)
    const core = toFlat(m.roles).flatMap((r, i) => (r === CORE ? [i] : []))
    expect(core).toEqual(fx.dbscan.core)
  })
  it('OPTICS matches scikit-learn', () => {
    const m = optics({ minSamples: 4 }).fit(dataset(X))
    expect(toFlat(m.ordering)).toEqual(fx.optics.ordering)
    close(m.reachability, fx.optics.reachability)
    close(m.coreDistances, fx.optics.core)
    expect(toFlat(m.predecessor)).toEqual(fx.optics.predecessor)
  })
  it('spectral clustering and mean shift match scikit-learn partitions', () => {
    const s = spectralClustering({ k: 3, affinity: { kind: 'rbf', lengthscale: 1 } }).fit(dataset(tensor(fx.xs)), {
      stream: stream(0),
    })
    expect(partition(s.labels)).toEqual(partition(fx.spectral))
    expect(s.components).toBe(1)
    const ms = meanShift({ bandwidth: 1.5 }).fit(dataset(X))
    close(ms.centres, fx.meanshift.centres, 2)
    expect(toFlat(ms.decide(X))).toEqual(fx.meanshift.labels)
  })
})

describe('trace protocol', () => {
  it('every clustering algorithm follows it', () => {
    expectProtocol(miniBatchKMeansSteps(X, { k: 3, batchSize: 10 }), {}, { n: 10 })
    expectProtocol(kMedoidsSteps(pairwiseDistances(X, X), { k: 3 }), {}, { n: 5 })
    expectProtocol(gaussianMixtureSteps(X, { k: 3 }), {}, { n: 6 })
    expectProtocol(agglomerativeSteps(X, { linkage: 'ward' }), undefined, { n: 20 })
    expectProtocol(meanShiftSteps(X, { bandwidth: 1.5 }), {}, { n: 6 })
  })
})
