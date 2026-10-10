/**
 * Prototypical networks (Snell, Swersky and Zemel, 2017): few-shot classification by distance to class prototypes in a
 * learned embedding. In an $N$-way $K$-shot episode, each class's prototype is the mean embedding of its $K$ support
 * points, $\cvec_k = \frac{1}{K} \sum_i f_{\phivec}(\xvec_i)$, and a query $\xvec$ is classified by a softmax over
 * negative squared distances, $p(y = k \mid \xvec) \propto \exp(-\lVert f_{\phivec}(\xvec) - \cvec_k \rVert^2)$. The
 * embedding $f_{\phivec}$ is trained episodically: every update draws a new episode of new classes and minimises the
 * cross-entropy of its queries, so the network learns a metric that transfers to classes it has never seen rather
 * than the classes themselves. With squared Euclidean distance the classifier is linear in the embedding, with
 * weights $2\cvec_k$ and bias $-\lVert \cvec_k \rVert^2$ (Snell et al., §2.4).
 *
 * The baseline is the same nearest-prototype rule on the raw inputs ($f$ the identity).
 */

import type { Params } from 'aifn-compute/foundation/pytree'
import { child, standardNormals, stream, units, type Stream } from 'aifn-compute/foundation/random'
import { fromData, matmul, neg, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { scaledSquaredDistances } from 'aifn-compute/learning/kernels'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { adamTrainer } from './shared'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `fewShotEpisode`. */
export interface FewShotEpisodeOptions {
  /** Classes per episode $N$ (default 5, at most 24). */
  ways?: number
  /** Labelled support points per class $K$ (default 1). */
  shots?: number
  /** Query points per class (default 5). */
  queries?: number
  /** The radius range $[r_{\min}, r_{\max}]$ (default $[0.3, 3]$). */
  radius?: readonly number[]
  /** The angular noise's standard deviation $\sigma$, radians (default 0.06). */
  angularNoise?: number
}

/**
 * One $N$-way $K$-shot episode: support and query points (inputs $NK \times 2$ and $NQ \times 2$, grouped by class)
 * with labels $0, \dots, N - 1$, and each class's direction.
 */
export interface FewShotEpisode {
  /** The labelled support points, $K$ per class. */
  readonly support: { x: Tensor; y: Tensor }
  /** The query points to classify, `queries` per class. */
  readonly query: { x: Tensor; y: Tensor }
  /** Each class's direction $\theta_c$, radians. */
  readonly angles: Float64Array
}

/**
 * A few-shot episode on 2-d inputs: each of $N$ classes is a direction $\theta_c$ from the origin (24 evenly spaced
 * slots with jitter, so classes differ by at least 7.5 degrees), and a point of class $c$ sits at angle
 * $\theta_c + \varepsilon$, $\varepsilon \sim \Gauss(0, \sigma^2)$, and a radius uniform on $[r_{\min}, r_{\max}]$.
 * The class is the angle and the radius is a nuisance, so prototypes compared by raw Euclidean distance confuse a far
 * point with a near point of a neighbouring class; an embedding that discards the radius does not. Every episode
 * draws new classes. Throws `DomainError` for more than 24 ways.
 *
 * @param s The stream the episode is drawn from.
 * @param options The episode's size and geometry.
 * @returns The episode.
 *
 * @example A 3-way 1-shot episode with two queries per class
 * const episode = fewShotEpisode(stream(1), { ways: 3, shots: 1, queries: 2 })
 * print('class directions, degrees:', Array.from(episode.angles, (a) => (a * 180) / Math.PI))
 * print('support points:', episode.support.x)
 * print('query labels:', episode.query.y)
 */
export function fewShotEpisode(s: Stream, options: FewShotEpisodeOptions = {}): FewShotEpisode {
  const { ways = 5, shots = 1, queries = 5, radius = [0.3, 3], angularNoise = 0.06 } = options
  if (ways > 24) throw new DomainError('fewShotEpisode', 'fewShotEpisode: at most 24 ways')
  const slotU = units(child(s, 'slots'), 24)
  const slots = Array.from({ length: 24 }, (_, i) => i).sort((p, q) => slotU[p] - slotU[q])
  const jitter = units(child(s, 'jitter'), ways)
  const angles = Float64Array.from(
    { length: ways },
    (_, c) => ((slots[c] + 0.5 * (jitter[c] - 0.5)) * 2 * Math.PI) / 24,
  )
  const draw = (name: string, per: number) => {
    const n = ways * per
    const u = units(child(s, name, 'radius'), n)
    const e = standardNormals(child(s, name, 'angle'), n)
    const x = new Float64Array(2 * n)
    const y = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      const c = Math.floor(i / per)
      const r = radius[0] + (radius[1] - radius[0]) * u[i]
      const a = angles[c] + angularNoise * e[i]
      x[2 * i] = r * Math.cos(a)
      x[2 * i + 1] = r * Math.sin(a)
      y[i] = c
    }
    return { x: fromData(x, [n, 2]), y: fromData(y, [n]) }
  }
  return { support: draw('support', shots), query: draw('query', queries), angles }
}

/**
 * The averaging matrix $\Amat$, with $A_{ki} = 1/n_k$ when support point $i$ belongs to class $k$ ($n_k$ the
 * points of class $k$) and 0 otherwise, so that $\Amat$ times the embedded support rows gives the prototypes.
 *
 * @param labels The support labels, $0, \dots, N - 1$, in any order.
 * @param ways The number of classes $N$.
 * @returns $\Amat$, $N \times$ the number of support points; a class with no support point has a row of zeros.
 */
function averaging(labels: Int32Array, ways: number): Tensor {
  const A = new Float64Array(ways * labels.length)
  const counts = new Float64Array(ways)
  for (const c of labels) counts[c]++
  labels.forEach((c, i) => (A[c * labels.length + i] = 1 / counts[c]))
  return fromData(A, [ways, labels.length])
}

/**
 * The logits $-\lVert f(\xvec) - \cvec_k \rVert^2$ of queries against the prototypes of a support set under an
 * embedding $f$, differentiable in whatever the embedding is.
 *
 * @param embed The embedding $f$: any differentiable map of rows (the identity gives the raw-input rule).
 * @param support The support points `x` (rows) and their labels `y`, $0, \dots, N - 1$.
 * @param query The queries, $n \times d$.
 * @param ways The number of classes $N$.
 * @returns The $n \times N$ logits; their softmax is $p(y = k \mid \xvec)$.
 *
 * @example Raw-input prototypes at 0 and 2: each query is nearer its own
 * const support = { x: tensor([[0, 0], [2, 0]]), y: [0, 1] }
 * print(prototypeLogits((x) => x, support, tensor([[0.5, 0], [1.8, 0]]), 2))
 */
export function prototypeLogits(
  embed: (x: Value) => Value,
  support: { x: Value; y: ArrayLike<number> },
  query: Value,
  ways: number,
): Value {
  const prototypes = matmul(averaging(Int32Array.from(support.y), ways), embed(support.x))
  return neg(scaledSquaredDistances(embed(query), prototypes, 1))
}

/**
 * The index of the largest entry of each row (the first on a tie).
 *
 * @param z The values, row-major $n \times k$.
 * @param n The number of rows.
 * @param k The number of columns.
 * @returns $n$ column indices.
 */
const argmaxRows = (z: ArrayLike<number>, n: number, k: number) =>
  Int32Array.from({ length: n }, (_, i) => {
    let best = 0
    for (let j = 1; j < k; j++) if (z[i * k + j] > z[i * k + best]) best = j
    return best
  })

/** Options of `prototypicalRun`. */
export interface PrototypicalOptions extends Pick<FewShotEpisodeOptions, 'radius' | 'angularNoise'> {
  /** Ways $N$ of the test and demo episodes (default 5). */
  ways?: number
  /** Shots $K$ of every episode (default 1). */
  shots?: number
  /** Queries per class of every episode (default 5). */
  queries?: number
  /** Ways of the training episodes (default: `ways`; Snell et al. train with more ways than they test). */
  trainWays?: number
  /** Episodes trained on, one Adam update each (default 1000). */
  episodes?: number
  /** Adam's step size (default 3e-3). */
  stepSize?: number
  /** Hidden widths of the embedding MLP (default $[32, 32]$, tanh; the embedding is 2-d). */
  hidden?: readonly number[]
  /** Held-out test episodes scoring accuracy (default 100). */
  testEpisodes?: number
  /** Grid cells per side of the prototype map (default 48). */
  grid?: number
  /** The grid's half-width: it spans $[-\mathit{box}, \mathit{box}]^2$ (default 3.2). */
  box?: number
  /** About how many checkpoints to keep after the start (default 20): one every `episodes / checkpoints` episodes. */
  checkpoints?: number
  /** The root stream's seed (default 0): the run is deterministic in it. */
  seed?: number | string
}

/** One checkpoint: the demo episode in the embedding and the nearest-prototype map of the input plane. */
export interface PrototypicalCheckpoint {
  /** Episodes trained on. */
  episode: number
  /** The embedded support points, row-major $NK \times 2$ (the embedding is 2-d). */
  support: Float64Array
  /** The embedded query points, row-major $NQ \times 2$. */
  query: Float64Array
  /** The prototypes, row-major $N \times 2$. */
  prototypes: Float64Array
  /** The predicted class of each grid cell, row-major in $(y, x)$. */
  map: Int32Array
  /** Mean query accuracy over the test episodes. */
  accuracy: number
}

/** A prototypical-network run so far. */
export interface PrototypicalRun {
  /** The episodes the run will train on. */
  episodes: number
  /** The episodes trained on. */
  done: number
  /** True for the last snapshot. */
  finished: boolean
  /** Ways of the test and demo episodes. */
  ways: number
  /** The grid's coordinates along each axis, `grid` values. */
  gridX: Float64Array
  /** The demo episode in the input plane: points row-major, 2 per row. */
  demo: { support: Float64Array; supportY: Int32Array; query: Float64Array; queryY: Int32Array; angles: Float64Array }
  /** The nearest-prototype map of the raw inputs (the baseline), row-major in $(y, x)$. */
  rawMap: Int32Array
  /** The baseline's mean query accuracy over the test episodes. */
  rawAccuracy: number
  /**
   * Per recorded episode (about 200 of them): the query cross-entropy before the update, and the query accuracy on
   * that episode after it.
   */
  history: { episode: number[]; loss: number[]; accuracy: number[] }
  /** The checkpoints so far, the first at episode 0. */
  checkpoints: PrototypicalCheckpoint[]
}

/**
 * Train a prototypical network episodically on direction classes and yield snapshots: at the start and at each
 * checkpoint (the last also returned). Each training episode is a fresh `fewShotEpisode` with `trainWays` ways; the
 * test episodes and a demo episode are drawn once. Everything comes from the root stream of `seed`, so the run is
 * deterministic in it.
 *
 * @param options The episodes, the training, the embedding and the grid.
 * @returns A generator of snapshots of the run.
 *
 * @example Test accuracy of raw-input prototypes, and of the embedding before and after forty training episodes
 * const options = { episodes: 40, ways: 3, testEpisodes: 10, hidden: [8], grid: 2, checkpoints: 1 }
 * let run
 * for (const snapshot of prototypicalRun(options)) run = snapshot
 * print('raw-input prototypes:', run.rawAccuracy)
 * print('learned embedding, before and after training:', run.checkpoints.map((c) => c.accuracy))
 */
export function* prototypicalRun(options: PrototypicalOptions = {}): Generator<PrototypicalRun, PrototypicalRun> {
  const { ways = 5, shots = 1, queries = 5, episodes = 1000, stepSize = 3e-3, hidden = [32, 32] } = options
  const { testEpisodes = 100, grid = 48, box = 3.2, checkpoints = 20, seed = 0, radius, angularNoise } = options
  const trainWays = options.trainWays ?? ways
  const root = stream(seed)
  const geometry = { radius, angularNoise }
  const net: Layer<Params[]> = Mlp([2, ...hidden, 2], { activation: 'tanh' })
  let params: Params[] = net.init(child(root, 'init'))
  const trainer = adamTrainer<Params[]>(stepSize, params)
  const embedWith = (p: Params[]) => (x: Value) => net.apply(p, x)
  const identity = (x: Value) => x
  const tests: FewShotEpisode[] = Array.from({ length: testEpisodes }, (_, i) =>
    fewShotEpisode(child(root, 'test', i), { ways, shots, queries, ...geometry }),
  )
  const accuracyOn = (embed: (x: Value) => Value, e: FewShotEpisode, k = ways) => {
    const y = toFlat(e.query.y)
    const z = toFlat(unwrap(prototypeLogits(embed, { x: e.support.x, y: toFlat(e.support.y) }, e.query.x, k)) as Tensor)
    const pred = argmaxRows(z, y.length, k)
    return pred.reduce((h, p, i) => h + (p === y[i] ? 1 : 0), 0) / y.length
  }
  const testAccuracy = (embed: (x: Value) => Value) =>
    tests.reduce((t, e) => t + accuracyOn(embed, e), 0) / tests.length
  const demo = fewShotEpisode(child(root, 'demo'), { ways, shots, queries, ...geometry })
  const gridX = Float64Array.from({ length: grid }, (_, i) => -box + (2 * box * i) / (grid - 1))
  const cells = new Float64Array(2 * grid * grid)
  for (let r = 0; r < grid; r++)
    for (let c = 0; c < grid; c++) {
      cells[2 * (r * grid + c)] = gridX[c]
      cells[2 * (r * grid + c) + 1] = gridX[r]
    }
  const cellX = fromData(cells, [grid * grid, 2])
  const demoSupport = { x: demo.support.x, y: toFlat(demo.support.y) }
  const mapOf = (embed: (x: Value) => Value) =>
    argmaxRows(toFlat(unwrap(prototypeLogits(embed, demoSupport, cellX, ways)) as Tensor), grid * grid, ways)
  const flat = (v: Value) => Float64Array.from(toFlat(unwrap(v) as Tensor))
  const checkpoint = (t: number): PrototypicalCheckpoint => {
    const embed = embedWith(params)
    const support = flat(embed(demo.support.x))
    return {
      episode: t,
      support,
      query: flat(embed(demo.query.x)),
      prototypes: flat(
        matmul(averaging(Int32Array.from(demoSupport.y), ways), fromData(support, [support.length / 2, 2])),
      ),
      map: mapOf(embed),
      accuracy: testAccuracy(embed),
    }
  }
  const history = { episode: [] as number[], loss: [] as number[], accuracy: [] as number[] }
  const shots_: PrototypicalCheckpoint[] = [checkpoint(0)]
  const base = {
    episodes,
    ways,
    gridX,
    demo: {
      support: flat(demo.support.x),
      supportY: Int32Array.from(toFlat(demo.support.y)),
      query: flat(demo.query.x),
      queryY: Int32Array.from(toFlat(demo.query.y)),
      angles: demo.angles,
    },
    rawMap: mapOf(identity),
    rawAccuracy: testAccuracy(identity),
  }
  const snapshot = (done: number, finished: boolean): PrototypicalRun => ({
    ...base,
    done,
    finished,
    history: { episode: [...history.episode], loss: [...history.loss], accuracy: [...history.accuracy] },
    checkpoints: shots_.slice(),
  })
  yield snapshot(0, false)
  const every = Math.max(1, Math.round(episodes / checkpoints))
  const recordEvery = Math.max(1, Math.floor(episodes / 200))
  for (let t = 1; t <= episodes; t++) {
    const e = fewShotEpisode(child(root, 'train', t), { ways: trainWays, shots, queries, ...geometry })
    const sy = toFlat(e.support.y)
    let lossValue = 0
    params = trainer.step(params, (p) => {
      const logits = prototypeLogits(embedWith(p), { x: e.support.x, y: sy }, e.query.x, trainWays)
      const loss = softmaxCrossEntropy(logits, e.query.y, { reduction: 'mean' })
      const u = unwrap(loss)
      lossValue = typeof u === 'number' ? u : toFlat(u as Tensor)[0]
      return loss
    })
    if (t % recordEvery === 0 || t === episodes) {
      history.episode.push(t)
      history.loss.push(lossValue)
      // Scored after the update, on the episode just trained on.
      history.accuracy.push(accuracyOn(embedWith(params), e, trainWays))
    }
    if (t % every === 0 || t === episodes) {
      shots_.push(checkpoint(t))
      yield snapshot(t, t === episodes)
    }
  }
  return snapshot(episodes, true)
}
