/**
 * Prototypical networks (Snell, Swersky and Zemel, 2017): few-shot classification by distance to class prototypes in a
 * learned embedding. In an N-way K-shot episode, each class's prototype is the mean embedding of its K support points,
 * c_k = (1/K) Σ f_φ(xᵢ), and a query x is classified by a softmax over negative squared distances,
 * p(y = k | x) ∝ exp(−‖f_φ(x) − c_k‖²). The embedding f_φ is trained episodically: every update draws a new episode
 * of new classes and minimises the cross-entropy of its queries, so the network learns a metric that transfers to
 * classes it has never seen rather than the classes themselves. With squared Euclidean distance the classifier is
 * linear in the embedding, with weights 2c_k and bias −‖c_k‖² (Snell et al., §2.4).
 *
 * The baseline is the same nearest-prototype rule on the raw inputs (f = identity).
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
  /** Classes per episode N (default 5), labelled support points per class K (default 1), query points per class (default 5). */
  ways?: number
  shots?: number
  queries?: number
  /** The radius range [rMin, rMax] (default [0.3, 3]) and the angular noise sd in radians (default 0.06). */
  radius?: readonly number[]
  angularNoise?: number
}

/** One N-way K-shot episode: support and query points with labels 0 … N − 1, and each class's direction. */
export interface FewShotEpisode {
  readonly support: { x: Tensor; y: Tensor }
  readonly query: { x: Tensor; y: Tensor }
  readonly angles: Float64Array
}

/**
 * A few-shot episode on 2-d inputs: each of N classes is a direction θ_c from the origin (24 evenly spaced slots with
 * jitter, so classes differ by at least 7.5°), and a point of class c sits at angle θ_c + ε, ε ~ N(0, σ²), and a radius
 * uniform on [rMin, rMax]. The class is the angle and the radius is a nuisance, so prototypes compared by raw Euclidean
 * distance confuse a far point with a near point of a neighbouring class; an embedding that discards the radius does not.
 * Every episode draws new classes.
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

/** The averaging matrix A [N, N·K] with A[k, i] = 1/K when support point i (grouped by class) belongs to class k. */
function averaging(labels: Int32Array, ways: number): Tensor {
  const A = new Float64Array(ways * labels.length)
  const counts = new Float64Array(ways)
  for (const c of labels) counts[c]++
  labels.forEach((c, i) => (A[c * labels.length + i] = 1 / counts[c]))
  return fromData(A, [ways, labels.length])
}

/**
 * The logits −‖f(x) − c_k‖² [n, N] of queries x [n, d] against prototypes from support (x, y) under an embedding
 * (any differentiable map of rows; the identity gives the raw-input rule).
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

const argmaxRows = (z: ArrayLike<number>, n: number, k: number) =>
  Int32Array.from({ length: n }, (_, i) => {
    let best = 0
    for (let j = 1; j < k; j++) if (z[i * k + j] > z[i * k + best]) best = j
    return best
  })

/** Options of `prototypicalRun`. */
export interface PrototypicalOptions extends Pick<FewShotEpisodeOptions, 'radius' | 'angularNoise'> {
  /** Test episodes: N ways (default 5), K shots (default 1), queries per class (default 5). */
  ways?: number
  shots?: number
  queries?: number
  /** Ways of the training episodes (default: `ways`; Snell et al. train with more ways than they test). */
  trainWays?: number
  /** Episodes trained on (default 1000), Adam step size (default 3e-3), hidden widths (default [32, 32]). */
  episodes?: number
  stepSize?: number
  hidden?: readonly number[]
  /** Held-out test episodes scoring accuracy (default 100). */
  testEpisodes?: number
  /** Grid cells per side of the prototype map (default 48) over [−box, box]² (default 3.2). */
  grid?: number
  box?: number
  checkpoints?: number
  seed?: number | string
}

/** One checkpoint: the demo episode in the embedding and the nearest-prototype map of the input plane. */
export interface PrototypicalCheckpoint {
  episode: number
  /** Embedded support [N·K × 2], query [N·Q × 2] and prototypes [N × 2] (the first two embedding coordinates). */
  support: Float64Array
  query: Float64Array
  prototypes: Float64Array
  /** The predicted class of each grid cell, row-major in (y, x). */
  map: Int32Array
  /** Accuracy on the test episodes. */
  accuracy: number
}

/** A prototypical-network run so far. */
export interface PrototypicalRun {
  episodes: number
  done: number
  finished: boolean
  ways: number
  gridX: Float64Array
  /** The demo episode in the input plane. */
  demo: { support: Float64Array; supportY: Int32Array; query: Float64Array; queryY: Int32Array; angles: Float64Array }
  /** The nearest-prototype map and test accuracy of the raw inputs (the baseline). */
  rawMap: Int32Array
  rawAccuracy: number
  /** Episode loss (query cross-entropy) and episode query accuracy per recorded episode. */
  history: { episode: number[]; loss: number[]; accuracy: number[] }
  checkpoints: PrototypicalCheckpoint[]
}

/** Train a prototypical network episodically on direction classes and yield snapshots (module docs). */
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
