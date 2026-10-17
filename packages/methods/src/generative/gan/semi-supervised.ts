/**
 * The semi-supervised GAN of Salimans, Goodfellow, Zaremba, Cheung, Radford and Chen ("Improved Techniques for Training
 * GANs", NeurIPS 2016, §3.1 and §5) on low-dimensional data, with a plain-data run for a worker.
 *
 * The discriminator is a classifier with $K$ logits $l_1, \dots, l_K$ and an implicit $(K+1)$-th logit fixed at 0 for
 * "generated", so $p(y = k \mid \xvec) = e^{l_k} / (Z(\xvec) + 1)$ with $Z = \sum_k e^{l_k}$, and
 * $D(\xvec) = Z / (Z + 1)$ is the probability that $\xvec$ is real. Its loss is the cross-entropy of the labelled
 * points over the $K$ real classes plus the GAN game value
 * $-\expect_{\text{real}} \log D(\xvec) - \expect_{\text{generated}} \log(1 - D(\xvec))$ on a batch of unlabelled
 * points and a batch of generated ones, with $\log D = -\operatorname{softplus}(-\log Z)$ and
 * $\log(1 - D) = -\operatorname{softplus}(\log Z)$. The generator is trained by feature matching,
 * $\lVert \operatorname{mean} \fvec(\xvec_{\text{real}}) - \operatorname{mean} \fvec(G(\zvec)) \rVert^2$ with $\fvec$
 * the discriminator's last hidden layer (§3.1), or by the non-saturating loss $-\operatorname{mean} \log D(G(\zvec))$.
 *
 * Two more sources of "generated" points, for comparison: `fixed`, a sampler that does not learn (Dai et al., 2017,
 * "Good Semi-supervised Learning that Requires a Bad GAN", §4: sample uniformly from a box with the points near the
 * data removed, the complement generator of `complementSampler`), and `none`, which drops the unsupervised term and
 * trains the classifier on the labels alone (both still take a zero-gradient generator update, so every source has
 * one state shape). The alternating updates are `aifn-compute/nn/training`'s `adversarialTraining`; networks are
 * `Mlp`s. Differences from the paper: no weight normalisation, no Gaussian noise on the discriminator's layers, Adam
 * for both players.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, normals, permutation, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  float64Data,
  fromData,
  logsumexp,
  mean,
  mul,
  shapeOf,
  square,
  sub,
  sum,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { live, type Algorithm } from 'aifn-compute/foundation/trace'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { adversarialTraining, type AdversarialTrainingState } from 'aifn-compute/nn/training'
import { softplus } from 'aifn-compute/numerics/special'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'

/** The networks: a generator, and a discriminator split into a feature trunk and a linear head of $K$ logits. */
export interface SemiSupervisedGan {
  /** The generator, latents ($L$ wide) to points ($d$ wide), ReLU units. */
  generator: Layer<Params[]>
  /** The discriminator's trunk, points to its last hidden layer $\fvec(\xvec)$, leaky-ReLU units. */
  features: Layer<Params[]>
  /** The discriminator's linear head, $\fvec(\xvec)$ to the $K$ real-class logits. */
  head: Layer<Params[]>
  /** The latent width $L$. */
  latent: number
  /** The data dimension $d$. */
  dimension: number
  /** The number of real classes $K$. */
  classes: number
}

/** The discriminator's parameters: the trunk's and the head's. */
export type CriticParams = { features: Params[]; head: Params[] }

/** Options of `semiSupervisedGan`. */
export interface SemiSupervisedGanOptions {
  /** The data dimension $d$ (default 2). */
  dimension?: number
  /** The number of real classes $K$ (default 2). */
  classes?: number
  /** Latent width $L$ (default 2). */
  latent?: number
  /** Hidden widths of both networks (default `[32, 32]`). */
  hidden?: readonly number[]
}

/**
 * The networks of a semi-supervised GAN: an `Mlp` generator with ReLU units, and a discriminator of an `Mlp` trunk
 * with leaky-ReLU units (also on its last layer, the features $\fvec$) and a linear head of $K$ logits.
 *
 * @param options The data dimension, the classes, the latent width and the hidden widths.
 * @returns The three networks and their sizes.
 *
 * @example The networks and the shapes of their outputs
 * const net = semiSupervisedGan({ hidden: [8] })
 * const { generator, critic } = initSemiSupervisedGan(net, stream(1))
 * print('generated:', generatedPoints(net, generator, tensor([[0, 0], [1, -1]])))
 * print('K + 1 probabilities:', classAndFakeProbabilities(net, critic, tensor([[0.5, 0.5]])))
 */
export function semiSupervisedGan(options: SemiSupervisedGanOptions = {}): SemiSupervisedGan {
  const d = options.dimension ?? 2
  const K = options.classes ?? 2
  const L = options.latent ?? 2
  const h = options.hidden ?? [32, 32]
  return {
    generator: Mlp([L, ...h, d], { activation: 'relu' }),
    features: Mlp([d, ...h], { activation: 'leakyRelu', outputActivation: 'leakyRelu' }),
    head: Mlp([h[h.length - 1], K]),
    latent: L,
    dimension: d,
    classes: K,
  }
}

/**
 * The $K$ real-class logits of the discriminator at points `x` (traced or not).
 *
 * @param net The networks.
 * @param critic The discriminator's parameters.
 * @param x The points, $n \times d$.
 * @returns The logits, $n \times K$.
 */
function classLogits(net: SemiSupervisedGan, critic: CriticParams, x: Value): Value {
  return net.head.apply(critic.head, net.features.apply(critic.features, x))
}

/**
 * The $K + 1$ class probabilities of the discriminator at points, from the logits $(l_1, \dots, l_K, 0)$: the $K$
 * real classes and, last, "generated", $1 / (Z + 1)$. Computed stably (shifted by the largest logit).
 *
 * @param net The networks.
 * @param critic The discriminator's parameters.
 * @param x The points, $n \times d$.
 * @returns Row-major $n \times (K + 1)$ probabilities, each row summing to 1.
 *
 * @example The probability of "generated" is $1 / (Z + 1)$
 * const net = semiSupervisedGan({ hidden: [4] })
 * const { critic } = initSemiSupervisedGan(net, stream(2))
 * print('p(class 0), p(class 1), p(generated):', classAndFakeProbabilities(net, critic, tensor([[0, 0], [2, 1]])))
 */
export function classAndFakeProbabilities(net: SemiSupervisedGan, critic: CriticParams, x: Tensor): Float64Array {
  const K = net.classes
  const l = float64Data(classLogits(net, critic, x) as Tensor)
  const n = l.length / K
  const out = new Float64Array(n * (K + 1))
  for (let i = 0; i < n; i++) {
    let top = 0
    for (let k = 0; k < K; k++) top = Math.max(top, l[i * K + k])
    let z = Math.exp(-top)
    for (let k = 0; k < K; k++) z += Math.exp(l[i * K + k] - top)
    for (let k = 0; k < K; k++) out[i * (K + 1) + k] = Math.exp(l[i * K + k] - top) / z
    out[i * (K + 1) + K] = Math.exp(-top) / z
  }
  return out
}

/**
 * $\log Z(\xvec) = \log \sum_k e^{l_k}$ of each row of logits.
 *
 * @param logits The logits, $n \times K$.
 * @returns $n$ values.
 */
const logZ = (logits: Value): Value => logsumexp(logits, -1)

/** Points that do not learn: a function of a stream and a count giving $n \times d$ points. */
export type FixedSampler = (s: Stream, n: number) => Tensor

/**
 * Dai et al.'s complement generator in two dimensions: uniform draws from the box $[x_0, x_1] \times [y_0, y_1]$,
 * keeping only those farther than `radius` from every row of `data`, by rejection in rounds of $4n$ candidates, round
 * $r$ drawn from `child(s, 'complement', r)`. Throws `ShapeError` unless the data have two columns, and the sampler
 * throws `NumericalError` when 50 rounds leave fewer than $n$ points (the box is nearly covered by the data's
 * neighbourhoods).
 *
 * @param data The points to keep away from, $m \times 2$.
 * @param box The box's horizontal and vertical ranges.
 * @param radius The distance $r$ every sampled point keeps from the data.
 * @returns A sampler `(s, n)` giving $n \times 2$ points.
 *
 * @example Points in the unit square away from its centre
 * const sampler = complementSampler(tensor([[0.5, 0.5]]), { x: [0, 1], y: [0, 1] }, 0.3)
 * const p = toArray(sampler(stream(1), 5))
 * print('points:', p)
 * print('distances from the centre:', p.map(([a, b]) => Math.hypot(a - 0.5, b - 0.5)))
 */
export function complementSampler(
  data: Tensor,
  box: { x: readonly [number, number]; y: readonly [number, number] },
  radius: number,
): FixedSampler {
  if (shapeOf(data).length !== 2 || shapeOf(data)[1] !== 2)
    throw new ShapeError('complementSampler', 'complementSampler: data must be m × 2 points', [shapeOf(data)])
  const D = float64Data(data)
  const m = D.length / 2
  const r2 = radius * radius
  return (s, n) => {
    const out: number[] = []
    for (let round = 0; round < 50 && out.length < 2 * n; round++) {
      const u = float64Data(uniform(child(s, 'complement', round), 0, 1, { shape: [4 * n, 2] }) as Tensor)
      for (let c = 0; c < 4 * n && out.length < 2 * n; c++) {
        const px = box.x[0] + (box.x[1] - box.x[0]) * u[2 * c]
        const py = box.y[0] + (box.y[1] - box.y[0]) * u[2 * c + 1]
        let far = true
        for (let j = 0; j < m && far; j++) if ((D[2 * j] - px) ** 2 + (D[2 * j + 1] - py) ** 2 < r2) far = false
        if (far) out.push(px, py)
      }
    }
    if (out.length < 2 * n)
      throw new NumericalError(
        'complementSampler',
        `complementSampler: only ${out.length / 2} of ${n} points lie farther than ${radius} from the data ` +
          'after 50 rounds',
        'not-converged',
      )
    return fromData(Float64Array.from(out), [n, 2])
  }
}

/** The data of a run. */
export interface SemiSupervisedGanProblem {
  /** Labelled inputs, $n_L \times d$. */
  labelledX: Tensor
  /** Their classes in $0, \dots, K - 1$. */
  labelledY: ArrayLike<number>
  /** Unlabelled inputs, $n_U \times d$. */
  unlabelledX: Tensor
  /** The number of real classes $K$. */
  classes: number
}

/** Where the "generated" points come from. */
export type FakeSource =
  | { kind: 'feature-matching' }
  | { kind: 'non-saturating' }
  | { kind: 'fixed'; sampler: FixedSampler }
  | { kind: 'none' }

/** Options of `semiSupervisedGanTraining`. */
export interface SemiSupervisedGanTrainingOptions {
  /** The networks. */
  net: SemiSupervisedGan
  /** The generator's game, a fixed sampler, or none (default feature matching). */
  fakes?: FakeSource
  /** Unlabelled and generated points per update (default 64, at most $n_U$). */
  batch?: number
  /** Weight of the unsupervised term in the discriminator's loss (default 1). */
  weight?: number
  /** The discriminator's update rule (default Adam, step 3e-3, $\beta_1 = 0.5$). */
  criticOptimizer?: UpdateRule<unknown>
  /** The generator's update rule (default Adam, step 3e-3, $\beta_1 = 0.5$). */
  generatorOptimizer?: UpdateRule<unknown>
}

/** The state: `adversarialTraining`'s, whose critic parameters are a `CriticParams`. */
export type SemiSupervisedGanState = AdversarialTrainingState<Params[], Params> & Status

/**
 * Semi-supervised GAN training (see the file comment), as `adversarialTraining`: each step one discriminator update
 * then one generator update. A step draws the unlabelled batch from `'real'`, the latents from `'latent'` and fixed
 * fakes from `'fixed'` under each player's stream. Start from `initSemiSupervisedGan(net, s)`.
 *
 * @param problem The labelled and unlabelled points and the number of classes.
 * @param options The networks, the source of generated points, the batch, the unsupervised weight and the update
 *   rules.
 * @returns The algorithm, to run with `run`, `trace` or `live` from `{ generator, critic }`.
 *
 * @example Twenty updates on six points
 * const problem = {
 *   labelledX: tensor([[-1, 0], [1, 0]]),
 *   labelledY: [0, 1],
 *   unlabelledX: tensor([[-1.2, 0.3], [-0.8, -0.2], [0.9, 0.4], [1.1, -0.3]]),
 *   classes: 2,
 * }
 * const net = semiSupervisedGan({ hidden: [8] })
 * const s = run(semiSupervisedGanTraining(problem, { net }), initSemiSupervisedGan(net, stream(1)), 20)
 * print('discriminator loss:', s.criticLoss, ' feature-matching loss:', s.generatorLoss)
 */
export function semiSupervisedGanTraining(
  problem: SemiSupervisedGanProblem,
  options: SemiSupervisedGanTrainingOptions,
): Algorithm<{ generator: Params[]; critic: Params }, SemiSupervisedGanState> {
  const { net } = options
  const fakes = options.fakes ?? { kind: 'feature-matching' }
  const w = options.weight ?? 1
  const [nU, d] = shapeOf(problem.unlabelledX) as [number, number]
  const B = Math.min(options.batch ?? 64, nU)
  const U = float64Data(problem.unlabelledX)
  const yL = Array.from(problem.labelledY)
  const adam = () => adamRule({ stepSize: 3e-3, beta1: 0.5 })

  // A uniform batch of B unlabelled points.
  const realBatch = (s: Stream) => {
    const order = float64Data(permutation(child(s, 'real'), nU))
    const out = new Float64Array(B * d)
    for (let r = 0; r < B; r++) out.set(U.subarray(order[r] * d, order[r] * d + d), r * d)
    return fromData(out, [B, d])
  }
  // A batch of B generated points: from the fixed sampler, or the generator at fresh latents.
  const generated = (g: Params[], s: Stream): Value =>
    fakes.kind === 'fixed'
      ? fakes.sampler(child(s, 'fixed'), B)
      : net.generator.apply(g, normals(child(s, 'latent'), [B, net.latent]))

  const criticLoss = (critic: Params, g: Params[], s: Stream): Value => {
    const c = critic as unknown as CriticParams
    const supervised = softmaxCrossEntropy(classLogits(net, c, problem.labelledX), yL)
    if (fakes.kind === 'none') return supervised
    const fake = generated(g, s)
    const real = logZ(classLogits(net, c, realBatch(s)))
    const gen = logZ(classLogits(net, c, fake))
    // −log D(real) = softplus(−log Z); −log(1 − D(fake)) = softplus(log Z).
    const unsup = add(mean(softplus(mul(-1, real))), mean(softplus(gen)))
    return add(supervised, mul(w, unsup))
  }

  const generatorLoss = (g: Params[], critic: Params, s: Stream): Value => {
    const c = critic as unknown as CriticParams
    // A source that does not learn still takes a (zero-gradient) generator update.
    if (fakes.kind === 'none' || fakes.kind === 'fixed')
      return mul(0, sum(net.generator.apply(g, fromData(new Float64Array(net.latent), [1, net.latent]))))
    const fake = net.generator.apply(g, normals(child(s, 'latent'), [B, net.latent]))
    if (fakes.kind === 'non-saturating') return mean(softplus(mul(-1, logZ(classLogits(net, c, fake)))))
    const fr = mean(net.features.apply(c.features, realBatch(s)), 0)
    const ff = mean(net.features.apply(c.features, fake), 0)
    return sum(square(sub(fr, ff)))
  }

  return adversarialTraining<Params[], Params>({
    criticLoss,
    generatorLoss,
    criticOptimizer: options.criticOptimizer ?? adam(),
    generatorOptimizer: options.generatorOptimizer ?? adam(),
  })
}

/**
 * Initial parameters for `semiSupervisedGanTraining` from one stream: the generator from child `'generator'`, the
 * discriminator's trunk from `'features'` and its head from `'head'`.
 *
 * @param net The networks.
 * @param s The stream.
 * @returns `{ generator, critic }`; the critic is a `CriticParams`, typed as a parameter tree.
 *
 * @example Parameters of both players
 * const net = semiSupervisedGan({ hidden: [4] })
 * const start = initSemiSupervisedGan(net, stream(1))
 * print('generator layers:', start.generator.length, ' critic parts:', Object.keys(start.critic))
 */
export function initSemiSupervisedGan(net: SemiSupervisedGan, s: Stream): { generator: Params[]; critic: Params } {
  return {
    generator: net.generator.init(child(s, 'generator')),
    critic: {
      features: net.features.init(child(s, 'features')),
      head: net.head.init(child(s, 'head')),
    } as unknown as Params,
  }
}

/**
 * The generator's points at given latents, as a flat array (the plain-array form of `sampleGenerator`).
 *
 * @param net The networks.
 * @param generator The generator's parameters.
 * @param z The latents, $n \times L$.
 * @returns Row-major $n \times d$ points.
 *
 * @example The points of an untrained generator
 * const net = semiSupervisedGan({ hidden: [4] })
 * const { generator } = initSemiSupervisedGan(net, stream(1))
 * print(generatedPoints(net, generator, normal(stream(2), 0, 1, { shape: [3, 2] })))
 */
export function generatedPoints(net: SemiSupervisedGan, generator: Params[], z: Tensor): Float64Array {
  return Float64Array.from(float64Data(net.generator.apply(generator, z) as Tensor))
}

// ── A plain-data run for a worker ────────────────────────────────────────────────────────────────────────────────────

/** Options of `semiSupervisedGanRun`; plain data, so a worker task can carry them. */
export type SemiSupervisedGanRunOptions = {
  /** The source of generated points: a generator's game, the complement sampler, or none (default feature matching). */
  fakes?: 'feature-matching' | 'non-saturating' | 'complement' | 'none'
  /** The complement sampler's box (default $[-1.5, 2.5] \times [-1, 1.5]$, the two-moons box). */
  box?: { x: readonly [number, number]; y: readonly [number, number] }
  /** The complement sampler's radius $r$ (default 0.25). */
  radius?: number
  /** The points the complement sampler keeps away from (default the labelled and unlabelled points). */
  avoid?: Tensor
  /** Hidden widths of both networks (default `[32, 32]`). */
  hidden?: readonly number[]
  /** Latent width (default 2). */
  latent?: number
  /** Updates (default 500). */
  steps?: number
  /** Adam's step size for both players, with $\beta_1 = 0.5$ (default 3e-3). */
  rate?: number
  /** Unlabelled and generated points per update (default 64). */
  batch?: number
  /** Weight of the unsupervised term (default 1). */
  weight?: number
  /** The key of the stream the initial parameters come from, `stream(seed)` (default 0). */
  seed?: number | string
  /** The key of the run's root stream (default 0, as `live`'s default). */
  stepSeed?: number | string
  /** The key of the stream of the shown points: the latents, or the complement sampler's draws (default 0). */
  shownSeed?: number | string
  /** Generated points shown per checkpoint (default 150). */
  shown?: number
  /** Steps between checkpoints (default 10); step 0 and the last are always kept. */
  every?: number
  /** True classes of the unlabelled points, for the accuracy. */
  truth?: ArrayLike<number>
  /** Further inputs ($m \times d$, e.g. a grid) whose $K + 1$ probabilities each checkpoint holds. */
  evaluate?: Tensor
}

/** One checkpoint of `semiSupervisedGanRun`: plain arrays only. */
export type SemiSupervisedGanCheckpoint = {
  /** Updates made (0 for the initial networks). */
  step: number
  /** The discriminator's loss before its last update (NaN at step 0). */
  criticLoss: number
  /** The generator's loss before its last update (NaN at step 0). */
  generatorLoss: number
  /** The shown generated points, row-major $n \times d$ (empty for `none`; fixed for `complement`). */
  fakes: Float64Array
  /** The $K + 1$ probabilities on the unlabelled points, row-major. */
  probabilities: Float64Array
  /** The $K + 1$ probabilities on the `evaluate` inputs, row-major (empty without them). */
  evaluate: Float64Array
  /** The arg max accuracy over the $K$ real classes on the unlabelled points (NaN without `truth`). */
  accuracy: number
  /** Whether training diverged, which ends the run. */
  diverged: boolean
}

/**
 * Train a semi-supervised GAN from plain data and yield a checkpoint at step 0, every `every` steps, at the last step
 * and on divergence (which ends the run). Both players use Adam with step size `rate` and $\beta_1 = 0.5$; the
 * initial parameters come from `initSemiSupervisedGan(net, stream(seed))`, the shown latents from
 * `normals(stream(shownSeed), [shown, L])` (or, for `complement`, the sampler's draws from `stream(shownSeed)`).
 * Deterministic in its seeds.
 *
 * @param problem The labelled and unlabelled points and the number of classes.
 * @param options The source of generated points, the networks, the training, the seeds and the checkpoints.
 * @returns A generator of checkpoints.
 *
 * @example Accuracy of a short run on six points
 * const problem = {
 *   labelledX: tensor([[-1, 0], [1, 0]]),
 *   labelledY: [0, 1],
 *   unlabelledX: tensor([[-1.2, 0.3], [-0.8, -0.2], [0.9, 0.4], [1.1, -0.3]]),
 *   classes: 2,
 * }
 * const options = { hidden: [8], steps: 20, every: 10, shown: 4, truth: [0, 0, 1, 1] }
 * for (const c of semiSupervisedGanRun(problem, options)) print('step', c.step, ' accuracy', c.accuracy)
 */
export function* semiSupervisedGanRun(
  problem: SemiSupervisedGanProblem,
  options: SemiSupervisedGanRunOptions = {},
): Generator<SemiSupervisedGanCheckpoint> {
  const {
    fakes: source = 'feature-matching',
    box = { x: [-1.5, 2.5] as const, y: [-1, 1.5] as const },
    radius = 0.25,
    steps = 500,
    rate = 3e-3,
    seed = 0,
    stepSeed = 0,
    shownSeed = 0,
    shown = 150,
    every = 10,
    truth,
    evaluate,
  } = options
  const d = shapeOf(problem.unlabelledX)[1]
  const { latent, hidden } = options
  const net = semiSupervisedGan({ dimension: d, classes: problem.classes, latent, hidden })
  const K = net.classes
  const avoid =
    options.avoid ??
    fromData(Float64Array.from([...float64Data(problem.labelledX), ...float64Data(problem.unlabelledX)]), [
      shapeOf(problem.labelledX)[0] + shapeOf(problem.unlabelledX)[0],
      d,
    ])
  const sampler = source === 'complement' ? complementSampler(avoid, box, radius) : null
  const fakes: FakeSource = sampler ? { kind: 'fixed', sampler } : ({ kind: source } as FakeSource)
  const rule = () => adamRule({ stepSize: rate, beta1: 0.5 }) as UpdateRule<unknown>
  const alg = semiSupervisedGanTraining(problem, {
    net,
    fakes,
    batch: options.batch,
    weight: options.weight,
    criticOptimizer: rule(),
    generatorOptimizer: rule(),
  })
  const z = normals(stream(shownSeed), [shown, net.latent])
  const shownFixed = sampler ? Float64Array.from(float64Data(sampler(stream(shownSeed), shown))) : null
  const accuracy = (p: Float64Array): number => {
    if (!truth) return NaN
    let right = 0
    for (let i = 0; i < truth.length; i++) {
      let best = 0
      for (let k = 1; k < K; k++) if (p[i * (K + 1) + k] > p[i * (K + 1) + best]) best = k
      if (best === truth[i]) right++
    }
    return truth.length ? right / truth.length : NaN
  }
  const it = live(alg, initSemiSupervisedGan(net, stream(seed)), { stream: stream(stepSeed) })
  for (let k = 0; k <= steps; k++) {
    const next = it.next()
    if (next.done) return
    const st = next.value.state
    const critic = st.critic as unknown as CriticParams
    if (k % every === 0 || k === steps || st.diverged) {
      const probabilities = classAndFakeProbabilities(net, critic, problem.unlabelledX)
      yield {
        step: k,
        criticLoss: Number(st.criticLoss),
        generatorLoss: Number(st.generatorLoss),
        fakes: source === 'none' ? new Float64Array(0) : (shownFixed ?? generatedPoints(net, st.generator, z)),
        probabilities,
        evaluate: evaluate ? classAndFakeProbabilities(net, critic, evaluate) : new Float64Array(0),
        accuracy: accuracy(probabilities),
        diverged: st.diverged,
      }
    }
    if (st.diverged) return
  }
}
