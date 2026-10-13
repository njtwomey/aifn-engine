/**
 * A learning-rate sweep of a small network: train it once per learning rate and seed, with everything else fixed, and
 * score each run by its validation loss. Bengio (2012) calls the learning rate the single most important
 * hyperparameter to tune; the sweep shows why, and how its best value moves with the batch size and the optimiser.
 *
 * The task is the two-moons classification problem (`moons` of `aifn-methods/data/synthetic`, as scikit-learn's
 * `make_moons`): a training set and a larger validation set from the same distribution. The network is a ReLU
 * multilayer perceptron with He-normal initialisation and one logit output, trained by minibatch Adam or SGD with
 * momentum on the binary cross-entropy with logits, at a constant learning rate for a fixed number of steps; a run
 * whose loss blows up is flagged as diverged. The seed sets the initial weights and the order of the minibatches.
 *
 * Streams are named from a seed string, so a page that composed the same sweep from `trainingLoop` gets the same
 * numbers: the training set from `stream(seed + '/train')`, the validation set from `stream(seed + '/validation')`,
 * and run $k = 1, 2, \dots$ initialised from `stream(seed + '/init/' + k)` and shuffled from
 * `stream(seed + '/order/' + k)`, for every learning rate.
 */

import type { Params } from 'aifn-compute/foundation/pytree'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, normal, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { heNormal } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, sgdRule } from 'aifn-compute/optim/first-order'

/** Options of `learningRateSweep`. */
export interface LearningRateSweepOptions {
  /** The task: `moons`, two interleaving half circles (the default and, for now, the only one). */
  task?: 'moons'
  /** The hidden layers' widths (default `[16, 16]`). */
  widths?: readonly number[]
  /** `adam` (the default) or `sgd`, SGD with momentum. */
  optimiser?: 'adam' | 'sgd'
  /** SGD's momentum (default 0.9); ignored by Adam. */
  momentum?: number
  /** The learning rates (default $10^{-3.5}, 10^{-3}, \dots, 10^{0.5}$, a half-decade grid). */
  rates?: readonly number[]
  /** The number of seeds per learning rate (default 3). */
  seeds?: number
  /** The minibatch size (default 32). */
  batch?: number
  /** The training steps of every run (default 200). */
  steps?: number
  /** Training points (default 200). */
  train?: number
  /** Validation points (default 1000). */
  validation?: number
  /** The standard deviation of the moons' noise (default 0.25). */
  noise?: number
  /** The name the streams are made from (default `lr-sweep`; see the file's introduction). */
  seed?: string
}

/** A learning-rate sweep's results. */
export interface LearningRateSweep {
  /** The learning rates, in the order given. */
  rates: number[]
  /** The validation cross-entropy of each run, `[rate][seed]`; NaN for a run that diverged. */
  losses: number[][]
  /** Whether each run diverged, `[rate][seed]`. */
  diverged: boolean[][]
}

/** The default half-decade grid of learning rates, $10^{-3.5}$ to $10^{0.5}$. */
const RATES = [-3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5].map((e) => 10 ** e)

/**
 * Two moons as `moons` of `aifn-methods/data/synthetic` draws them with even spacing: $\lceil n/2 \rceil$ points on
 * the outer moon $(\cos t, \sin t)$ (label 0), then $\lfloor n/2 \rfloor$ on the inner moon
 * $(1 - \cos t, \tfrac12 - \sin t)$ (label 1), with $t$ evenly spaced on $[0, \pi]$, and Gaussian noise from the
 * stream's child `noise` added to each coordinate in turn. That module sits above this one, so the recipe is repeated
 * here; the test checks the two agree.
 *
 * @param s The stream.
 * @param n The number of points.
 * @param noise The noise standard deviation.
 * @returns The inputs ($n \times 2$) and the labels as a column ($n \times 1$).
 */
function twoMoons(s: Stream, n: number, noise: number): { x: Tensor; y: Tensor } {
  const nOut = Math.ceil(n / 2)
  const nIn = n - nOut
  const eps = child(s, 'noise')
  const x = new Float64Array(2 * n)
  const y = new Float64Array(n)
  const angle = (i: number, m: number) => (m > 1 ? (Math.PI * i) / (m - 1) : 0)
  for (let i = 0; i < n; i++) {
    const outer = i < nOut
    const t = outer ? angle(i, nOut) : angle(i - nOut, nIn)
    const px = outer ? Math.cos(t) : 1 - Math.cos(t)
    const py = outer ? Math.sin(t) : 0.5 - Math.sin(t)
    x[2 * i] = px + (noise > 0 ? noise * normal(eps) : 0)
    x[2 * i + 1] = py + (noise > 0 ? noise * normal(eps) : 0)
    y[i] = outer ? 0 : 1
  }
  return { x: fromData(x, [n, 2]), y: fromData(y, [n, 1]) }
}

/**
 * A learning-rate sweep (see the file's introduction): train the network once per learning rate and seed, and score
 * each run by its validation cross-entropy. Costs `rates.length` $\times$ `seeds` training runs of `steps` minibatch
 * steps. Results are plain arrays, so the sweep runs in a worker. Throws `DomainError` for an unknown task or
 * optimiser, or a non-positive count.
 *
 * @param options The task, network, optimiser, grid and budget (see `LearningRateSweepOptions`).
 * @returns The rates, the validation losses (NaN when diverged) and the divergence flags (see `LearningRateSweep`).
 *
 * @example Too small, about right and too large, one seed each
 * const r = learningRateSweep({ rates: [0.001, 0.03, 3], seeds: 1, steps: 30, widths: [8] })
 * print('rates:', r.rates)
 * print('validation loss:', r.losses.map((row) => row[0]))
 */
export function learningRateSweep(options: LearningRateSweepOptions = {}): LearningRateSweep {
  const {
    task = 'moons',
    widths = [16, 16],
    optimiser = 'adam',
    momentum = 0.9,
    rates = RATES,
    seeds = 3,
    batch = 32,
    steps = 200,
    train = 200,
    validation = 1000,
    noise = 0.25,
    seed = 'lr-sweep',
  } = options
  const where = 'learningRateSweep'
  if (task !== 'moons') throw new DomainError(where, `${where}: unknown task '${task}'`)
  if (optimiser !== 'adam' && optimiser !== 'sgd')
    throw new DomainError(where, `${where}: unknown optimiser '${optimiser}'`)
  for (const [name, v] of Object.entries({ seeds, batch, steps, train, validation }))
    if (!(Number.isInteger(v) && v >= 1))
      throw new DomainError(where, `${where}: ${name} must be a positive integer, got ${v}`)
  if (rates.some((r) => !(r > 0))) throw new DomainError(where, `${where}: learning rates must be positive`)

  const data = twoMoons(stream(`${seed}/train`), train, noise)
  const valid = twoMoons(stream(`${seed}/validation`), validation, noise)
  const net = Mlp([2, ...widths, 1], { activation: 'relu', init: heNormal() })
  const losses: number[][] = []
  const diverged: boolean[][] = []
  for (const rate of rates) {
    const row: number[] = []
    const flags: boolean[] = []
    for (let k = 1; k <= seeds; k++) {
      const rule = optimiser === 'adam' ? adamRule({ stepSize: rate }) : sgdRule({ stepSize: rate, momentum })
      const algorithm = trainingLoop({
        loss: (w: Params[], b: { x: Tensor; y: Tensor }) => binaryCrossEntropyWithLogits(net.apply(w, b.x), b.y),
        data,
        batchSize: batch,
        optimizer: rule,
      })
      const final = run(algorithm, { params: net.init(stream(`${seed}/init/${k}`)) }, steps, {
        stream: stream(`${seed}/order/${k}`),
      })
      const failed = final.diverged === true
      flags.push(failed)
      row.push(failed ? NaN : Number(unwrap(binaryCrossEntropyWithLogits(net.apply(final.params, valid.x), valid.y))))
    }
    losses.push(row)
    diverged.push(flags)
  }
  return { rates: [...rates], losses, diverged }
}
