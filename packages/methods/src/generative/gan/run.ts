/**
 * A GAN training run as a generator of plain-data snapshots, for a worker to stream (as `aifn-methods/gym`'s
 * `training`): the losses at every step, and at checkpoints the generated points from fixed latents, the
 * discriminator on a grid, the generator's gradient at a subset of the generated points, and, when the data's density
 * is known, the optimal discriminator and the mode coverage.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, mul, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { generatorLoss, type AdversarialGame } from 'aifn-compute/learning/losses'
import { adamRule, rmspropRule, sgdRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { knownDensity, mixtureLogDensityOf, squareGrid } from '../densities'
import { modeCoverage, optimalDiscriminator, type ModeCoverage } from './diagnostics'
import { discriminate, gan, ganTraining, latents, sampleGenerator, scoresAt } from './gan'

/** An optimiser by name, with its step size and (Adam) β₁ or (SGD) momentum. */
export type OptimizerSpec = {
  name: 'adam' | 'sgd' | 'rmsprop'
  stepSize: number
  /** Adam's β₁ (default 0.5) or SGD's momentum (default 0). */
  beta1?: number
}

/** The update rule an `OptimizerSpec` names. */
export function optimizerOf(spec: OptimizerSpec): UpdateRule<unknown> {
  switch (spec.name) {
    case 'adam':
      return adamRule({ stepSize: spec.stepSize, beta1: spec.beta1 ?? 0.5 }) as UpdateRule<unknown>
    case 'sgd':
      return sgdRule({ stepSize: spec.stepSize, momentum: spec.beta1 ?? 0 }) as UpdateRule<unknown>
    case 'rmsprop':
      return rmspropRule({ stepSize: spec.stepSize }) as UpdateRule<unknown>
  }
}

/** A dataset as the run reads it: points, mode labels, and the known density in `meta.truth.model` when there is one. */
export type GanData = {
  x: Tensor
  y?: Tensor
  /** A truth with a labelled density (`knownDensity`) gives D* and mode coverage. */
  meta?: { truth?: unknown }
}

/** Options of `ganRun`; plain data, so a worker task can carry them. */
export type GanRunOptions = {
  game?: AdversarialGame
  /** Generator updates. Default 2000. */
  steps?: number
  criticSteps?: number
  batchSize?: number
  hidden?: readonly number[]
  latent?: number
  generator?: OptimizerSpec
  critic?: OptimizerSpec
  /** Gradient-penalty weight (default 10 for `wasserstein`, else 0). */
  penalty?: number
  seed?: number | string
  /** Generated points shown per checkpoint (fixed latents). Default 512. */
  samples?: number
  /** Points carrying gradient arrows (the first of the shown points). Default 64. */
  arrows?: number
  /** Grid cells per side. Default 48. */
  grid?: number
  /** Checkpoints besides step 0. Default 40. */
  checkpoints?: number
}

/** One checkpoint of a run. */
export type GanCheckpoint = {
  step: number
  /** Generated points from the run's fixed latents, row-major [samples × 2]. */
  samples: Float64Array
  /** The discriminator on the grid: D(x) = σ(logit) for the minimax and non-saturating games, else the critic score. */
  field: Float64Array
  /** D*(x) on the grid when the data density is known and the game has a probability discriminator; else null. */
  optimal: Float64Array | null
  /** −∇ₓ of the generator's loss at the first `arrows` points, row-major [arrows × 2]: where G is pushed to move them. */
  push: Float64Array
  coverage: ModeCoverage | null
}

/** A run so far. */
export type GanRun = {
  game: AdversarialGame
  steps: number
  done: number
  finished: boolean
  /** The grid's half-width and axes. */
  box: number
  gridX: Float64Array
  gridY: Float64Array
  /** log p_data on the grid, or null when unknown. */
  dataLogDensity: Float64Array | null
  /** Real points [N × 2] and their mode labels. */
  data: Float64Array
  labels: Int32Array
  /** Mode centres (the mean of each mode's real points), [k × 2]; empty without labels. */
  modes: Float64Array
  /** Losses per generator update. */
  criticLoss: Float64Array
  generatorLoss: Float64Array
  checkpoints: GanCheckpoint[]
}

const probabilityGame = (g: AdversarialGame) => g === 'minimax' || g === 'non-saturating'

/**
 * Train a GAN on `data` and yield snapshots: after about every twentieth of the run and at the end. Checkpoints are
 * evenly spaced, from step 0. Deterministic in `seed`.
 */
export function* ganRun(data: GanData, options: GanRunOptions = {}): Generator<GanRun, GanRun> {
  const {
    game = 'non-saturating',
    steps = 2000,
    criticSteps = game === 'wasserstein' ? 5 : 1,
    batchSize = 128,
    hidden = [64, 64],
    latent = 2,
    seed = 0,
    samples = 512,
    arrows = 64,
    grid = 48,
    checkpoints = 40,
  } = options
  const genSpec = options.generator ?? { name: 'adam', stepSize: game === 'wasserstein' ? 1e-4 * 5 : 1e-3 }
  const criticSpec = options.critic ?? genSpec
  const net = gan({ latent, hidden })
  const x = data.x
  const N = x.shape[0]
  const xs = Float64Array.from(toFlat(x))
  const labels = data.y ? Int32Array.from(toFlat(data.y)) : new Int32Array(N)
  const model = knownDensity(data.meta?.truth)
  let extent = 0
  for (const v of xs) extent = Math.max(extent, Math.abs(v))
  const box = Math.ceil(extent * 1.3 * 2) / 2
  const g = squareGrid(box, grid)
  const dataLogDensity = model ? mixtureLogDensityOf(model, g.points) : null
  const k = model ? model.classes : labels.reduce((m, v) => Math.max(m, v + 1), 0)
  const modes = new Float64Array(2 * k)
  const counts = new Float64Array(k)
  for (let i = 0; i < N; i++) {
    const j = labels[i]
    if (j < 0 || j >= k) continue
    modes[2 * j] += xs[2 * i]
    modes[2 * j + 1] += xs[2 * i + 1]
    counts[j]++
  }
  for (let j = 0; j < k; j++) {
    modes[2 * j] /= Math.max(1, counts[j])
    modes[2 * j + 1] /= Math.max(1, counts[j])
  }

  const root = stream(seed)
  const z = latents(net, child(root, 'shown'), samples)
  const alg = ganTraining({
    net,
    data: x,
    game,
    batchSize,
    criticSteps,
    penalty: options.penalty,
    generatorOptimizer: optimizerOf(genSpec),
    criticOptimizer: optimizerOf(criticSpec),
  })
  const start = {
    generator: net.generator.init(child(root, 'generator')),
    critic: net.discriminator.init(child(root, 'critic')),
  }
  // −∇ₓ ℓ_G(D(x)) summed over points, so each row is its own point's push.
  const push = (critic: Params[], pts: Tensor): Float64Array => {
    const n = pts.shape[0]
    const { grad } = valueAndGrad((p: Value) => mul(n, generatorLoss(discriminate(net, critic, p), game)))(pts)
    return Float64Array.from(toFlat(unwrap(grad as Value) as Tensor), (v) => -v)
  }
  const checkpoint = (step: number, generator: Params[], critic: Params[]): GanCheckpoint => {
    const shown = sampleGenerator(net, generator, z)
    const raw = scoresAt(net, critic, g.points)
    const field = probabilityGame(game) ? Float64Array.from(raw, (v) => 1 / (1 + Math.exp(-v))) : raw
    const sampleData = Float64Array.from(toFlat(shown))
    const first = fromData(sampleData.slice(0, 2 * Math.min(arrows, samples)), [Math.min(arrows, samples), 2])
    return {
      step,
      samples: sampleData,
      field,
      optimal:
        model && dataLogDensity && probabilityGame(game)
          ? optimalDiscriminator(dataLogDensity, shown, g.points).value
          : null,
      push: push(critic, first),
      coverage: model ? modeCoverage(model, shown, { real: x }) : null,
    }
  }
  const every = Math.max(1, Math.round(steps / checkpoints))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  const criticLoss: number[] = []
  const generatorLosses: number[] = []
  const shots: GanCheckpoint[] = []
  const snapshot = (done: number, finished: boolean): GanRun => ({
    game,
    steps,
    done,
    finished,
    box,
    gridX: g.x,
    gridY: g.y,
    dataLogDensity,
    data: xs,
    labels,
    modes,
    criticLoss: Float64Array.from(criticLoss),
    generatorLoss: Float64Array.from(generatorLosses),
    checkpoints: shots.slice(),
  })
  for (const { step, state, stopped } of live(alg, start, { stream: child(root, 'train') })) {
    if (step > 0) {
      criticLoss.push(state.criticLoss)
      generatorLosses.push(state.generatorLoss)
    }
    if (step % every === 0 || step === steps || stopped) shots.push(checkpoint(step, state.generator, state.critic))
    // The last snapshot is yielded too: a worker streams yielded values and keeps the last.
    if (step === steps || stopped) {
      const last = snapshot(step, true)
      yield last
      return last
    }
    if (step > 0 && step % chunk === 0) yield snapshot(step, false)
  }
  return snapshot(steps, true)
}
