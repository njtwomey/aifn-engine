/**
 * A latent ODE (Rubanova, Chen & Duvenaud, 2019; Chen et al., 2018, §5) on irregularly sampled trajectories, kept
 * minimal: a GRU reads the observed points backwards in time (a mask marks which grid times were observed) into
 * $q(\zvec_0) = \Gauss(\muvec, \diag \sigmavec^2)$; $\zvec_0$ is drawn by reparameterisation, a neural ODE
 * $\zvec' = f_{\thetavec}(\zvec)$ carries it over the whole time grid, and a linear decoder maps $\zvec(t)$ to the
 * observation. Training maximises the ELBO on the observed points of the first half of the grid, the observation
 * window; the second half is extrapolation.
 */

import type { OdeFlowOptions } from 'aifn-compute/dynamics/ode'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, normals, stream, units, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  exp,
  fromData,
  mul,
  permute,
  reshape,
  shapeOfValue,
  slice,
  square,
  stack,
  sub,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { GruCell, Linear, Mlp, OdeBlock, linear, type CellParams, type LinearParams } from 'aifn-compute/nn/layers'
import { xavierUniform } from 'aifn-compute/nn/init'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { flatOf, now, planeGrid, spaced, workCounter } from './shared'

/** Trajectory families: a sine of random amplitude, frequency and phase (1-d), or a 2-d spiral either way round. */
export type TrajectoryKind = 'sine' | 'spiral'

/** Trajectories on a shared time grid with a mask of observed points. */
export type TrajectorySet = {
  /** The family the trajectories were drawn from. */
  kind: TrajectoryKind
  /** Grid times ($T$ of them). */
  times: Float64Array
  /** Grid points in the observation window (the first `observedUntil` times); the rest is extrapolation. */
  observedUntil: number
  /** Noisy values $[N, T, D]$, at every grid time (the mask says which count as observed). */
  x: Tensor
  /** The noiseless values, $[N, T, D]$. */
  truth: Tensor
  /** 1 where a point of the window was observed, $[N, T]$ (0 outside the window). */
  mask: Tensor
}

/** Options of {@link trajectories}. */
export type TrajectoryOptions = {
  /** The family. Default `'sine'`. */
  kind?: TrajectoryKind
  /** Trajectories. Default 96. */
  n?: number
  /** Grid times, evenly spaced on $[0, \text{duration}]$. Default 40. */
  grid?: number
  /** The length of the time window. Default 4. */
  duration?: number
  /** Probability of observing each grid point in the window (irregular sampling). Default 0.4. */
  observed?: number
  /** Standard deviation of the Gaussian noise added to every value. Default 0.05. */
  noise?: number
}

/**
 * Irregularly sampled trajectories on $[0, \text{duration}]$, observed in the first half of the grid (deterministic in
 * the stream). With $a, b, c$ uniform on $[0, 1]$ per trajectory, a sine is $(0.5 + a)\sin((1.5 + 1.5b)t + 2\pi c)$,
 * and a spiral (Chen et al., 2018, §5.1) has radius $(0.8 + 0.6a)e^{-t/4}$ and angle $\pm(1.2 + 0.6b)t + 2\pi c$,
 * turning either way with even odds. Each grid point of the window is observed with probability `observed`, and the
 * first always is.
 *
 * @param s The stream the trajectories' parameters, the noise and the mask are drawn from.
 * @param options The family, the number of trajectories, the grid, the window's length, the observation probability
 *   and the noise; see `TrajectoryOptions`.
 * @returns The set: grid times, the window's length, the noisy values, the truth and the mask.
 *
 * @example Three sines on a grid of ten, observed at random in the first five
 * const set = trajectories(stream(0), { n: 3, grid: 10 })
 * print('values:', set.x.shape, ' window:', set.observedUntil, 'of', set.times.length, 'grid times')
 * print('mask:', set.mask)
 * print('first trajectory, truth:', toArray(set.truth)[0].map(([v]) => v))
 */
export function trajectories(s: Stream, options: TrajectoryOptions = {}): TrajectorySet {
  const { kind = 'sine', n = 96, grid = 40, duration = 4, observed = 0.4, noise = 0.05 } = options
  const d = kind === 'sine' ? 1 : 2
  const times = Float64Array.from(spaced(0, duration, grid))
  const observedUntil = Math.floor(grid / 2)
  const u = units(child(s, 'params'), n * 4)
  const eps = toFlat(normals(child(s, 'noise'), [n * grid * d]))
  const keep = units(child(s, 'mask'), n * grid)
  const truth = new Float64Array(n * grid * d)
  const mask = new Float64Array(n * grid)
  for (let i = 0; i < n; i++) {
    const [a, b, c, e] = [u[4 * i], u[4 * i + 1], u[4 * i + 2], u[4 * i + 3]]
    for (let j = 0; j < grid; j++) {
      const t = times[j]
      if (kind === 'sine') {
        truth[i * grid + j] = (0.5 + a) * Math.sin((1.5 + 1.5 * b) * t + 2 * Math.PI * c)
      } else {
        // A decaying spiral, clockwise or counter-clockwise (Chen et al., 2018, §5.1).
        const dir = e < 0.5 ? 1 : -1
        const r = (0.8 + 0.6 * a) * Math.exp(-0.25 * t)
        const phi = dir * (1.2 + 0.6 * b) * t + 2 * Math.PI * c
        truth[(i * grid + j) * 2] = r * Math.cos(phi)
        truth[(i * grid + j) * 2 + 1] = r * Math.sin(phi)
      }
    }
    let count = 0
    for (let j = 0; j < observedUntil; j++) {
      const on = keep[i * grid + j] < observed || (j === 0 && count === 0)
      mask[i * grid + j] = on ? 1 : 0
      if (on) count++
    }
  }
  const x = Float64Array.from(truth, (v, k) => v + noise * eps[k])
  return {
    kind,
    times,
    observedUntil,
    x: fromData(x, [n, grid, d]),
    truth: fromData(truth, [n, grid, d]),
    mask: fromData(mask, [n, grid]),
  }
}

/** Options of {@link latentOde}. */
export type LatentOdeOptions = {
  /** Observation dimension. */
  dim: number
  /** Latent dimension. Default 2 (drawable). */
  latent?: number
  /** Hidden width of the latent field. Default 24. */
  hidden?: number
  /** Hidden size of the GRU encoder. Default 16. */
  encoder?: number
  /** The solver and gradient method of the latent flow. Default that of `OdeBlock`. */
  solver?: OdeFlowOptions
}

/**
 * Parameters of a latent ODE: the GRU `encoder`, the linear `head` from its state to the mean and log variance of
 * $q(\zvec_0)$, the latent `field`'s MLP layers and the linear `decoder`.
 */
export type LatentOdeParams = { encoder: CellParams; head: LinearParams; field: Params[]; decoder: LinearParams }

/**
 * The encoder, the latent ODE and the decoder. The encoder is a GRU over $(\xvec_j m_j, m_j)$ ($m_j$ the mask) read
 * from the end of the window back to its start, whose state is kept where a point is unobserved; a linear head maps its
 * last state to the mean and log variance of $q(\zvec_0)$. The latent field is an autonomous tanh MLP with two hidden
 * layers in an `OdeBlock`, and the decoder is linear. Differentiable throughout.
 *
 * @param options The observation and latent dimensions, the widths and the solver; see `LatentOdeOptions`.
 * @returns The model: `latent` (its dimension $L$), the `block`, `init`, `encode(params, x, mask, window)` (the mean
 *   and log variance of $q(\zvec_0)$, $[B, L]$ each, from values $[B, T, D]$ and mask $[B, T]$ over the first `window`
 *   grid times) and `decode(params, z0, times)` (the decoded means $[B, T, D]$ and the latent states $[T, B, L]$).
 *
 * @example Encode irregular observations, then decode over the whole grid
 * const set = trajectories(stream(0), { n: 4, grid: 10 })
 * const model = latentOde({ dim: 1 })
 * const p = model.init(stream(1))
 * const q = model.encode(p, set.x, set.mask, set.observedUntil)
 * print('q(z0) means:', q.mean)
 * const { x, z } = model.decode(p, q.mean, Array.from(set.times))
 * print('decoded means:', x.shape, ' latent path:', z.shape)
 */
export function latentOde(options: LatentOdeOptions) {
  const { dim, latent = 2, hidden = 24, encoder = 16, solver = {} } = options
  const init = xavierUniform()
  const cell = GruCell(dim + 1, encoder)
  const head = Linear(encoder, 2 * latent, { init })
  const block = OdeBlock(Mlp([latent, hidden, hidden, latent], { activation: 'tanh', init }), solver)
  const decoder = Linear(latent, dim, { init })
  /** $q(\zvec_0)$ from observations $[B, T, D]$ and mask $[B, T]$, reading the window's grid backwards. */
  const encode = (p: LatentOdeParams, x: Value, mask: Value, window: number) => {
    const b = shapeOfValue(x)[0]
    let h: Value = cell.initialState([b]).h
    for (let j = window - 1; j >= 0; j--) {
      const xj = reshape(slice(x, null, j), [b, dim])
      const mj = reshape(slice(mask, null, [j, j + 1]), [b, 1])
      const next = cell.step(p.encoder, concat([mul(xj, mj), mj], 1), { h }).h
      // Unobserved grid points leave the state unchanged.
      h = add(mul(mj, next), mul(sub(1, mj), h))
    }
    const out = linear(h, p.head.weight, p.head.bias)
    return { mean: slice(out, null, [0, latent]), logVariance: slice(out, null, [latent, 2 * latent]) }
  }
  /** Decoded means $[B, T, D]$ at `times` from $\zvec_0$ $[B, L]$, and the latent states $[T, B, L]$. */
  const decode = (p: LatentOdeParams, z0: Value, times: readonly number[], override?: OdeFlowOptions) => {
    const zs = block.flow(p.field, z0, times, override)
    const z = stack(zs) // [T, B, L]
    const xs = linear(z, p.decoder.weight, p.decoder.bias) // [T, B, D]
    return { x: permute(xs, [1, 0, 2]), z }
  }
  return {
    latent,
    block,
    init: (s: Stream): LatentOdeParams => ({
      encoder: cell.init(child(s, 'encoder')),
      head: head.init(child(s, 'head')),
      field: block.init(child(s, 'field')),
      decoder: decoder.init(child(s, 'decoder')),
    }),
    encode,
    decode,
  }
}

/** Options of {@link latentOdeRun}; plain data. */
export type LatentOdeRunOptions = TrajectoryOptions &
  Omit<LatentOdeOptions, 'dim'> & {
    /** Optimiser steps. Default 300. */
    steps?: number
    /** Minibatch size (at most $N$). Default 32. */
    batchSize?: number
    /** Adam's step size. Default 0.01. */
    learningRate?: number
    /** Rescale gradients whose global norm exceeds this (keeps late Adam steps from spiking). Default 1. */
    clipNorm?: number
    /** Observation noise $\sigma$ of the Gaussian likelihood. Default 0.1. */
    sigma?: number
    /** Seed of the data, the initialisation, the minibatches and the draws of $\zvec_0$. Default 0. */
    seed?: number
    /** Trajectories drawn. Default 4. */
    shown?: number
    /** Checkpoints over the run (from step 0). Default 12. */
    checkpoints?: number
  }

/** A checkpoint of a latent ODE run. */
export type LatentOdeCheckpoint = {
  /** The optimiser step it was taken at. */
  step: number
  /** Mean squared error against the truth over every trajectory and grid time of the window (interpolation). */
  interpolation: number
  /** The same after the window (extrapolation). */
  extrapolation: number
  /** Decoded means of the shown trajectories at $\zvec_0 = \muvec$: $[\text{shown} \times T \times D]$. */
  predictions: Float64Array
  /** Their latent paths: $[T \times \text{shown} \times L]$. */
  latents: Float64Array
  /** The latent field on an $11 \times 11$ quiver grid over $[-2.5, 2.5]^2$ ($L = 2$), $[g^2 \times 2]$, or null. */
  field: Float64Array | null
}

/** A snapshot of a latent ODE run. */
export type LatentOdeRun = {
  /** The trajectory family. */
  kind: TrajectoryKind
  /** The steps the run was asked for. */
  steps: number
  /** The steps taken so far. */
  done: number
  /** True on the last snapshot. */
  finished: boolean
  /** The message of the error that ended the run early, or null. */
  error: string | null
  /** The observation dimension $D$. */
  dim: number
  /** The latent dimension $L$. */
  latent: number
  /** The grid times. */
  times: Float64Array
  /** Grid points in the observation window. */
  observedUntil: number
  /** The shown trajectories' noisy values, $[\text{shown} \times T \times D]$. */
  observations: Float64Array
  /** Their truth, $[\text{shown} \times T \times D]$. */
  truth: Float64Array
  /** Their masks, $[\text{shown} \times T]$. */
  mask: Float64Array
  /** The quiver grid's axis ($L = 2$), else empty. */
  fieldAxis: Float64Array
  /** Per iteration: the minibatch loss, the negative ELBO per observed point. */
  loss: Float64Array
  /** Per iteration: function evaluations of the forward solves. */
  nfeForward: Float64Array
  /** Per iteration: function evaluations of the backward pass (backprop's replay the forward). */
  nfeBackward: Float64Array
  /** Per iteration: wall milliseconds. */
  wallMs: Float64Array
  /** The checkpoints so far. */
  checkpoints: LatentOdeCheckpoint[]
}

/**
 * Train a latent ODE on irregular trajectories and yield snapshots (about every twentieth of the run and at the end).
 * The loss of a minibatch is its negative ELBO per observed point, constants dropped: $(R + K) / n_{\text{obs}}$, with
 * $R = \frac{1}{2\sigma^2} \sum_j m_j \norm{\hat\xvec_j - \xvec_j}^2$ over the window's grid times (mask $m_j$) and
 * $K = \KL(q(\zvec_0) \,\|\, \Gauss(\zeros, \Imat))$, both summed over the minibatch, and one reparameterised draw
 * of $\zvec_0$ per trajectory. Adam minimises it, with the gradient's global norm clipped. Checkpoints decode every
 * trajectory from its mean $\zvec_0 = \muvec$. An error during training ends the run with a finished snapshot
 * carrying its message. Deterministic in `seed` (wall times aside).
 *
 * @param options The data (`TrajectoryOptions`), the model (`LatentOdeOptions` without `dim`, which the data sets), the
 *   likelihood's $\sigma$, the optimisation and what checkpoints record; see `LatentOdeRunOptions`.
 * @returns A generator of `LatentOdeRun` snapshots; the last has `finished` set.
 *
 * @example The loss falls, and the fit holds better inside the window than beyond it
 * const options = { n: 16, grid: 10, steps: 20, batchSize: 16, solver: { stepSize: 0.5 }, checkpoints: 1, shown: 1 }
 * let run
 * for (const r of latentOdeRun(options)) run = r
 * print('loss at steps 0, 10, 20:', [0, 10, 20].map((k) => run.loss[k]))
 * const last = run.checkpoints.at(-1)
 * print('mean squared error, interpolation:', last.interpolation, ' extrapolation:', last.extrapolation)
 */
export function* latentOdeRun(options: LatentOdeRunOptions = {}): Generator<LatentOdeRun, LatentOdeRun> {
  const {
    kind = 'sine',
    steps = 300,
    batchSize = 32,
    learningRate = 0.01,
    clipNorm = 1,
    sigma = 0.1,
    seed = 0,
    shown: shownCount = 4,
    checkpoints = 12,
  } = options
  const solver: OdeFlowOptions = { method: 'rk4', stepSize: 0.1, gradient: 'backprop', ...options.solver }
  const gradientMethod = solver.gradient ?? 'backprop'
  const counter = workCounter()
  const root = stream(seed)
  const set = trajectories(child(root, 'data'), options)
  const [n, T, D] = set.x.shape
  const model = latentOde({ ...options, dim: D, solver: { ...solver, onSolve: counter.onSolve } })
  const L = model.latent
  const times = Array.from(set.times)
  const window = set.observedUntil
  const weight = 1 / (2 * sigma * sigma)

  const loss = (p: LatentOdeParams, batch: { x: Tensor; mask: Tensor }, ctx: { stream?: Stream }): Value => {
    const b = batch.x.shape[0]
    const q = model.encode(p, batch.x, batch.mask, window)
    const eps = normals(child(ctx.stream!, 'z0'), [b, L])
    const z0 = add(q.mean, mul(exp(mul(0.5, q.logVariance)), eps))
    const { x } = model.decode(p, z0, times.slice(0, window))
    const observedX = slice(batch.x, null, [0, window])
    const m = reshape(slice(batch.mask, null, [0, window]), [b, window, 1])
    const count = Math.max(
      1,
      toFlat(batch.mask).reduce((a, v) => a + v, 0),
    )
    const recon = mul(weight / count, sum(mul(m, square(sub(x, observedX)))))
    const kl = mul(0.5 / count, sum(sub(add(square(q.mean), exp(q.logVariance)), add(1, q.logVariance))))
    return add(recon, kl)
  }
  const alg = trainingLoop({
    loss: loss as never,
    data: { x: set.x, mask: set.mask },
    batchSize: Math.min(batchSize, n),
    optimizer: adamRule({ stepSize: learningRate }) as UpdateRule<unknown>,
    clipNorm,
  })

  const shownIdx = Array.from({ length: Math.min(shownCount, n) }, (_, i) => i)
  const pick = (t: Tensor, cols: number) => Float64Array.from(toFlat(t).slice(0, shownIdx.length * cols))
  const observations = pick(set.x, T * D)
  const truth = pick(set.truth, T * D)
  const mask = pick(set.mask, T)
  const fieldGrid = 11
  const quiver = L === 2 ? planeGrid(2.5, fieldGrid) : null
  const allTruth = toFlat(set.truth)

  const checkpoint = (step: number, p: LatentOdeParams): LatentOdeCheckpoint => {
    counter.take('none')
    const q = model.encode(p, set.x, set.mask, window)
    const { x, z } = model.decode(p, q.mean, times)
    const pred = flatOf(x)
    let si = 0
    let se = 0
    for (let i = 0; i < n; i++)
      for (let j = 0; j < T; j++)
        for (let k = 0; k < D; k++) {
          const e = (pred[(i * T + j) * D + k] - allTruth[(i * T + j) * D + k]) ** 2
          if (j < window) si += e
          else se += e
        }
    const zs = flatOf(z) // [T, n, L]
    const latents = new Float64Array(T * shownIdx.length * L)
    for (let j = 0; j < T; j++)
      for (let i = 0; i < shownIdx.length; i++)
        for (let k = 0; k < L; k++) latents[(j * shownIdx.length + i) * L + k] = zs[(j * n + i) * L + k]
    const field = quiver ? flatOf(model.block.field(p.field, 0, quiver.points)) : null
    counter.take('none')
    return {
      step,
      interpolation: si / (n * window * D),
      extrapolation: se / (n * (T - window) * D),
      predictions: pred.slice(0, shownIdx.length * T * D),
      latents,
      field,
    }
  }

  const every = Math.max(1, Math.round(steps / checkpoints))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  const losses: number[] = []
  const nfeF: number[] = []
  const nfeB: number[] = []
  const wall: number[] = []
  const shots: LatentOdeCheckpoint[] = []
  let error: string | null = null
  const snapshot = (done: number, finished: boolean): LatentOdeRun => ({
    kind,
    steps,
    done,
    finished,
    error,
    dim: D,
    latent: L,
    times: set.times,
    observedUntil: window,
    observations,
    truth,
    mask,
    fieldAxis: quiver ? quiver.axis : new Float64Array(0),
    loss: Float64Array.from(losses),
    nfeForward: Float64Array.from(nfeF),
    nfeBackward: Float64Array.from(nfeB),
    wallMs: Float64Array.from(wall),
    checkpoints: shots.slice(),
  })
  type Item = { step: number; state: { params: unknown; loss: number }; stopped?: unknown }
  const iterator = live(alg, { params: model.init(child(root, 'init')) as never }, { stream: child(root, 'train') })
  let lastEnd = now()
  for (;;) {
    let next: IteratorResult<Item, void>
    try {
      next = iterator.next() as IteratorResult<Item, void>
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
      const last = snapshot(losses.length, true)
      yield last
      return last
    }
    if (next.done) break
    const { step, state, stopped } = next.value
    const ms = now() - lastEnd
    const work = counter.take(gradientMethod)
    losses.push(state.loss)
    nfeF.push(work.forward)
    nfeB.push(work.backward)
    wall.push(ms)
    if (step % every === 0 || step === steps || stopped) shots.push(checkpoint(step, state.params as LatentOdeParams))
    if (step === steps || stopped) {
      const last = snapshot(step, true)
      yield last
      return last
    }
    if (step > 0 && step % chunk === 0) yield snapshot(step, false)
    lastEnd = now()
  }
  return snapshot(steps, true)
}
