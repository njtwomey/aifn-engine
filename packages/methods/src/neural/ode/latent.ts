/**
 * A latent ODE (Rubanova, Chen & Duvenaud, 2019; Chen et al., 2018, §5) on irregularly sampled trajectories, kept
 * minimal: a GRU reads the observed points backwards in time (a mask marks which grid times were observed) into
 * q(z₀) = 𝒩(μ, diag σ²); z₀ is drawn by reparameterisation, a neural ODE z′ = f_θ(z) carries it over the whole time
 * grid, and a linear decoder maps z(t) to the observation. Training maximises the ELBO on the observed points of the
 * first half of the window; the second half is extrapolation.
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
  kind: TrajectoryKind
  /** Grid times (length T). */
  times: Float64Array
  /** Grid points in the observation window (the first `observedUntil` times); the rest is extrapolation. */
  observedUntil: number
  /** Noisy values [N, T, D] and the noiseless truth. */
  x: Tensor
  truth: Tensor
  /** 1 where a point of the window was observed, [N, T] (0 outside the window). */
  mask: Tensor
}

/** Options of {@link trajectories}. */
export type TrajectoryOptions = {
  kind?: TrajectoryKind
  /** Trajectories. Default 96. */
  n?: number
  /** Grid times on [0, duration]. Default 40 on [0, 4]. */
  grid?: number
  duration?: number
  /** Probability of observing each grid point in the window (irregular sampling). Default 0.4. */
  observed?: number
  noise?: number
}

/** Irregularly sampled trajectories on [0, duration], observed in the first half (deterministic in the stream). */
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
  solver?: OdeFlowOptions
}

export type LatentOdeParams = { encoder: CellParams; head: LinearParams; field: Params[]; decoder: LinearParams }

/** The encoder, the latent ODE and the decoder. */
export function latentOde(options: LatentOdeOptions) {
  const { dim, latent = 2, hidden = 24, encoder = 16, solver = {} } = options
  const init = xavierUniform()
  const cell = GruCell(dim + 1, encoder)
  const head = Linear(encoder, 2 * latent, { init })
  const block = OdeBlock(Mlp([latent, hidden, hidden, latent], { activation: 'tanh', init }), solver)
  const decoder = Linear(latent, dim, { init })
  /** q(z₀) from observations x [B, T, D] and mask [B, T], reading the window's grid backwards. */
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
  /** Decoded means [B, T, D] at `times` from z₀ [B, L], and the latent states [T, B, L]. */
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
    steps?: number
    batchSize?: number
    learningRate?: number
    /** Rescale gradients whose global norm exceeds this (keeps late Adam steps from spiking). Default 1. */
    clipNorm?: number
    /** Observation noise σ of the Gaussian likelihood. Default 0.1. */
    sigma?: number
    seed?: number
    /** Trajectories drawn. Default 4. */
    shown?: number
    checkpoints?: number
  }

/** A checkpoint of a latent ODE run. */
export type LatentOdeCheckpoint = {
  step: number
  /** Mean squared error against the truth on the window's grid (interpolation) and after it (extrapolation). */
  interpolation: number
  extrapolation: number
  /** Decoded means of the shown trajectories at z₀ = μ: [shown × T × D]. */
  predictions: Float64Array
  /** Their latent paths: [T × shown × L]. */
  latents: Float64Array
  /** The latent field on the quiver grid (L = 2), [g² × 2], or null. */
  field: Float64Array | null
}

/** A snapshot of a latent ODE run. */
export type LatentOdeRun = {
  kind: TrajectoryKind
  steps: number
  done: number
  finished: boolean
  error: string | null
  dim: number
  latent: number
  times: Float64Array
  observedUntil: number
  /** The shown trajectories: noisy values, truth [shown × T × D] and masks [shown × T]. */
  observations: Float64Array
  truth: Float64Array
  mask: Float64Array
  fieldAxis: Float64Array
  loss: Float64Array
  nfeForward: Float64Array
  nfeBackward: Float64Array
  wallMs: Float64Array
  checkpoints: LatentOdeCheckpoint[]
}

/** Train a latent ODE on irregular trajectories and yield snapshots. */
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
