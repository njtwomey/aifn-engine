/**
 * Feature visualisation by activation maximisation: gradient ascent on the input of a differentiable unit to find an
 * image that excites it (Erhan, Bengio, Courville and Vincent, 2009, "Visualizing Higher-Layer Features of a Deep
 * Network", §3.1; Simonyan, Vedaldi and Zisserman, 2014, §2), with the regularisers of Yosinski et al. (2015),
 * "Understanding Neural Networks Through Deep Visualization", and Olah, Mordvintsev and Schubert (2017), "Feature
 * Visualization": an L2 penalty $\lambda \lVert \xvec \rVert^2$ subtracted from the objective, random jitter, and a
 * Gaussian blur every few steps. Each step is
 *
 * $$
 * \xvec \leftarrow \operatorname{clip}\big(\xvec + \eta\, \gvec / (\operatorname{rms}(\gvec) + 10^{-12})\big), \qquad
 * \gvec = \nabla f(\xvec) - 2\lambda\xvec,
 * $$
 *
 * with the step normalised by the root mean square of $\gvec$ when `normalise` is on (the default), so $\eta$ is in the
 * units of $\xvec$ whatever the unit's scale; with `normalise` off it is plain gradient ascent. Jitter takes the
 * gradient at a copy of $\xvec$ shifted circularly by up to `jitter` pixels in each direction, shifted back: right for
 * units that tolerate shifts (convolutional units), misaligned for a unit tied to absolute positions.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { integers, normals } from 'aifn-compute/foundation/random'
import { fromData, item, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { gaussianBlur } from 'aifn-compute/signal/image'

/** Options of `activationMaximisation`. */
export type ActivationMaximisationOptions = {
  /** The step size $\eta$ (default 0.05): in units of $\xvec$ when `normalise` is on. */
  stepSize?: number
  /** The L2 penalty weight $\lambda \ge 0$ (default 0). */
  l2?: number
  /** The largest circular shift, in pixels, applied before each gradient (default 0: none). */
  jitter?: number
  /** The standard deviation in pixels of the Gaussian blur applied to $\xvec$ (default 0: none). */
  blur?: number
  /** Blur every this many steps (default 4). */
  blurEvery?: number
  /** Clip $\xvec$ to this interval after each step (default: no clipping). */
  clip?: readonly [number, number]
  /** Normalise each step by the root mean square of $\gvec$ (default true). */
  normalise?: boolean
  /** Mean and standard deviation of the random starting image (default `[0.5, 0.05]`) when the start gives none. */
  init?: readonly [number, number]
}

/** The state of `activationMaximisation` after $t$ steps. */
export type ActivationMaximisationState = Status & {
  /** The image, row-major over `shape`. */
  x: Float64Array
  /** $f(\xvec)$ at this image, without the penalty. */
  value: number
  /** The penalised objective $f(\xvec) - \lambda \lVert \xvec \rVert^2$. */
  objective: number
}

/**
 * Circular shift of each $H \times W$ plane of a flat array by $(d_r, d_c)$.
 *
 * @param x The planes, row-major, `planes` $\times H \times W$ values; not modified.
 * @param planes The number of planes.
 * @param H Rows per plane.
 * @param W Columns per plane.
 * @param dr The shift down, in rows (negative is up).
 * @param dc The shift right, in columns.
 * @returns The shifted copy.
 */
function roll(x: Float64Array, planes: number, H: number, W: number, dr: number, dc: number): Float64Array {
  const out = new Float64Array(x.length)
  const mod = (a: number, n: number) => ((a % n) + n) % n
  for (let p = 0; p < planes; p++)
    for (let r = 0; r < H; r++)
      for (let c = 0; c < W; c++) out[p * H * W + mod(r + dr, H) * W + mod(c + dc, W)] = x[p * H * W + r * W + c]
  return out
}

/**
 * Activation maximisation as an algorithm: gradient ascent on the input of `f` with an L2 penalty, jitter, blur and
 * clipping (see the file's introduction). `init` takes `{ x }`, the starting image row-major over `shape`, or nothing
 * (a random image from the init stream); each step draws its jitter (2 integers) from its own stream. The recorded
 * `value` is $f(\xvec)$ at the unjittered image after the step. Throws `ShapeError` for a shape of another rank or a
 * start of the wrong size; a non-finite objective sets `diverged`.
 *
 * @param f The unit, a differentiable scalar function of the image; it receives a tensor of shape `shape`.
 * @param shape The image's shape, `[H, W]` or `[C, H, W]`.
 * @param options The step size, penalty, jitter, blur, clipping, normalisation and random start.
 * @returns The algorithm, to run with `run`, `trace` or `live`.
 *
 * @example A linear unit with an L2 penalty climbs to $\wvec / (2\lambda)$
 * const w = tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
 * const alg = activationMaximisation((x) => sum(mul(x, w)), [3, 3], { stepSize: 0.25, l2: 1, normalise: false })
 * const start = { x: new Array(9).fill(0) }
 * for (const t of [1, 2, 3, 60]) print(`step ${t}: x[0] =`, run(alg, start, t).x[0])
 * print('value and objective at the limit:', run(alg, start, 60).value, run(alg, start, 60).objective)
 */
export function activationMaximisation(
  f: (x: Tensor) => Value,
  shape: readonly number[],
  options: ActivationMaximisationOptions = {},
): Algorithm<{ x?: ArrayLike<number> } | undefined, ActivationMaximisationState> {
  const where = 'activationMaximisation'
  const {
    stepSize = 0.05,
    l2 = 0,
    jitter = 0,
    blur = 0,
    blurEvery = 4,
    clip,
    normalise = true,
    init = [0.5, 0.05],
  } = options
  if (shape.length !== 2 && shape.length !== 3)
    throw new ShapeError(where, `${where}: shape must be [H, W] or [C, H, W]`, [shape])
  if (!(l2 >= 0)) throw new DomainError(where, `${where}: the L2 weight ${l2} is negative`)
  const H = shape[shape.length - 2]
  const W = shape[shape.length - 1]
  const planes = shape.length === 3 ? shape[0] : 1
  const n = planes * H * W
  const vg = valueAndGrad((x: Tensor) => f(x))
  const valueOf = (x: Float64Array) => {
    const v = f(fromData(x, shape))
    return typeof v === 'number' ? v : item(v as Tensor)
  }
  const penalised = (x: Float64Array, v: number) => {
    let s = 0
    for (let i = 0; i < n; i++) s += x[i] * x[i]
    return v - l2 * s
  }
  const clipped = (x: Float64Array) => {
    if (clip) for (let i = 0; i < n; i++) x[i] = Math.min(clip[1], Math.max(clip[0], x[i]))
    return x
  }
  return {
    name: 'activationMaximisation',
    init(start, s) {
      const x = start?.x ? Float64Array.from(start.x) : Float64Array.from(toFlat(normals(s, n, init[0], init[1])))
      if (x.length !== n) throw new ShapeError(where, `${where}: the start has ${x.length} values, not ${n}`)
      clipped(x)
      const value = valueOf(x)
      return { t: 0, x, value, objective: penalised(x, value) }
    },
    step(state, ctx) {
      const J = Math.max(0, Math.floor(jitter))
      const dr = J > 0 ? integers(ctx.stream, 2 * J + 1) - J : 0
      const dc = J > 0 ? integers(ctx.stream, 2 * J + 1) - J : 0
      const shifted = J > 0 ? roll(state.x, planes, H, W, dr, dc) : state.x
      const { grad } = vg(fromData(shifted, shape))
      let g: Float64Array = Float64Array.from(toFlat(grad as Tensor))
      if (J > 0) g = roll(g, planes, H, W, -dr, -dc)
      for (let i = 0; i < n; i++) g[i] -= 2 * l2 * state.x[i]
      let scale = stepSize
      if (normalise) {
        let ss = 0
        for (let i = 0; i < n; i++) ss += g[i] * g[i]
        scale = stepSize / (Math.sqrt(ss / n) + 1e-12)
      }
      let x = new Float64Array(n)
      for (let i = 0; i < n; i++) x[i] = state.x[i] + scale * g[i]
      const t = state.t + 1
      if (blur > 0 && blurEvery > 0 && t % blurEvery === 0) {
        const out = new Float64Array(n)
        for (let p = 0; p < planes; p++) {
          const plane = fromData(x.slice(p * H * W, (p + 1) * H * W), [H, W])
          out.set(toFlat(gaussianBlur(plane, blur)), p * H * W)
        }
        x = out
      }
      clipped(x)
      const value = valueOf(x)
      const objective = penalised(x, value)
      return { t, x, value, objective, diverged: !Number.isFinite(objective) }
    },
  }
}
