/**
 * Samplers: from noise to data by reversing the forward process with a noise predictor (see `predictor.ts`). Each is
 * a traceable `Algorithm` whose state holds every particle, so a trace of `x` gives the sampling paths.
 *
 * - `ddpmSampler`: ancestral sampling of DDPM (Ho et al., 2020, Algorithm 2), one step per noise level.
 * - `ddimSampler`: DDIM (Song, Meng & Ermon, 2021, eq. 12) on a subsequence of levels; $\eta = 0$ is deterministic.
 * - `reverseSdeSampler`: Euler–Maruyama on the reverse-time SDE
 *   $d\xvec = [f\xvec - g^2\nabla \log p_t] \, dt + g \, d\bar\wvec$ (Anderson, 1982; Song et al., 2021, eq. 6).
 * - `probabilityFlowSampler`: the probability-flow ODE $d\xvec/dt = f\xvec - \frac{1}{2}g^2\nabla \log p_t$ (Song et
 *   al., 2021, eq. 13), integrated with `aifn-compute/dynamics/ode`'s classical fourth-order Runge–Kutta (or Euler),
 *   deterministic after the initial draw.
 *
 * The prior draw comes from the `init` stream and step $k$'s noise from the runner's step stream, so states are plain
 * data and every step is a pure function of its state. Every sampler reports the time on the shared axis
 * $\tau \in [0, 1]$ as `tau`, and the noise prediction and Tweedie's clean point at the point it stepped from.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { normals, type Stream, child } from 'aifn-compute/foundation/random'
import { add, mul, reshape, sub, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { rungeKutta, type Rhs } from 'aifn-compute/dynamics/ode'
import { predictClean, predictNoise, type NoisePredictor } from './predictor'
import { alphaBarAt, betaAt, stepTime, type ForwardSde, type NoiseSchedule } from './schedules'

/** Where sampling starts: given points `x` ($[n, d]$), or `n` fresh draws from the prior in `dimension` dimensions. */
export type SamplerStart = { x: Tensor } | { n: number; dimension: number }

/** The state of every sampler. */
export type SamplerState = Status & {
  /** Steps taken. */
  t: number
  /** The current noise level: a discrete step $t$ (DDPM, DDIM; 0 is clean) or a continuous time (SDE, ODE). */
  time: number
  /**
   * The same level on the shared continuous axis $\tau \in [0, 1]$: $t/T$ for the discrete samplers (`stepTime`), the
   * time itself for the continuous ones. A continuous sampler on `scheduleSde(schedule)` and a discrete one on
   * `schedule` agree on $\tau$.
   */
  tau: number
  /** The particles, shape $[n, d]$. */
  x: Tensor
  /** The predicted noise $\hat\epsilonvec$ at the previous point (null at the start). */
  noise: Tensor | null
  /** The clean point $\hat\xvec_0$ that `noise` implies by Tweedie's formula (null at the start). */
  clean: Tensor | null
  /** Noise-predictor evaluations so far. */
  evaluations: number
}

/**
 * The state every sampler starts from: the given points, or draws from $\Gauss(\zeros, \sigma^2\Imat)$, at a level.
 *
 * @param opts The start: points, or a count and a dimension.
 * @param s The `init` stream; draws come from its child `'prior'`.
 * @param priorStd The prior's standard deviation $\sigma$ (unused for given points).
 * @param time The starting level, in the sampler's own units (a step, or a time).
 * @param tau The same level on the shared axis $\tau \in [0, 1]$.
 * @returns The state before any step.
 */
function start(opts: SamplerStart, s: Stream, priorStd: number, time: number, tau: number): SamplerState {
  const x = 'x' in opts ? opts.x : mul(normals(child(s, 'prior'), [opts.n, opts.dimension]), priorStd)
  return { t: 0, time, tau, x, noise: null, clean: null, evaluations: 0 }
}

/** Options of `ddpmSampler`. */
export type DdpmOptions = {
  /**
   * The reverse-step variance $\sigma_t^2$: `beta` ($\beta_t$, the default) or `posterior` ($\tilde\beta_t$), the two
   * choices Ho et al. found to give similar samples (§3.2).
   */
  variance?: 'beta' | 'posterior'
}

/**
 * DDPM ancestral sampling (Ho et al., 2020, Algorithm 2): from $\xvec_T \sim \Gauss(\zeros, \Imat)$, for
 * $t = T, \dots, 1$,
 * $\xvec_{t-1} = (\xvec_t - c_t \hat\epsilonvec(\xvec_t, \bar\alpha_t)) / \sqrt{\alpha_t} + \sigma_t\zvec$ with
 * $c_t = \beta_t / \sqrt{1 - \bar\alpha_t}$ and $\zvec \sim \Gauss(\zeros, \Imat)$, with no noise on the last step.
 * $T$ steps, one predictor call each.
 *
 * @param predictor The noise predictor $\hat\epsilonvec$.
 * @param schedule The noise schedule, whose every step is visited.
 * @param options The reverse-step variance.
 * @param options.variance `'beta'` for $\sigma_t^2 = \beta_t$, `'posterior'` for $\sigma_t^2 = \tilde\beta_t$.
 * @returns The algorithm, started from a `SamplerStart` and done at step 0.
 *
 * @example One reverse step from $t = 100$ for two given points
 * const mixture = gaussianMixtureData([1, 1], [[-2], [2]], [0.5, 0.5])
 * const sampler = ddpmSampler(mixtureNoisePredictor(mixture), linearSchedule(100))
 * const one = run(sampler, { x: tensor([[0.5], [-1]]) }, 1)
 * print('time', one.time, ' tau', one.tau, ' x', one.x)
 * print('predicted noise', one.noise, ' clean point', one.clean)
 *
 * @example A full run with the exact predictor recovers the data's moments
 * const mixture = gaussianMixtureData([1, 1], [[-2], [2]], [0.5, 0.5])
 * const sampler = ddpmSampler(mixtureNoisePredictor(mixture), linearSchedule(200))
 * const end = run(sampler, { n: 1000, dimension: 1 }, 200)
 * print('steps', end.t, ' time', end.time)
 * const exact = mixtureMoments(mixture)
 * print('sample mean', mean(end.x), ' variance', variance(end.x))
 * print('exact mean', exact.mean, ' variance', exact.covariance)
 */
export function ddpmSampler(
  predictor: NoisePredictor,
  schedule: NoiseSchedule,
  { variance = 'beta' }: DdpmOptions = {},
): Algorithm<SamplerStart, SamplerState> {
  return {
    name: 'ddpm',
    init: (opts, s) => start(opts, s, 1, schedule.steps, 1),
    step: (st, ctx) => {
      const t = st.time
      const beta = betaAt(schedule, t)
      const ab = alphaBarAt(schedule, t)
      const noise = predictor(st.x, ab)
      let x = mul(sub(st.x, mul(noise, beta / Math.sqrt(1 - ab))), 1 / Math.sqrt(1 - beta))
      if (t > 1) {
        const sigma2 = variance === 'beta' ? beta : ((1 - alphaBarAt(schedule, t - 1)) / (1 - ab)) * beta
        x = add(x, mul(normals(ctx.stream, st.x.shape), Math.sqrt(sigma2)))
      }
      return {
        ...st,
        t: st.t + 1,
        time: t - 1,
        tau: stepTime(schedule, t - 1),
        x,
        noise,
        clean: predictClean(noise, st.x, Math.sqrt(ab), Math.sqrt(1 - ab)),
        evaluations: st.evaluations + 1,
      }
    },
    done: (st) => st.time <= 0,
  }
}

/** Options of `ddimSampler`. */
export type DdimOptions = {
  /** Number of sampling steps $S \le T$; the levels used are $S$ evenly spaced steps from $T$ down to 1. Default 50. */
  steps?: number
  /** $\eta \in [0, 1]$: 0 is the deterministic DDIM, 1 matches DDPM's posterior variance. Default 0. */
  eta?: number
}

/**
 * $S$ evenly spaced steps from $T$ down to 1 (distinct integers: rounding may merge some, and $S$ is clamped to
 * $1, \dots, T$), then 0. The levels `ddimSampler` visits.
 *
 * @param T The schedule's number of steps $T$.
 * @param S The number of sampling steps wanted.
 * @returns The steps in descending order, ending with 0.
 *
 * @example Five levels of 1000, and more steps than levels
 * print(ddimTimesteps(1000, 5))
 * print(ddimTimesteps(10, 20))
 */
export function ddimTimesteps(T: number, S: number): number[] {
  const n = Math.max(1, Math.min(S, T))
  const out = new Set<number>()
  for (let i = n - 1; i >= 0; i--) out.add(n === 1 ? T : Math.round(1 + (i * (T - 1)) / (n - 1)))
  return [...out, 0]
}

/**
 * DDIM (Song, Meng & Ermon, 2021, eq. 12) from level $t$ to the next level $t' < t$ of the subsequence:
 * $\hat\xvec_0 = (\xvec_t - \sqrt{1 - \bar\alpha_t}\,\hat\epsilonvec) / \sqrt{\bar\alpha_t}$ and
 * $\xvec_{t'} = \sqrt{\bar\alpha_{t'}}\hat\xvec_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\hat\epsilonvec + \sigma\zvec$
 * with $\sigma = \eta \sqrt{(1 - \bar\alpha_{t'})/(1 - \bar\alpha_t)} \sqrt{1 - \bar\alpha_t / \bar\alpha_{t'}}$. With
 * $\eta = 0$ the map from $\xvec_T$ to $\xvec_0$ is deterministic. One predictor call per step.
 *
 * @param predictor The noise predictor $\hat\epsilonvec$.
 * @param schedule The noise schedule, whose levels are subsampled by `ddimTimesteps`.
 * @param options The number of steps and the stochasticity.
 * @param options.steps The number $S$ of sampling steps.
 * @param options.eta The stochasticity $\eta$.
 * @returns The algorithm, started from a `SamplerStart` and done at step 0.
 *
 * @example Twenty deterministic steps of a 1000-step schedule; mirrored starts give mirrored samples
 * const mixture = gaussianMixtureData([1, 1], [[-2], [2]], [0.5, 0.5])
 * const sampler = ddimSampler(mixtureNoisePredictor(mixture), linearSchedule(1000), { steps: 20 })
 * const end = run(sampler, { n: 1000, dimension: 1 }, 20)
 * print('evaluations', end.evaluations, ' mean', mean(end.x), ' variance', variance(end.x))
 * print('from +0.3 and -0.3:', run(sampler, { x: tensor([[0.3], [-0.3]]) }, 20).x)
 */
export function ddimSampler(
  predictor: NoisePredictor,
  schedule: NoiseSchedule,
  { steps = 50, eta = 0 }: DdimOptions = {},
): Algorithm<SamplerStart, SamplerState> {
  const levels = ddimTimesteps(schedule.steps, steps)
  return {
    name: 'ddim',
    init: (opts, s) => start(opts, s, 1, levels[0], stepTime(schedule, levels[0])),
    step: (st, ctx) => {
      const t = st.time
      const next = levels[st.t + 1]
      const ab = alphaBarAt(schedule, t)
      const abNext = alphaBarAt(schedule, next)
      const noise = predictor(st.x, ab)
      const clean = predictClean(noise, st.x, Math.sqrt(ab), Math.sqrt(1 - ab))
      const sigma = eta * Math.sqrt((1 - abNext) / (1 - ab)) * Math.sqrt(1 - ab / abNext)
      let x = add(mul(clean, Math.sqrt(abNext)), mul(noise, Math.sqrt(Math.max(0, 1 - abNext - sigma * sigma))))
      if (sigma > 0) x = add(x, mul(normals(ctx.stream, st.x.shape), sigma))
      return {
        ...st,
        t: st.t + 1,
        time: next,
        tau: stepTime(schedule, next),
        x,
        noise,
        clean,
        evaluations: st.evaluations + 1,
      }
    },
    done: (st) => st.time <= 0,
  }
}

/** Options of the continuous-time samplers. */
export type ContinuousOptions = {
  /** Number of steps $N$ from $t = 1$ to `end`. Default 500 for the SDE, 100 for the ODE. */
  steps?: number
  /** The final time $\varepsilon > 0$ (the score is singular at $t = 0$). Default 1e-3. */
  end?: number
}

/**
 * The reverse-time SDE by Euler–Maruyama (Song et al., 2021, eq. 6 and appendix D): with $h = (1 - \varepsilon)/N$
 * and time falling from 1,
 * $\xvec \leftarrow \xvec - [f(t)\xvec - g(t)^2\nabla \log p_t(\xvec)]h + g(t)\sqrt{h}\,\zvec$, the score from the
 * predictor through `predictNoise`. Starts from the SDE's prior; $N$ steps, one predictor call each.
 *
 * @param predictor The VP noise predictor $\hat\epsilonvec$.
 * @param sde The forward SDE.
 * @param options The number of steps and the final time.
 * @param options.steps The number of steps $N$.
 * @param options.end The final time $\varepsilon$.
 * @returns The algorithm, started from a `SamplerStart` and done after $N$ steps.
 *
 * @example Two hundred steps on the VP SDE recover the data's moments
 * const mixture = gaussianMixtureData([1, 1], [[-2], [2]], [0.5, 0.5])
 * const sampler = reverseSdeSampler(mixtureNoisePredictor(mixture), vpSde(), { steps: 200 })
 * const end = run(sampler, { n: 1000, dimension: 1 }, 200)
 * const exact = mixtureMoments(mixture)
 * print('time', end.time, ' mean', mean(end.x), ' variance', variance(end.x))
 * print('exact mean', exact.mean, ' variance', exact.covariance)
 */
export function reverseSdeSampler(
  predictor: NoisePredictor,
  sde: ForwardSde,
  { steps = 500, end = 1e-3 }: ContinuousOptions = {},
): Algorithm<SamplerStart, SamplerState> {
  const h = (1 - end) / steps
  return {
    name: 'reverse-sde',
    init: (opts, s) => start(opts, s, sde.priorStd, 1, 1),
    step: (st, ctx) => {
      const t = st.time
      const m = sde.meanScale(t)
      const sd = sde.std(t)
      const g = sde.diffusion(t)
      const noise = predictNoise(predictor, st.x, m, sd)
      const score = mul(noise, -1 / sd)
      const drift = sub(mul(st.x, sde.drift(t)), mul(score, g * g))
      const z = normals(ctx.stream, st.x.shape)
      const x = add(sub(st.x, mul(drift, h)), mul(z, g * Math.sqrt(h)))
      const time = st.t + 1 === steps ? end : 1 - (st.t + 1) * h
      return {
        ...st,
        t: st.t + 1,
        time,
        tau: time,
        x,
        noise,
        clean: predictClean(noise, st.x, m, sd),
        evaluations: st.evaluations + 1,
      }
    },
    done: (st) => st.t >= steps,
  }
}

/** Options of `probabilityFlowSampler`. */
export type ProbabilityFlowOptions = ContinuousOptions & {
  /** `'rk4'` (default; four predictor calls per step) or `'euler'` (one). */
  method?: 'rk4' | 'euler'
}

/**
 * The probability-flow ODE $d\xvec/dt = f(t)\xvec - \frac{1}{2}g(t)^2\nabla \log p_t(\xvec)$ (Song et al., 2021,
 * eq. 13) from $t = 1$ to $\varepsilon$, whose solutions have the same marginals $p_t$ as the SDE but are deterministic
 * given $\xvec_T$. Integrated by `aifn-compute/dynamics/ode`'s `rungeKutta` with the classical fourth-order tableau
 * (Kutta, 1901) or Euler's, time falling in $N$ equal steps. Only the prior draw is random.
 *
 * @param predictor The VP noise predictor $\hat\epsilonvec$.
 * @param sde The forward SDE.
 * @param options The number of steps, the final time and the integrator.
 * @param options.steps The number of steps $N$.
 * @param options.end The final time $\varepsilon$.
 * @param options.method The integrator, `'rk4'` or `'euler'`.
 * @returns The algorithm, started from a `SamplerStart` and done after $N$ steps; `evaluations` counts every
 *   predictor call.
 *
 * @example Fifty RK4 steps on the VE SDE; mirrored starts give mirrored samples
 * const mixture = gaussianMixtureData([1, 1], [[-2], [2]], [0.5, 0.5])
 * const sampler = probabilityFlowSampler(mixtureNoisePredictor(mixture), veSde(), { steps: 50 })
 * const end = run(sampler, { n: 1000, dimension: 1 }, 50)
 * print('time', end.time, ' evaluations', end.evaluations, ' mean', mean(end.x), ' variance', variance(end.x))
 * print('from +50 and -50, one prior standard deviation:', run(sampler, { x: tensor([[50], [-50]]) }, 50).x)
 */
export function probabilityFlowSampler(
  predictor: NoisePredictor,
  sde: ForwardSde,
  { steps = 100, end = 1e-3, method = 'rk4' }: ProbabilityFlowOptions = {},
): Algorithm<SamplerStart, SamplerState> {
  const h = (1 - end) / steps
  // The ODE's velocity at (x, t), and the noise prediction it came from.
  const flow = (x: Tensor, t: number) => {
    const g = sde.diffusion(t)
    const sd = sde.std(t)
    const noise = predictNoise(predictor, x, sde.meanScale(t), sd)
    return { v: sub(mul(x, sde.drift(t)), mul(noise, (-0.5 * g * g) / sd)), noise }
  }
  return {
    name: 'probability-flow',
    init: (opts, s) => start(opts, s, sde.priorStd, 1, 1),
    step: (st, ctx) => {
      const t = st.time
      const shape = st.x.shape
      // The first stage evaluates the flow at (x, t); its noise prediction is the one the state reports.
      let first: Tensor | null = null
      const rhs: Rhs = (time, x) => {
        const out = flow(reshape(x, shape), time)
        first ??= out.noise
        return reshape(out.v, [x.shape[0]])
      }
      const ode = rungeKutta(rhs, method === 'euler' ? 'euler' : 'rk4', { stepSize: -h })
      const solved = ode.step(
        ode.init({ x0: reshape(st.x, [st.x.shape.reduce((a, b) => a * b, 1)]), t0: t }, ctx.stream),
        ctx,
      )
      const x = reshape(solved.x, shape)
      const noise = first ?? flow(st.x, t).noise
      const time = st.t + 1 === steps ? end : 1 - (st.t + 1) * h
      const clean = predictClean(noise, st.x, sde.meanScale(t), sde.std(t))
      return { ...st, t: st.t + 1, time, tau: time, x, noise, clean, evaluations: st.evaluations + solved.evaluations }
    },
    done: (st) => st.t >= steps,
  }
}
