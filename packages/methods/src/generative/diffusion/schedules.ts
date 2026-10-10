/**
 * Noise schedules, discrete and continuous, and the map between their time axes.
 *
 * A discrete schedule (Ho et al., 2020) lists $\beta_1, \dots, \beta_T$ and the derived $\alpha_t = 1 - \beta_t$,
 * $\bar\alpha_t = \alpha_1 \cdots \alpha_t$ and signal-to-noise ratios
 * $\mathrm{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$, so that
 * $\xvec_t \mid \xvec_0 \sim \Gauss(\sqrt{\bar\alpha_t}\,\xvec_0, (1 - \bar\alpha_t)\Imat)$. A continuous forward SDE
 * $d\xvec = f(t)\xvec \, dt + g(t) \, d\wvec$ on $t \in [0, 1]$ (Song et al., 2021, "Score-based generative modeling
 * through stochastic differential equations", §3.4 and appendix B) has the Gaussian marginal
 * $\xvec_t \mid \xvec_0 \sim \Gauss(m(t)\xvec_0, s(t)^2\Imat)$. Step $t$ of a $T$-step schedule sits at time
 * $\tau = t/T$ of the continuous axis (`stepTime`), and `scheduleSde` and `sdeSchedule` turn either kind into the
 * other. Steps outside the schedule throw `DomainError`.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A discrete noise schedule of $T$ steps. Entry $t - 1$ of each tensor (each of length $T$) belongs to step
 * $t = 1, \dots, T$.
 */
export type NoiseSchedule = {
  /** How it was made: `linearSchedule`, `cosineSchedule`, or anything else (`scheduleFromBetas`, `sdeSchedule`). */
  readonly kind: 'linear' | 'cosine' | 'custom'
  /** The number of steps $T$. */
  readonly steps: number
  /** $\beta_1, \dots, \beta_T$, the variance added at each step. */
  readonly betas: Tensor
  /** $\alpha_t = 1 - \beta_t$. */
  readonly alphas: Tensor
  /** $\bar\alpha_t = \prod_{s \le t} \alpha_s$, the fraction of signal variance left after $t$ steps. */
  readonly alphaBars: Tensor
  /** $\mathrm{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$. */
  readonly snr: Tensor
  /** Steps whose $\beta_t$ was capped (the cosine schedule caps it at `maxBeta`); empty when none was. */
  readonly capped: readonly number[]
}

/**
 * A schedule from its $\beta_t$: the running products $\bar\alpha_t$ and the signal-to-noise ratios are derived here.
 * Throws `DomainError` when a $\beta_t$ is not in $(0, 1)$.
 *
 * @param betas $\beta_1, \dots, \beta_T$, one per step; their count is the number of steps $T$.
 * @param kind The label stored as the schedule's `kind`.
 * @param capped The steps whose $\beta_t$ was capped by the caller, stored as `capped` (only reported, not applied).
 * @returns The schedule.
 *
 * @example Three steps of $\beta = 0.1, 0.2, 0.5$
 * const s = scheduleFromBetas([0.1, 0.2, 0.5])
 * print('alpha:', s.alphas)
 * print('alpha bar:', s.alphaBars)
 * print('SNR:', s.snr)
 */
export function scheduleFromBetas(
  betas: ArrayLike<number>,
  kind: NoiseSchedule['kind'] = 'custom',
  capped: readonly number[] = [],
): NoiseSchedule {
  const T = betas.length
  const b = Float64Array.from(betas)
  const a = new Float64Array(T)
  const ab = new Float64Array(T)
  const snr = new Float64Array(T)
  let prod = 1
  for (let i = 0; i < T; i++) {
    if (!(b[i] > 0 && b[i] < 1)) throw new DomainError('schedule', `schedule: β_${i + 1} = ${b[i]} is not in (0, 1)`)
    a[i] = 1 - b[i]
    prod *= a[i]
    ab[i] = prod
    snr[i] = prod / (1 - prod)
  }
  const v = (d: Float64Array) => fromData(d, [T])
  return { kind, steps: T, betas: v(b), alphas: v(a), alphaBars: v(ab), snr: v(snr), capped }
}

/**
 * The linear schedule of Ho et al. (2020, §4): $\beta_t$ rises linearly from `betaStart` = 1e-4 to `betaEnd` = 0.02
 * over $T = 1000$ steps (the defaults), leaving $\bar\alpha_T \approx 4 \cdot 10^{-5}$.
 *
 * @param steps The number of steps $T$.
 * @param options The end points of the line.
 * @param options.betaStart $\beta_1$ (and every $\beta_t$ when $T = 1$).
 * @param options.betaEnd $\beta_T$.
 * @returns The schedule, of kind `'linear'`.
 *
 * @example How much signal is left after $t$ of 1000 steps
 * const lin = linearSchedule(1000)
 * print('alpha bar at t = 1, 250, 500, 1000:', [1, 250, 500, 1000].map((t) => alphaBarAt(lin, t)))
 */
export function linearSchedule(
  steps = 1000,
  { betaStart = 1e-4, betaEnd = 0.02 }: { betaStart?: number; betaEnd?: number } = {},
): NoiseSchedule {
  const betas = Float64Array.from({ length: steps }, (_, i) =>
    steps === 1 ? betaStart : betaStart + ((betaEnd - betaStart) * i) / (steps - 1),
  )
  return scheduleFromBetas(betas, 'linear')
}

/**
 * The cosine schedule of Nichol & Dhariwal (2021, eq. 17): $\bar\alpha_t = f(t)/f(0)$ with
 * $f(t) = \cos^2\bigl(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\bigr)$, so $\bar\alpha$ falls slowly at both ends;
 * $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$, capped at `maxBeta` = 0.999 as in the paper. The steps capped are
 * reported in `capped`: always the last, where $f(T) = 0$ makes $\beta_T = 1$, and with the defaults only the last.
 *
 * @param steps The number of steps $T$.
 * @param options The offset and the cap.
 * @param options.offset The offset $s$, which keeps $\beta_t$ from being too small near $t = 0$.
 * @param options.maxBeta The largest $\beta_t$ allowed; larger ones are replaced by it.
 * @returns The schedule, of kind `'cosine'`.
 *
 * @example The cosine schedule keeps more signal than the linear one in the middle
 * const lin = linearSchedule(1000)
 * const cos = cosineSchedule(1000)
 * for (const t of [250, 500, 750]) print(`t = ${t}: linear`, alphaBarAt(lin, t), ' cosine', alphaBarAt(cos, t))
 * print('capped steps:', cos.capped)
 */
export function cosineSchedule(
  steps = 1000,
  { offset = 0.008, maxBeta = 0.999 }: { offset?: number; maxBeta?: number } = {},
): NoiseSchedule {
  const f = (t: number) => Math.cos((((t / steps + offset) / (1 + offset)) * Math.PI) / 2) ** 2
  const capped: number[] = []
  const betas = Float64Array.from({ length: steps }, (_, i) => {
    const beta = 1 - f(i + 1) / f(i)
    if (beta > maxBeta) {
      capped.push(i + 1)
      return maxBeta
    }
    return beta
  })
  return scheduleFromBetas(betas, 'cosine', capped)
}

/**
 * $\bar\alpha_t$ at step $t$, with $\bar\alpha_0 = 1$ (no noise). Throws `DomainError` for a step that is not an
 * integer in $0, \dots, T$.
 *
 * @param schedule The schedule.
 * @param t The step, from 0 to $T$.
 * @returns $\bar\alpha_t$.
 *
 * @example A constant $\beta = 0.1$ leaves $0.9^t$ of the signal
 * const s = linearSchedule(10, { betaStart: 0.1, betaEnd: 0.1 })
 * print('alpha bar at t = 0, 1, 2:', [0, 1, 2].map((t) => alphaBarAt(s, t)))
 */
export function alphaBarAt(schedule: NoiseSchedule, t: number): number {
  if (t === 0) return 1
  if (!Number.isInteger(t) || t < 0 || t > schedule.steps)
    throw new DomainError('alphaBarAt', `step ${t} is not in 0 … ${schedule.steps}`)
  return schedule.alphaBars.data[t - 1]
}

/**
 * $\beta_t$ at step $t$. Throws `DomainError` for a step that is not an integer in $1, \dots, T$.
 *
 * @param schedule The schedule.
 * @param t The step, from 1 to $T$.
 * @returns $\beta_t$.
 *
 * @example The ends and the middle of the linear schedule
 * const lin = linearSchedule(1000)
 * print('beta at t = 1, 500, 1000:', betaAt(lin, 1), betaAt(lin, 500), betaAt(lin, 1000))
 */
export function betaAt(schedule: NoiseSchedule, t: number): number {
  if (!Number.isInteger(t) || t < 1 || t > schedule.steps)
    throw new DomainError('betaAt', `step ${t} is not in 1 … ${schedule.steps}`)
  return schedule.betas.data[t - 1]
}

// ── Continuous-time forward SDEs ─────────────────────────────────────────────────────────────────────────────────────

/**
 * A linear forward SDE $d\xvec = f(t)\xvec \, dt + g(t) \, d\wvec$ on $t \in [0, 1]$, with its Gaussian marginal
 * $\xvec_t \mid \xvec_0 \sim \Gauss(m(t)\xvec_0, s(t)^2\Imat)$.
 */
export type ForwardSde = {
  /** Which SDE: variance-preserving, sub-VP or variance-exploding. */
  readonly kind: 'vp' | 'subVp' | 've'
  /** The discrete schedule this SDE interpolates (`scheduleSde`), when it came from one. */
  readonly schedule?: NoiseSchedule
  /** The drift coefficient $f(t)$ (the drift is $f(t)\xvec$). */
  drift(t: number): number
  /** The diffusion coefficient $g(t)$. */
  diffusion(t: number): number
  /** The mean scale $m(t)$ of the marginal. */
  meanScale(t: number): number
  /** The standard deviation $s(t)$ of the marginal. */
  std(t: number): number
  /** The standard deviation $\sigma$ of the prior $\Gauss(\zeros, \sigma^2\Imat)$ at $t = 1$, where sampling starts. */
  priorStd: number
}

/**
 * The variance-preserving SDE (Song et al., 2021, eq. 11), the continuous limit of DDPM:
 * $\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$, $f = -\beta(t)/2$, $g = \sqrt{\beta(t)}$,
 * $m(t) = \exp(-\frac{1}{2}\int_0^t \beta)$ and $s(t) = \sqrt{1 - m(t)^2}$, so $m^2 + s^2 = 1$ at every time. The
 * prior is $\Gauss(\zeros, \Imat)$.
 *
 * @param options The rate's ends.
 * @param options.betaMin $\beta_{\min} = \beta(0)$.
 * @param options.betaMax $\beta_{\max} = \beta(1)$.
 * @returns The SDE, of kind `'vp'`.
 *
 * @example The marginal's scale and spread over time
 * const sde = vpSde()
 * for (const t of [0.1, 0.5, 1]) {
 *   const m = sde.meanScale(t)
 *   const s = sde.std(t)
 *   print(`t = ${t}: m`, m, ' s', s, ' m^2 + s^2', m * m + s * s)
 * }
 */
export function vpSde({ betaMin = 0.1, betaMax = 20 }: { betaMin?: number; betaMax?: number } = {}): ForwardSde {
  const beta = (t: number) => betaMin + t * (betaMax - betaMin)
  const integral = (t: number) => betaMin * t + 0.5 * (betaMax - betaMin) * t * t
  const meanScale = (t: number) => Math.exp(-0.5 * integral(t))
  return {
    kind: 'vp',
    drift: (t) => -0.5 * beta(t),
    diffusion: (t) => Math.sqrt(beta(t)),
    meanScale,
    std: (t) => Math.sqrt(-Math.expm1(-integral(t))),
    priorStd: 1,
  }
}

/**
 * The sub-VP SDE (Song et al., 2021, eq. 12): the VP drift and mean scale with
 * $g(t)^2 = \beta(t)(1 - e^{-2\int_0^t \beta})$, whose marginal standard deviation $s(t) = 1 - m(t)^2$ is below the
 * VP one, $\sqrt{1 - m(t)^2}$, at every time. The prior is $\Gauss(\zeros, \Imat)$.
 *
 * @param options The rate's ends, as `vpSde`'s.
 * @param options.betaMin $\beta_{\min} = \beta(0)$.
 * @param options.betaMax $\beta_{\max} = \beta(1)$.
 * @returns The SDE, of kind `'subVp'`.
 *
 * @example Less noise than the VP SDE at every time
 * const vp = vpSde()
 * const subVp = subVpSde()
 * for (const t of [0.1, 0.5, 1]) print(`t = ${t}: VP std`, vp.std(t), ' sub-VP std', subVp.std(t))
 */
export function subVpSde({ betaMin = 0.1, betaMax = 20 }: { betaMin?: number; betaMax?: number } = {}): ForwardSde {
  const vp = vpSde({ betaMin, betaMax })
  const beta = (t: number) => betaMin + t * (betaMax - betaMin)
  const integral = (t: number) => betaMin * t + 0.5 * (betaMax - betaMin) * t * t
  return {
    ...vp,
    kind: 'subVp',
    diffusion: (t) => Math.sqrt(beta(t) * -Math.expm1(-2 * integral(t))),
    std: (t) => -Math.expm1(-integral(t)),
  }
}

/**
 * The variance-exploding SDE (Song et al., 2021, eq. 9): $\sigma(t) = \sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t$,
 * $f = 0$, $g = \sigma(t)\sqrt{2 \log(\sigma_{\max}/\sigma_{\min})}$, $m = 1$ and $s(t) = \sigma(t)$. The prior is
 * $\Gauss(\zeros, \sigma_{\max}^2\Imat)$.
 *
 * @param options The noise scale's ends.
 * @param options.sigmaMin $\sigma_{\min} = \sigma(0)$.
 * @param options.sigmaMax $\sigma_{\max} = \sigma(1)$, also the prior's standard deviation.
 * @returns The SDE, of kind `'ve'`.
 *
 * @example The noise grows geometrically and the signal is never scaled
 * const ve = veSde()
 * for (const t of [0, 0.5, 1]) print(`t = ${t}: std`, ve.std(t), ' mean scale', ve.meanScale(t))
 */
export function veSde({ sigmaMin = 0.01, sigmaMax = 50 }: { sigmaMin?: number; sigmaMax?: number } = {}): ForwardSde {
  const sigma = (t: number) => sigmaMin * (sigmaMax / sigmaMin) ** t
  const k = Math.sqrt(2 * Math.log(sigmaMax / sigmaMin))
  return {
    kind: 've',
    drift: () => 0,
    diffusion: (t) => sigma(t) * k,
    meanScale: () => 1,
    std: sigma,
    priorStd: sigmaMax,
  }
}

// ── One time axis: discrete steps and SDE time ──────────────────────────────────────────────────────────────────────

/**
 * The continuous time $\tau = t/T \in [0, 1]$ of discrete step $t = 0, \dots, T$. Every sampler reports this axis as
 * `tau`, so a DDPM or DDIM run on a schedule and a continuous run on `scheduleSde(schedule)` are plotted against the
 * same time. Throws `DomainError` for a step that is not an integer in $0, \dots, T$.
 *
 * @param schedule The schedule, for its $T$.
 * @param t The step.
 * @returns $\tau = t/T$.
 *
 * @example A step's time, and back
 * const s = linearSchedule(1000)
 * print('step 250 is at time', stepTime(s, 250), '; time 0.3 is step', timeStep(s, 0.3))
 */
export function stepTime(schedule: NoiseSchedule, t: number): number {
  if (!Number.isInteger(t) || t < 0 || t > schedule.steps)
    throw new DomainError('stepTime', `step ${t} is not in 0 … ${schedule.steps}`)
  return t / schedule.steps
}

/**
 * The discrete step nearest to time $\tau \in [0, 1]$, $\operatorname{round}(\tau T)$: the inverse of `stepTime` on the
 * grid. Throws `DomainError` for a time outside $[0, 1]$.
 *
 * @param schedule The schedule, for its $T$.
 * @param tau The time $\tau$.
 * @returns The step, from 0 to $T$.
 *
 * @example Times between grid points round to the nearest step
 * const s = linearSchedule(10)
 * print('steps of tau = 0, 0.33, 0.37, 1:', [0, 0.33, 0.37, 1].map((tau) => timeStep(s, tau)))
 */
export function timeStep(schedule: NoiseSchedule, tau: number): number {
  if (!(tau >= 0 && tau <= 1)) throw new DomainError('timeStep', `time ${tau} is not in [0, 1]`)
  return Math.round(tau * schedule.steps)
}

/**
 * The variance-preserving SDE that passes through a discrete schedule: at $\tau = t/T$ its marginal is exactly the
 * schedule's, $m(\tau)^2 = \bar\alpha_t$ and $s(\tau)^2 = 1 - \bar\alpha_t$, with $\log \bar\alpha$ linear in $\tau$
 * between steps. Its rate is piecewise constant, $\beta(\tau) = -T \log \alpha_t$ on $((t - 1)/T, t/T]$, so
 * $f = -\beta/2$ and $g = \sqrt{\beta}$, and $\beta(\tau)/T \approx \beta_t$ for small $\beta_t$: the continuous limit
 * of DDPM (Song et al., 2021, appendix B) taken on the schedule itself rather than on a separately parameterised
 * $\beta(t)$. DDPM, DDIM and probability-flow sampling on `scheduleSde(schedule)` therefore share one time axis and
 * one marginal at every step. Times outside $[0, 1]$ are clamped.
 *
 * @param schedule The discrete schedule to pass through.
 * @returns The SDE, of kind `'vp'`, with `schedule` attached.
 *
 * @example The SDE's marginal matches the schedule at its steps
 * const s = linearSchedule(1000)
 * const sde = scheduleSde(s)
 * for (const t of [100, 500]) {
 *   print(`step ${t}: alpha bar`, alphaBarAt(s, t), ' m(t/T)^2', sde.meanScale(stepTime(s, t)) ** 2)
 * }
 */
export function scheduleSde(schedule: NoiseSchedule): ForwardSde {
  const T = schedule.steps
  // logAb[t] = log ᾱₜ with logAb[0] = 0; rate[t − 1] = −T log αₜ.
  const logAb = new Float64Array(T + 1)
  const rate = new Float64Array(T)
  for (let t = 1; t <= T; t++) {
    const logAlpha = Math.log1p(-schedule.betas.data[t - 1])
    logAb[t] = logAb[t - 1] + logAlpha
    rate[t - 1] = -T * logAlpha
  }
  const cell = (tau: number) => Math.min(T, Math.max(1, Math.ceil(tau * T - 1e-12)))
  const logAlphaBar = (tau: number) => {
    const u = Math.min(1, Math.max(0, tau)) * T
    const t = cell(u / T)
    return logAb[t - 1] + (u - (t - 1)) * (logAb[t] - logAb[t - 1])
  }
  const beta = (tau: number) => rate[cell(tau) - 1]
  return {
    kind: 'vp',
    schedule,
    drift: (tau) => -0.5 * beta(tau),
    diffusion: (tau) => Math.sqrt(beta(tau)),
    meanScale: (tau) => Math.exp(0.5 * logAlphaBar(tau)),
    std: (tau) => Math.sqrt(-Math.expm1(logAlphaBar(tau))),
    priorStd: 1,
  }
}

/**
 * A discrete schedule of $T$ steps read off a forward SDE at $\tau = t/T$: $\bar\alpha_t = m^2/(m^2 + s^2)$, the signal
 * level at which the VP noise predictor sees the SDE's marginal (see `predictNoise`), and
 * $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$ with $\bar\alpha_0 = 1$. For the VP SDE $\bar\alpha_t = m(t/T)^2$,
 * so `sdeSchedule(vpSde(), 1000)` is close to `linearSchedule(1000)`; `sdeSchedule(scheduleSde(s), s.steps)` gives back
 * `s`. Throws `DomainError` when `steps` is not a positive integer, and (from `scheduleFromBetas`) when a $\beta_t$
 * falls outside $(0, 1)$.
 *
 * @param sde The forward SDE.
 * @param steps The number of steps $T$.
 * @returns The schedule, of kind `'custom'`.
 *
 * @example The VP SDE read as 1000 steps is close to the linear schedule
 * const fromSde = sdeSchedule(vpSde(), 1000)
 * const lin = linearSchedule(1000)
 * for (const t of [100, 500, 1000]) {
 *   print(`step ${t}: from the VP SDE`, alphaBarAt(fromSde, t), ' linear', alphaBarAt(lin, t))
 * }
 */
export function sdeSchedule(sde: ForwardSde, steps: number): NoiseSchedule {
  if (!Number.isInteger(steps) || steps < 1)
    throw new DomainError('sdeSchedule', `sdeSchedule: steps = ${steps} is not a positive integer`)
  const betas = new Float64Array(steps)
  let prev = 1
  for (let t = 1; t <= steps; t++) {
    const m = sde.meanScale(t / steps)
    const sd = sde.std(t / steps)
    const ab = (m * m) / (m * m + sd * sd)
    betas[t - 1] = 1 - ab / prev
    prev = ab
  }
  return scheduleFromBetas(betas, 'custom')
}
