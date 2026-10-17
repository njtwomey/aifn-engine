/**
 * Two classification problems on the real line whose risk is known in closed form for every threshold
 * $h_t(x) = \indicator[x > t]$, and a learning-curve experiment over them, for the value of unlabelled data.
 *
 * `clustersProblem` has two equally likely classes $x \mid y = 0 \sim \Gauss(-s/2, 1)$ and
 * $x \mid y = 1 \sim \Gauss(s/2, 1)$, $s$ the separation in standard deviations: the input density has a trough at the
 * Bayes boundary $t = 0$ once $s > 2$, and the Bayes risk is $\Phi(-s/2)$ (the two-component mixture of Castelli and
 * Cover, 1995). `uniformProblem` draws $x \sim \Unif(0, 1)$ and labels it $\indicator[x > t^*]$: the input density is
 * flat, so it says nothing about where $t^*$ lies, and the Bayes risk is 0 (the setting of Ben-David, Lu and Pál, 2008,
 * §6.2). The risk of a threshold is $R(t) = \tfrac12(1 - \Phi(t + s/2)) + \tfrac12\Phi(t - s/2)$ for the clusters and
 * $\lvert t - t^* \rvert$ for the uniform problem.
 *
 * The experiment (`thresholdLearningCurve`) draws, for each repeat, $u$ unlabelled inputs and the largest number of
 * labelled points per class, and fits the supervised `thresholdErm` and the semi-supervised `lowDensityThreshold` of
 * `aifn-methods/learning/semi-supervised` on the first $m$ labelled points of each class for every $m$ in the list
 * (nested samples, so each curve is a sequence of growing samples). It returns the mean risk of each learner and its
 * standard error.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'
import {
  lowDensityThreshold,
  thresholdErm,
  unlabelledDensity,
  type GridDensity,
  type LineSample,
} from 'aifn-methods/learning/semi-supervised'

/** A problem on the line: how to draw inputs and how to score a threshold. */
export interface LineProblem {
  /** The input range used to bound open intervals and to draw the problem. */
  domain: readonly [number, number]
  /** The Bayes threshold. */
  bayes: number
  /** The Bayes risk. */
  bayesRisk: number
  /** The risk $R(t)$ of $h_t(x) = \indicator[x > t]$. */
  risk(t: number): number
  /** The input density $p(x)$. */
  density(x: number): number
  /** One input drawn from class $c$. */
  drawClass(s: Stream, c: 0 | 1): number
  /** One unlabelled input drawn from $p(x)$. */
  drawInput(s: Stream): number
  /** The input distribution function $F$, for the median rule. */
  cdf(x: number): number
  /** Its inverse $F^{-1}$. */
  quantile(p: number): number
}

/** A line problem as plain data, for a worker: the clusters with their separation, or the uniform problem. */
export type LineProblemSpec = { kind: 'clusters'; separation: number } | { kind: 'uniform'; boundary: number }

/**
 * Two unit-variance Gaussian classes at $\pm s/2$ with equal priors; see the file comment. The quantile is found by 60
 * bisections of the mixture's distribution function, started from the bracketing normal quantiles.
 *
 * @param separation The distance $s$ between the class means, in standard deviations.
 * @returns The problem on the domain $[-s/2 - 4, s/2 + 4]$.
 *
 * @example The Bayes risk is $\Phi(-s/2)$, and the risk at the Bayes threshold equals it
 * const p = clustersProblem(4)
 * print('Bayes risk:', p.bayesRisk, ' R(0):', p.risk(0), ' R(0.5):', p.risk(0.5))
 */
export function clustersProblem(separation: number): LineProblem {
  const h = separation / 2
  const domain: [number, number] = [-h - 4, h + 4]
  const cdf = (x: number) => 0.5 * normalCdf(x + h) + 0.5 * normalCdf(x - h)
  return {
    domain,
    bayes: 0,
    bayesRisk: normalCdf(-h),
    risk: (t) => 0.5 * (1 - normalCdf(t + h)) + 0.5 * normalCdf(t - h),
    density: (x) => (Math.exp(-0.5 * (x + h) ** 2) + Math.exp(-0.5 * (x - h) ** 2)) / (2 * Math.sqrt(2 * Math.PI)),
    drawClass: (s, c) => normal(s, c === 0 ? -h : h, 1),
    drawInput: (s) => normal(s, uniform(s) < 0.5 ? -h : h, 1),
    cdf,
    quantile: (p) => {
      let a = normalQuantile(Math.min(Math.max(p, 1e-12), 1 - 1e-12)) - h
      let b = a + 2 * h
      for (let i = 0; i < 60; i++) {
        const mid = (a + b) / 2
        if (cdf(mid) < p) a = mid
        else b = mid
      }
      return (a + b) / 2
    },
  }
}

/**
 * $x \sim \Unif(0, 1)$ labelled $\indicator[x > t^*]$; see the file comment.
 *
 * @param boundary The true threshold $t^*$, clipped to $[0.01, 0.99]$.
 * @returns The problem on the domain $[0, 1]$, with Bayes risk 0.
 *
 * @example The risk of a threshold is its distance from $t^*$
 * const p = uniformProblem(0.3)
 * print('R(0.45):', p.risk(0.45), ' R(0.3):', p.risk(0.3))
 */
export function uniformProblem(boundary: number): LineProblem {
  const t = Math.min(Math.max(boundary, 0.01), 0.99)
  return {
    domain: [0, 1],
    bayes: t,
    bayesRisk: 0,
    risk: (x) => Math.abs(Math.min(Math.max(x, 0), 1) - t),
    density: (x) => (x >= 0 && x <= 1 ? 1 : 0),
    drawClass: (s, c) => (c === 0 ? uniform(s, 0, t) : uniform(s, t, 1)),
    drawInput: (s) => uniform(s),
    cdf: (x) => Math.min(Math.max(x, 0), 1),
    quantile: (p) => p,
  }
}

/**
 * The line problem a plain-data spec names: `clustersProblem` or `uniformProblem`.
 *
 * @param spec The kind of problem and its parameter.
 * @returns The problem.
 *
 * @example The same problem from its spec
 * print(lineProblem({ kind: 'clusters', separation: 4 }).bayesRisk)
 */
export function lineProblem(spec: LineProblemSpec): LineProblem {
  if (spec.kind === 'clusters') return clustersProblem(spec.separation)
  if (spec.kind === 'uniform') return uniformProblem(spec.boundary)
  throw new DomainError('lineProblem', `lineProblem: unknown kind ${(spec as { kind: string }).kind}`)
}

/** One draw of the experiment: the unlabelled inputs and the labelled points of each class, in drawing order. */
export interface LineDraw {
  /** The unlabelled inputs. */
  unlabelled: number[]
  /** The labelled inputs of class 0. */
  class0: number[]
  /** The labelled inputs of class 1. */
  class1: number[]
}

/**
 * Draw $u$ unlabelled inputs and $m$ labelled points per class, from the children `'unlabelled'` and
 * `'labelled', c` of the stream.
 *
 * @param s The stream.
 * @param problem The line problem.
 * @param unlabelled The number $u$ of unlabelled inputs.
 * @param perClass The number $m$ of labelled points of each class.
 * @returns The draw.
 *
 * @example Three labelled points per class of the clusters problem
 * const d = drawLine(stream(1), clustersProblem(4), 5, 3)
 * print('class 0:', d.class0, ' class 1:', d.class1)
 */
export function drawLine(s: Stream, problem: LineProblem, unlabelled: number, perClass: number): LineDraw {
  const su = child(s, 'unlabelled')
  const s0 = child(s, 'labelled', 0)
  const s1 = child(s, 'labelled', 1)
  return {
    unlabelled: Array.from({ length: unlabelled }, () => problem.drawInput(su)),
    class0: Array.from({ length: perClass }, () => problem.drawClass(s0, 0)),
    class1: Array.from({ length: perClass }, () => problem.drawClass(s1, 1)),
  }
}

/**
 * The labelled sample made of the first $m$ points of each class of a draw.
 *
 * @param draw The draw.
 * @param m The number of points taken from each class.
 * @returns The sample, class 0 first.
 *
 * @example The first point of each class
 * print(firstLabelled(drawLine(stream(1), uniformProblem(0.3), 0, 4), 1))
 */
export function firstLabelled(draw: LineDraw, m: number): LineSample {
  const a = draw.class0.slice(0, m)
  const b = draw.class1.slice(0, m)
  return { x: [...a, ...b], y: [...a.map(() => 0), ...b.map(() => 1)] }
}

/**
 * Both learners on the first $m$ labelled points of each class and the draw's unlabelled inputs.
 *
 * @param problem The line problem, whose domain bounds the intervals.
 * @param draw The draw.
 * @param m The number of labelled points of each class.
 * @param density The unlabelled density (default `unlabelledDensity` of the draw's unlabelled inputs); pass it to
 *   reuse one estimate across sizes.
 * @returns The supervised and the semi-supervised threshold.
 *
 * @example One labelled point per class: the trough helps on the clusters
 * const p = clustersProblem(4)
 * const fit = fitBoth(p, drawLine(stream(2), p, 200, 1), 1)
 * print('thresholds:', fit, ' risks:', p.risk(fit.supervised), p.risk(fit.semiSupervised))
 */
export function fitBoth(
  problem: LineProblem,
  draw: LineDraw,
  m: number,
  density: GridDensity | null = unlabelledDensity(draw.unlabelled, problem.domain),
): { supervised: number; semiSupervised: number } {
  const sample = firstLabelled(draw, m)
  return {
    supervised: thresholdErm(sample, problem.domain),
    semiSupervised: lowDensityThreshold(sample, density, problem.domain),
  }
}

/** Mean risk and its standard error for one learner at each labelled size. */
export interface CurvePoint {
  /** The mean risk over the repeats. */
  mean: number
  /** Its standard error. */
  se: number
}

/** The learning curves of the supervised and the semi-supervised threshold. */
export interface ThresholdCurves {
  /** The labelled sizes per class. */
  perClass: number[]
  /** The supervised learner's curve. */
  supervised: CurvePoint[]
  /** The semi-supervised learner's curve. */
  semiSupervised: CurvePoint[]
}

/**
 * Learning curves of `thresholdErm` and `lowDensityThreshold` on a line problem; see the file comment. Repeat $r$
 * draws from the child `'repeat', r` of the stream. Given the problem as a plain-data spec, the call is a pure function
 * of serialisable inputs, so a worker can run it (about 65 ms for seven sizes, 200 unlabelled points and 100 repeats).
 *
 * @param s The stream.
 * @param problem The line problem, or its spec.
 * @param options The labelled sizes per class (increasing), the unlabelled count $u$ and the number of repeats.
 * @param options.perClass The labelled sizes $m$ per class.
 * @param options.unlabelled The number $u$ of unlabelled inputs in each repeat.
 * @param options.repeats The number of repeats.
 * @returns Mean risk and standard error per size for both learners.
 *
 * @example Unlabelled data help on the clusters, not on the uniform problem
 * const sizes = { perClass: [1, 4], unlabelled: 200, repeats: 20 }
 * const c = thresholdLearningCurve(stream(1), { kind: 'clusters', separation: 4 }, sizes)
 * const u = thresholdLearningCurve(stream(1), { kind: 'uniform', boundary: 0.3 }, sizes)
 * const means = (curve) => curve.map((p) => p.mean)
 * print('clusters:', means(c.supervised), means(c.semiSupervised))
 * print('uniform:', means(u.supervised), means(u.semiSupervised))
 */
export function thresholdLearningCurve(
  s: Stream,
  problem: LineProblem | LineProblemSpec,
  options: { perClass: readonly number[]; unlabelled: number; repeats: number },
): ThresholdCurves {
  const p = 'risk' in problem ? problem : lineProblem(problem)
  const sizes = [...options.perClass]
  const max = Math.max(...sizes)
  const R = options.repeats
  const sums = sizes.map(() => ({ a: 0, a2: 0, b: 0, b2: 0 }))
  for (let r = 0; r < R; r++) {
    const draw = drawLine(child(s, 'repeat', r), p, options.unlabelled, max)
    const density = unlabelledDensity(draw.unlabelled, p.domain)
    sizes.forEach((m, j) => {
      const { supervised, semiSupervised } = fitBoth(p, draw, m, density)
      const ra = p.risk(supervised)
      const rb = p.risk(semiSupervised)
      sums[j].a += ra
      sums[j].a2 += ra * ra
      sums[j].b += rb
      sums[j].b2 += rb * rb
    })
  }
  const point = (sum: number, sum2: number): CurvePoint => {
    const mean = sum / R
    const variance = R > 1 ? Math.max(0, (sum2 - R * mean * mean) / (R - 1)) : 0
    return { mean, se: Math.sqrt(variance / R) }
  }
  return {
    perClass: sizes,
    supervised: sums.map((v) => point(v.a, v.a2)),
    semiSupervised: sums.map((v) => point(v.b, v.b2)),
  }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

const THRESHOLD_NOTES = ['when-unlabelled-data-helps']

fn(
  {
    key: 'clustersProblem',
    name: 'Two Gaussian clusters on a line',
    summary: 'Unit-variance classes at ±s/2: a density trough marks the Bayes threshold once s > 2.',
    role: 'construction',
    notes: THRESHOLD_NOTES,
    cite: ['castelli1995exponential'],
  },
  clustersProblem,
)
fn(
  {
    key: 'uniformProblem',
    name: 'Uniform inputs with a threshold label',
    summary: 'x uniform on [0, 1], labelled 1 above t*: the input density says nothing about the boundary.',
    role: 'construction',
    notes: THRESHOLD_NOTES,
    cite: ['bendavid2008unlabeled'],
  },
  uniformProblem,
)
fn(
  {
    key: 'thresholdLearningCurve',
    name: 'Threshold learning curves',
    summary: 'Mean risk of the supervised and the low-density threshold as the labelled sample grows.',
    role: 'simulation',
    random: true,
    notes: THRESHOLD_NOTES,
    cite: ['castelli1995exponential', 'bendavid2008unlabeled'],
  },
  thresholdLearningCurve,
)
