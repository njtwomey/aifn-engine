/**
 * Seeded model outputs with a known truth, for calibration and conformal prediction: a classifier's logits that are
 * over- or under-confident (and optionally biased per class) relative to the true class probabilities, and a
 * regression model's point and quantile predictions on heteroscedastic data, with the interval width wrong by a set
 * factor. The calibration maps and conformal procedures these feed are in `aifn-compute/learning/calibration` and
 * `aifn-compute/learning/conformal`.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { categorical, child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { type Tensor } from 'aifn-compute/foundation/tensor'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { checkCount, labels, matrix, vector, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `classifierOutputs`. */
export interface ClassifierOutputsOptions {
  /** Examples n (default 2000). */
  n?: number
  /** Classes K (default 3). */
  classes?: number
  /** The standard deviation of the true logits: how separable the classes are (default 1.5). */
  separation?: number
  /** The factor on the reported logits: > 1 overconfident, < 1 underconfident, 1 calibrated (default 2.5). */
  temperature?: number
  /** A per-class offset added to the reported logits (class k gets bias · (k − (K − 1)/2)); 0 for none (default 0). */
  bias?: number
}

/** A classifier's outputs: reported logits `x` [n, K], labels `y` drawn from the true probabilities. */
export interface ClassifierOutputs extends Dataset {
  /** The true class probabilities [n, K]. */
  readonly trueProbabilities: Tensor
}

/**
 * Classifier outputs with a known truth: true logits zᵢ ~ N(0, s²I), labels yᵢ ~ softmax(zᵢ), and reported logits
 * T·zᵢ + b, where T > 1 makes the classifier overconfident (temperature scaling undoes it exactly) and the class
 * offsets b make it biased (which needs a matrix map such as Dirichlet calibration).
 */
export function classifierOutputs(s: Stream, options: ClassifierOutputsOptions = {}): ClassifierOutputs {
  const { n = 2000, classes: K = 3, separation = 1.5, temperature = 2.5, bias = 0 } = options
  checkCount(n, 'classifierOutputs')
  const cz = child(s, 'logits')
  const cy = child(s, 'labels')
  const reported = new Float64Array(n * K)
  const truth = new Float64Array(n * K)
  const y = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const z = Float64Array.from({ length: K }, () => separation * normal(cz))
    const hi = Math.max(...z)
    const e = z.map((v) => Math.exp(v - hi))
    const total = e.reduce((a, b) => a + b, 0)
    for (let k = 0; k < K; k++) {
      truth[i * K + k] = e[k] / total
      reported[i * K + k] = temperature * z[k] + bias * (k - (K - 1) / 2)
    }
    y[i] = categorical(cy, truth.subarray(i * K, (i + 1) * K))
  }
  return {
    kind: 'dataset',
    x: matrix(reported, n, K),
    y: labels(y),
    trueProbabilities: matrix(truth, n, K),
    meta: {
      name: 'classifier outputs',
      description: `${n} reported logits of a ${K}-class classifier, ${temperature > 1 ? 'overconfident' : temperature < 1 ? 'underconfident' : 'calibrated in scale'} (logits × ${temperature})${bias ? ', with per-class offsets' : ''}; labels drawn from the true probabilities.`,
      task: 'classification',
      featureNames: Array.from({ length: K }, (_, k) => `logit ${k}`),
      labelNames: Array.from({ length: K }, (_, k) => `class ${k}`),
      key: s.key,
    },
  }
}

/** Options of `quantileModelOutputs`. */
export interface QuantileModelOptions {
  /** Examples n (default 1000). */
  n?: number
  /** The noise standard deviation at the left end (default 0.2). */
  noise?: number
  /** The noise sd grows linearly to noise · (1 + heteroscedastic) at the right end (default 3). */
  heteroscedastic?: number
  /** The miscoverage level α of the predicted interval (default 0.1). */
  alpha?: number
  /**
   * The model's error in the interval width: it predicts the 1 − α interval with sd scale · σ(x) (default 0.6: too
   * narrow, so the raw intervals under-cover).
   */
  scale?: number
  /** 0: the model's width follows σ(x); 1: the model's width is constant (it ignores the heteroscedasticity). */
  homoscedastic?: number
}

/** A regression model's outputs on heteroscedastic data, with the true conditional sd beside them. */
export interface QuantileModelOutputs extends Dataset {
  /** The model's point prediction (the true mean) [n]. */
  readonly prediction: Tensor
  /** The model's predicted α/2 and 1 − α/2 quantiles [n]. */
  readonly lower: Tensor
  readonly upper: Tensor
  /** The true conditional standard deviation σ(xᵢ) [n]. */
  readonly sd: Tensor
}

/**
 * Data y = sin(2πx) + σ(x)ε on x ~ U(0, 1), with σ growing to the right, and a model that predicts the true mean and a
 * 1 − α Gaussian interval whose width is wrong by `scale` (and blind to the heteroscedasticity as `homoscedastic` → 1).
 * Split conformal on |y − ŷ| repairs the coverage with a constant width; conformalised quantile regression repairs it
 * while keeping the model's shape.
 */
export function quantileModelOutputs(s: Stream, options: QuantileModelOptions = {}): QuantileModelOutputs {
  const { n = 1000, noise = 0.2, heteroscedastic = 3, alpha = 0.1, scale = 0.6, homoscedastic = 0 } = options
  checkCount(n, 'quantileModelOutputs')
  if (!(alpha > 0 && alpha < 1))
    throw new DomainError('quantileModelOutputs', 'quantileModelOutputs: α must be in (0, 1)')
  const z = normalQuantile(1 - alpha / 2) as number
  const cx = child(s, 'x')
  const ce = child(s, 'noise')
  const x = new Float64Array(n)
  const y = new Float64Array(n)
  const f = new Float64Array(n)
  const sd = new Float64Array(n)
  const lo = new Float64Array(n)
  const hi = new Float64Array(n)
  const average = noise * (1 + heteroscedastic / 2)
  for (let i = 0; i < n; i++) {
    x[i] = uniform(cx)
    f[i] = Math.sin(2 * Math.PI * x[i])
    sd[i] = noise * (1 + heteroscedastic * x[i])
    y[i] = f[i] + sd[i] * normal(ce)
    const width = scale * z * ((1 - homoscedastic) * sd[i] + homoscedastic * average)
    lo[i] = f[i] - width
    hi[i] = f[i] + width
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 1),
    y: vector(y),
    f: vector(f),
    prediction: vector(f),
    lower: vector(lo),
    upper: vector(hi),
    sd: vector(sd),
    meta: {
      name: 'quantile model outputs',
      description: `${n} observations of y = sin(2πx) with noise sd growing from ${noise} to ${noise * (1 + heteroscedastic)}, and a model's ${Math.round(100 * (1 - alpha))}% intervals ${scale < 1 ? 'too narrow' : scale > 1 ? 'too wide' : 'of the right width'} by a factor ${scale}.`,
      task: 'regression',
      featureNames: ['x'],
      targetName: 'y',
      key: s.key,
    },
  }
}

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'classifierOutputs',
    name: 'Miscalibrated classifier outputs',
    summary: 'Over- or under-confident logits with labels drawn from the true class probabilities.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(1, 100000, { default: 2000 }),
      classes: int(2, 20, { default: 3 }),
      separation: real(0, 5, { default: 1.5 }),
      temperature: real(0.1, 10, { default: 2.5 }),
      bias: real(-3, 3, { default: 0 }),
    }),
    truth: false,
    random: true,
    notes: ['temperature-scaling', 'classifier-calibration', 'conformal-prediction'],
    cite: ['guo2017'],
  },
  classifierOutputs,
)

dataset(
  {
    key: 'quantileModelOutputs',
    name: 'Quantile model outputs',
    summary: 'Heteroscedastic data with a model’s point prediction and an interval of the wrong width.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(1, 100000, { default: 1000 }),
      noise: real(0, 2, { default: 0.2 }),
      heteroscedastic: real(0, 10, { default: 3 }),
      alpha: real(0.01, 0.5, { default: 0.1 }),
      scale: real(0.1, 3, { default: 0.6 }),
      homoscedastic: real(0, 1, { default: 0 }),
    }),
    truth: false,
    random: true,
    notes: ['conformal-prediction', 'quantile-calibration'],
    cite: ['romano2019'],
  },
  quantileModelOutputs,
)
