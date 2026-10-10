/**
 * A small explainable boosting machine (Lou, Caruana and Gehrke, 2012, "Intelligible models for classification and
 * regression"; Nori et al., 2019, InterpretML): a GAM whose shape functions are step functions on binned features,
 * learned by cyclic gradient boosting of one-split trees with a small learning rate, one feature at a time. Regression
 * uses squared error; classification uses the logistic loss with Newton leaf values (Friedman, 2001). Unlike
 * interpret's `ExplainableBoostingRegressor`, the bins are of equal width over each feature's training range, there
 * are no pairwise interaction terms and no bagging.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { Bernoulli, Normal } from 'aifn-compute/probability/distributions'
import {
  withExpectation,
  withSampling,
  type Decides,
  type Distribution,
  type Estimator,
  type Expects,
  type Fitted,
  type Predicts,
  type Samples,
  type Supervised,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { sigmoid } from 'aifn-compute/numerics/special'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { ShapeError } from 'aifn-compute/foundation/errors'

type F64 = Float64Array

/** Hyperparameters of `explainableBoostingMachine`. */
export type EbmParams = {
  /** `'regression'` (default; squared error) or `'classification'` (logistic loss on 0/1 responses). */
  task?: 'regression' | 'classification'
  /** Equal-width bins per feature over the training range (default 32). */
  bins?: number
  /** Boosting rounds; each visits every feature once (default 1000). */
  rounds?: number
  /** Shrinkage $\nu$ of each tree (default 0.01). */
  learningRate?: number
  /**
   * Least total Hessian in a leaf, in points for squared error (default 2); for classification the least total
   * $p(1 - p)$ is 0.01 times this.
   */
  minLeaf?: number
}

/** A boosting state. */
export type EbmState = Status & {
  /** The intercept (the mean, or the log-odds of the mean for classification). */
  intercept: number
  /** Shape functions [d, bins]: the value added for each bin of each feature. */
  shapes: Tensor
  /** Mean training loss (squared error, or log loss) after this round. */
  loss: number
  /** Boosting rounds done. */
  t: number
}

/**
 * Binning of each feature: $\text{bin}(v) = \lfloor B (v - \text{lo}) / (\text{hi} - \text{lo}) \rfloor$ clamped
 * to $[0, B - 1]$, with `lo` and `hi` per feature ($d$ values each) and $B$ = `bins`.
 */
type Binning = { lo: F64; hi: F64; bins: number }

/**
 * The bin of a value of one feature; values outside the training range fall in the end bins.
 *
 * @param b The binning.
 * @param j The feature's index.
 * @param v The value.
 * @returns The bin, in $[0, B - 1]$.
 */
function binOf(b: Binning, j: number, v: number): number {
  const u = Math.floor(((v - b.lo[j]) / (b.hi[j] - b.lo[j])) * b.bins)
  return Math.min(b.bins - 1, Math.max(0, u))
}

/**
 * The EBM boosting loop as a traceable algorithm (see the module comment). Each round visits the features in turn: it
 * sums the loss's gradient $G$ and Hessian $H$ per bin of the feature, takes the split of the bins that maximises
 * $G_L^2/H_L + G_R^2/H_R$, and adds $-\nu G/H$ of each side to that side's bins. The intercept stays at the mean of
 * $\yvec$ (or its log-odds) and the shapes are not centred here; the estimator centres them.
 *
 * @param data The training features `x` ($n \times d$) and responses `y` ($n$; 0 or 1 for classification).
 * @param params The task, bins, learning rate and least leaf size (see `EbmParams`); `rounds` is not read here.
 * @returns The algorithm, with the `binning` it made from the training range.
 *
 * @example The training loss falls round by round
 * const r = stream(0)
 * const x = uniform(r, -1, 1, { shape: [100, 2] })
 * const y = tensor(toArray(x).map(([a, b]) => Math.sin(3 * a) + b * b))
 * const alg = ebmBoosting({ x, y }, { bins: 8, learningRate: 0.1 })
 * for (const k of [0, 10, 100]) print('round', k, 'loss =', run(alg, undefined, k).loss)
 */
export function ebmBoosting(
  data: Supervised<Tensor, Tensor>,
  params: EbmParams = {},
): Algorithm<void, EbmState> & { binning: Binning } {
  const { task = 'regression', bins = 32, learningRate = 0.01, minLeaf = 2 } = params
  const [n, d] = data.x.shape
  const X = toFlat(data.x)
  const y = Float64Array.from(toFlat(data.y))
  const lo = new Float64Array(d).fill(Infinity)
  const hi = new Float64Array(d).fill(-Infinity)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) {
      lo[j] = Math.min(lo[j], X[i * d + j])
      hi[j] = Math.max(hi[j], X[i * d + j])
    }
  for (let j = 0; j < d; j++) if (!(hi[j] > lo[j])) hi[j] = lo[j] + 1
  const binning: Binning = { lo, hi, bins }
  const binned = new Int32Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) binned[i * d + j] = binOf(binning, j, X[i * d + j])
  const classify = task === 'classification'
  const mean = y.reduce((a, b) => a + b, 0) / n
  const intercept = classify ? Math.log(mean / (1 - mean)) : mean
  const predict = (shapes: F64) => {
    const f = new Float64Array(n).fill(intercept)
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) f[i] += shapes[j * bins + binned[i * d + j]]
    return f
  }
  const lossOf = (f: F64) => {
    let s = 0
    for (let i = 0; i < n; i++) {
      if (classify) s += Math.log1p(Math.exp(-Math.abs(f[i]))) + Math.max(f[i], 0) - y[i] * f[i]
      else s += (y[i] - f[i]) ** 2
    }
    return s / n
  }
  return {
    name: 'ebm-boosting',
    binning,
    init: () => {
      const shapes = new Float64Array(d * bins)
      return { t: 0, intercept, shapes: fromData(shapes, [d, bins]), loss: lossOf(predict(shapes)) }
    },
    step: (state) => {
      const shapes = Float64Array.from(toFlat(state.shapes))
      const f = predict(shapes)
      for (let j = 0; j < d; j++) {
        // Gradient and Hessian of the loss per bin.
        const G = new Float64Array(bins)
        const H = new Float64Array(bins)
        for (let i = 0; i < n; i++) {
          const b = binned[i * d + j]
          if (classify) {
            const p = sigmoid(f[i])
            G[b] += p - y[i]
            H[b] += p * (1 - p)
          } else {
            G[b] += f[i] - y[i]
            H[b] += 1
          }
        }
        let gTotal = 0
        let hTotal = 0
        for (let b = 0; b < bins; b++) {
          gTotal += G[b]
          hTotal += H[b]
        }
        // The best single split maximises G_L²/H_L + G_R²/H_R (the loss reduction of a two-leaf Newton step).
        let best = -Infinity
        let cut = -1
        let gl = 0
        let hl = 0
        let gLeft = 0
        let hLeft = 0
        for (let b = 0; b < bins - 1; b++) {
          gl += G[b]
          hl += H[b]
          const hr = hTotal - hl
          if (hl < minLeaf * (classify ? 0.01 : 1) || hr < minLeaf * (classify ? 0.01 : 1)) continue
          const gain = (gl * gl) / hl + ((gTotal - gl) * (gTotal - gl)) / hr
          if (gain > best) [best, cut, gLeft, hLeft] = [gain, b, gl, hl]
        }
        if (cut < 0) continue
        const left = (-learningRate * gLeft) / hLeft
        const right = (-learningRate * (gTotal - gLeft)) / (hTotal - hLeft)
        for (let b = 0; b < bins; b++) shapes[j * bins + b] += b <= cut ? left : right
        for (let i = 0; i < n; i++) f[i] += binned[i * d + j] <= cut ? left : right
      }
      const loss = lossOf(f)
      return { t: state.t + 1, intercept, shapes: fromData(shapes, [d, bins]), loss, diverged: !Number.isFinite(loss) }
    },
  }
}

/** A fitted explainable boosting machine. */
export interface EbmModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Distribution>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Trained<EbmState> {
  /** Tags a fitted model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'explainable-boosting-machine'
  /** The task it was fitted for. */
  readonly task: 'regression' | 'classification'
  /** The intercept after centring the shapes. */
  readonly intercept: number
  /**
   * Shape functions ($d \times$ `bins`), each centred to mean zero over the training points (the intercept absorbs
   * it).
   */
  readonly shapes: Tensor
  /** Bin edges per feature ($d \times (\text{bins} + 1)$). */
  readonly edges: Tensor
  /** Root-mean-square training residual (regression; computed on the 0/1 scale for classification). */
  readonly noiseSd: number
}

/**
 * An explainable boosting machine: `forward` gives the additive score (the prediction for regression, the logit for
 * classification), `decide` the prediction or class, `predictive` a normal with the training residual spread or a
 * Bernoulli, and the boosting run is kept in `training` (every $\max(1, \lfloor \text{rounds}/200 \rfloor)$-th
 * round unless the fit's `trace.every` says otherwise). After boosting, each shape is centred over the training points
 * and the intercept takes the means, as in interpret.
 *
 * @param params The task, bins, rounds, learning rate and least leaf size (see `EbmParams`).
 * @returns The estimator; its `fit` takes features `x` ($n \times d$) and responses `y` ($n$).
 *
 * @example Each feature's shape: a sine of the first, a parabola of the second
 * const r = stream(0)
 * const x = uniform(r, -1, 1, { shape: [100, 2] })
 * const y = tensor(toArray(x).map(([a, b]) => Math.sin(3 * a) + b * b))
 * const model = explainableBoostingMachine({ bins: 8, rounds: 200, learningRate: 0.1 }).fit({ x, y })
 * const edges = toArray(model.edges)[0]
 * print('bin centres =', edges.slice(0, 8).map((e, b) => (e + edges[b + 1]) / 2))
 * print('shape of x0 =', toArray(model.shapes)[0])
 * print('shape of x1 =', toArray(model.shapes)[1])
 * print('intercept =', model.intercept, 'residual sd =', model.noiseSd)
 */
export function explainableBoostingMachine(params: EbmParams = {}): Estimator<Supervised<Tensor, Tensor>, EbmModel> {
  const { rounds = 1000, bins = 32, task = 'regression' } = params
  return {
    name: 'explainable-boosting-machine',
    params,
    fit(data, options = {}) {
      const alg = ebmBoosting(data, params)
      const training: Trace<EbmState> = trace(alg, undefined, rounds, {
        every: options.trace?.every ?? Math.max(1, Math.floor(rounds / 200)),
        record: { loss: (s) => s.loss },
      })
      const final = training.final
      const [n, d] = data.x.shape
      const X = toFlat(data.x)
      const raw = Float64Array.from(toFlat(final.shapes))
      // Centre each shape over the training points.
      let intercept = final.intercept
      for (let j = 0; j < d; j++) {
        let m = 0
        for (let i = 0; i < n; i++) m += raw[j * bins + binOf(alg.binning, j, X[i * d + j])] / n
        for (let b = 0; b < bins; b++) raw[j * bins + b] -= m
        intercept += m
      }
      const forward = (x: Tensor) => {
        const [m, cols] = x.shape
        if (cols !== d) throw new ShapeError('ebm', `ebm: fitted on ${d} features, given ${cols}`)
        const v = toFlat(x)
        const out = new Float64Array(m).fill(intercept)
        for (let i = 0; i < m; i++)
          for (let j = 0; j < d; j++) out[i] += raw[j * bins + binOf(alg.binning, j, v[i * d + j])]
        return fromData(out, [m])
      }
      const y = toFlat(data.y)
      const train = toFlat(forward(data.x))
      const noiseSd = Math.sqrt(train.reduce((s, f, i) => s + (y[i] - f) ** 2, 0) / n)
      const edges = new Float64Array(d * (bins + 1))
      for (let j = 0; j < d; j++)
        for (let b = 0; b <= bins; b++)
          edges[j * (bins + 1) + b] = alg.binning.lo[j] + ((alg.binning.hi[j] - alg.binning.lo[j]) * b) / bins
      const prob = (x: Tensor) =>
        fromData(
          Float64Array.from(toFlat(forward(x)), (f) => sigmoid(f)),
          [x.shape[0]],
        )
      const base = {
        kind: 'model' as const,
        name: 'explainable-boosting-machine' as const,
        task,
        intercept,
        shapes: fromData(raw, [d, bins]),
        edges: fromData(edges, [d, bins + 1]),
        noiseSd,
        training,
        forward,
        decide: (x: Tensor) =>
          task === 'classification'
            ? fromData(
                Int32Array.from(toFlat(forward(x)), (f) => (f > 0 ? 1 : 0)),
                [x.shape[0]],
              )
            : forward(x),
        predictive: (x: Tensor): Distribution =>
          task === 'classification'
            ? Bernoulli(prob(x))
            : Normal(forward(x), fromData(new Float64Array(x.shape[0]).fill(noiseSd), [x.shape[0]])),
      }
      return withSampling(withExpectation(base))
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'explainableBoostingMachine',
    module: 'learning/generalised/gam',
    name: 'Explainable boosting machine',
    summary: 'A GAM of binned shape functions learned by cyclic gradient boosting.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'sample'],
    hyper: space({
      task: oneOf(['regression', 'classification']),
      bins: int(2, 256, { default: 32 }),
      rounds: int(1, 5000, { default: 1000 }),
      learningRate: real(1e-3, 1, { default: 0.01, scale: 'log' }),
      minLeaf: int(1, 100, { default: 2 }),
    }),
    notes: ['explainable-boosting-machines'],
    cite: ['lou2012', 'nori2019'],
  },
  explainableBoostingMachine,
)
