/**
 * Deep ordinal regression: an ordinal output layer on a multilayer perceptron (`aifn-compute/nn`'s `Mlp`, trained by its
 * `trainingLoop` with Adam). The network maps x to one score g(x) (its last layer has no bias), and the head turns g
 * into ordered classes:
 *
 * - `coral` (Cao, Mirjalili & Raschka, 2020, "Rank consistent ordinal regression for neural networks", Pattern
 *   Recognition Letters 140): K − 1 binary units P(y > k | x) = σ(g(x) + b_k) sharing g and with their own biases,
 *   each trained by binary cross-entropy on 1[y > k]. The estimates decrease in k whenever the biases do, which holds
 *   at the optimum; probabilities are their differences (clipped, as `binaryDecomposition` repairs them).
 * - `cumulative`: the cumulative-link (proportional-odds) model with η = g(x), the negative log-likelihood of
 *   `aifn-compute/probability/likelihoods`' `ordinalLikelihood`, thresholds kept increasing by `orderedBijector`. With no
 *   hidden layer this is `ordinalRegression` fitted by gradient descent.
 */

import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream as rootStream } from 'aifn-compute/foundation/random'
import {
  add,
  dense,
  expandDims,
  fromData,
  mean,
  mul,
  neg,
  reshape,
  sub,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace, type Trace } from 'aifn-compute/foundation/trace'
import {
  categoricalPredictive,
  defineModel,
  matrixShape,
  targetValues,
  withExpectation,
  withSampling,
  type AnyUnivariate,
  type Decides,
  type Estimator,
  type Expects,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Samples,
  type Scores,
  type Supervised,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { Linear, Mlp, Sequential, type Layer } from 'aifn-compute/nn/layers'
import type { Activation } from 'aifn-compute/nn/functional'
import { trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import { logSigmoid } from 'aifn-compute/numerics/special'
import { adamRule } from 'aifn-compute/optim/first-order'
import { orderedBijector } from 'aifn-compute/probability/bijectors'
import { ordinalLikelihood } from 'aifn-compute/probability/likelihoods'
import { differenceExceedance } from './decomposition'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The ordinal head of `deepOrdinalRegression`. */
export type DeepOrdinalHead = 'coral' | 'cumulative'

/** Hyperparameters of `deepOrdinalRegression`. */
export type DeepOrdinalRegressionParams = {
  /** The head (default `coral`). */
  head?: DeepOrdinalHead
  /** Hidden layer sizes (default [16]); [] is a linear score. */
  hidden?: readonly number[]
  /** Hidden activation (default `relu`). */
  activation?: Activation
  /** Adam step size (default 0.01). */
  learningRate?: number
  /** Optimiser steps (default 500). */
  steps?: number
  /** Examples per step (default: the whole set). */
  batchSize?: number
  /** Number of classes K (default: the largest label + 1). */
  classes?: number
}

/** The trainable parameters: the network's and the head's K − 1 biases (CORAL) or threshold coordinates. */
type DeepOrdinalParams = { net: Params[]; head: Tensor }

/** A fitted deep ordinal model. */
export interface DeepOrdinalRegressionModel
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Trained<TrainingState<DeepOrdinalParams>> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'deep-ordinal-regression'
  readonly head: DeepOrdinalHead
  readonly classes: number
  readonly params: DeepOrdinalParams
  /** CORAL's biases b [K − 1], or the cumulative model's increasing thresholds θ [K − 1]. */
  readonly cutpoints: Tensor
  /** P(y > k | x) [m, K − 1]. */
  exceedance(x: Tensor): Tensor
  /** Class probabilities P(y = k | x) [m, K]. */
  probabilities(x: Tensor): Tensor
}

/** The score network: an MLP to the last hidden layer, then a bias-free linear map to one unit. */
function scoreNetwork(d: number, hidden: readonly number[], activation: Activation): Layer<Params[]> {
  const last = hidden.length ? hidden[hidden.length - 1] : d
  const body = hidden.length ? [Mlp([d, ...hidden], { activation, outputActivation: activation })] : []
  return Sequential(...body, Linear(last, 1, { bias: false }) as Layer<Params>) as Layer<Params[]>
}

/**
 * Deep ordinal regression (see the module comment): labels are class indices 0 … K − 1. Capabilities: `forward` and
 * `score` (the network's score g(x), larger for higher classes), `predictive` (categorical over the K classes),
 * `decide` (the most probable class), `expect` (E[y]), `sample`. Initialisation and minibatches draw from the fit's
 * stream (default `stream('deep-ordinal')`); the run is kept in `training`.
 */
export function deepOrdinalRegression(
  params: DeepOrdinalRegressionParams = {},
): Estimator<Supervised<Tensor, Tensor>, DeepOrdinalRegressionModel> {
  const { head = 'coral', hidden = [16], activation = 'relu', learningRate = 0.01, steps = 500 } = params
  const likelihood = ordinalLikelihood('cumulative', 'logit')
  const ordered = orderedBijector()
  return {
    name: 'deep-ordinal-regression',
    params,
    fit({ x, y }, options: FitOptions = {}) {
      const [n, d] = matrixShape(x, 'deepOrdinalRegression')
      const t = targetValues(y, 'deepOrdinalRegression')
      if (t.length !== n)
        throw new ShapeError('deepOrdinalRegression', `deepOrdinalRegression: ${n} inputs but ${t.length} labels`)
      if (!t.every((v) => Number.isInteger(v) && v >= 0))
        throw new DomainError('deepOrdinalRegression', 'deepOrdinalRegression: labels must be class indices 0, 1, …')
      const K = params.classes ?? Math.max(...t) + 1
      if (K < 2) throw new DomainError('deepOrdinalRegression', 'deepOrdinalRegression: needs at least two classes')
      const m = K - 1
      const s = options.stream ?? rootStream('deep-ordinal')
      const net = scoreNetwork(d, hidden, activation)
      const score = (p: DeepOrdinalParams, input: Value): Value => reshape(net.apply(p.net, input, {}), [-1])
      // CORAL starts from decreasing biases, the cumulative head from increasing thresholds, both spread on [−1, 1].
      const spread = Float64Array.from({ length: m }, (_, k) => (m === 1 ? 0 : -1 + (2 * k) / (m - 1)))
      const head0 =
        head === 'coral'
          ? fromData(spread.reverse(), [m])
          : (ordered.inverse(fromData(Float64Array.from(spread).sort(), [m])) as Tensor)
      const init: DeepOrdinalParams = { net: net.init(child(s, 'init')), head: head0 }
      // Constant targets 1[y > k] [n, K − 1] for CORAL.
      const exceeds = new Float64Array(n * m)
      for (let i = 0; i < n; i++) for (let k = 0; k < m; k++) exceeds[i * m + k] = t[i] > k ? 1 : 0
      const data = {
        x: fromData(Float64Array.from(dense.data(x)), [n, d]),
        y: fromData(Int32Array.from(t), [n]),
        exceeds: fromData(exceeds, [n, m]),
      }
      const loss = (p: DeepOrdinalParams, batch: typeof data): Value => {
        const g = score(p, batch.x)
        if (head === 'coral') {
          // Σ_k −[t log σ(z) + (1 − t) log σ(−z)], z = g + b_k, averaged over the batch.
          const z = add(expandDims(g, -1), p.head)
          const ll = add(mul(batch.exceeds, logSigmoid(z)), mul(sub(1, batch.exceeds), logSigmoid(neg(z))))
          return neg(mean(sum(ll, -1)))
        }
        return neg(mean(likelihood.logLik(batch.y, g, ordered.forward(p.head))))
      }
      const training: Trace<TrainingState<DeepOrdinalParams>> = trace(
        trainingLoop<DeepOrdinalParams, typeof data>({
          data,
          loss,
          batchSize: params.batchSize,
          optimizer: adamRule({ stepSize: learningRate }),
        }),
        { params: init },
        steps,
        {
          stream: child(s, 'train'),
          every: options.trace?.every ?? 1,
          keep: 'none',
          record: { loss: (st) => st.loss },
        },
      )
      const fitted = training.final.params
      const cutpoints =
        head === 'coral'
          ? fromData(Float64Array.from(toFlat(fitted.head)), [m])
          : fromData(Float64Array.from(toFlat(ordered.forward(fitted.head) as Tensor)), [m])
      const cuts = toFlat(cutpoints)
      const forward = (input: Tensor): Tensor => {
        const [, cols] = matrixShape(input, 'deepOrdinalRegression.forward')
        if (cols !== d)
          throw new ShapeError('deepOrdinalRegression', `deepOrdinalRegression: fitted on ${d} features, given ${cols}`)
        return unwrap(score(fitted, input)) as Tensor
      }
      const exceedance = (input: Tensor): Tensor => {
        const g = toFlat(forward(input))
        const out = new Float64Array(g.length * m)
        g.forEach((gi, i) => {
          for (let k = 0; k < m; k++) {
            // CORAL: σ(g + b_k); cumulative: P(y > k) = 1 − F(θ_k − g) = σ(g − θ_k).
            const z = head === 'coral' ? gi + cuts[k] : gi - cuts[k]
            out[i * m + k] = Math.exp(logSigmoidNumber(z))
          }
        })
        return fromData(out, [g.length, m])
      }
      const probabilities = (input: Tensor): Tensor => {
        const q = exceedance(input)
        const rows = q.shape[0]
        return fromData(differenceExceedance(toFlat(q), rows, K, 'clip'), [rows, K])
      }
      const decide = (input: Tensor): Tensor => {
        const P = toFlat(probabilities(input))
        const rows = P.length / K
        const out = new Int32Array(rows)
        for (let i = 0; i < rows; i++) for (let k = 1; k < K; k++) if (P[i * K + k] > P[i * K + out[i]]) out[i] = k
        return fromData(out, [rows])
      }
      const base = {
        kind: 'model' as const,
        name: 'deep-ordinal-regression' as const,
        head,
        classes: K,
        params: fitted,
        cutpoints,
        training,
        forward,
        score: forward,
        exceedance,
        probabilities,
        decide,
        predictive: (input: Tensor): AnyUnivariate => categoricalPredictive(probabilities(input)),
      }
      return withSampling(withExpectation(base))
    },
  }
}

/** log σ(z) for a number. */
const logSigmoidNumber = (z: number): number => logSigmoid(z) as number

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'deepOrdinalRegression',
    module: 'learning/generalised/ordinal',
    name: 'Deep ordinal regression',
    summary: 'An MLP score with a CORAL or cumulative-link ordinal head, trained by Adam.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'score', 'sample'],
    random: true,
    hyper: space({
      head: oneOf(['coral', 'cumulative']),
      learningRate: real(1e-4, 1, { default: 0.01, scale: 'log' }),
      steps: int(1, 10000, { default: 500 }),
    }),
    notes: ['deep-ordinal-regression', 'ordinal-regression'],
    cite: ['cao2020coral'],
  },
  deepOrdinalRegression,
)
