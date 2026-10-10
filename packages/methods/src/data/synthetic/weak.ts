/**
 * Data for weak supervision with known ground truth: labelling functions with known accuracies and coverages, crowd
 * workers with known confusion matrices, positive–unlabelled samples with a known label frequency, bags with known
 * class proportions, multiple-instance bags with known witnesses, and complementary labels. Features are 2-d Gaussian
 * blobs of unit standard deviation, one per class on a regular polygon (`blobs` with the `polygon` layout, or the same
 * placement drawn directly), so the true labels are learnable from $\xvec$. Every generator returns the true labels
 * beside the weak ones, and the parameters it drew (accuracies, confusions, proportions), so that an estimate can be
 * checked against them.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { child, integers, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkCount, labels, matrix, type DatasetMeta } from '../types'
import { blobs } from './points'

/**
 * Features and true labels from `classes` Gaussian blobs of unit standard deviation on a polygon, `separation`
 * standard deviations apart, grouped by class with equal class sizes (the first `n mod classes` one larger).
 *
 * @param s The random stream passed to `blobs`.
 * @param n The number of points.
 * @param classes The number of classes, one blob each.
 * @param separation The distance between neighbouring blob centres, in standard deviations.
 * @returns `x`, the points ($n \times 2$), and `y`, the true class of each.
 */
function features(s: Stream, n: number, classes: number, separation: number) {
  const d = blobs(s, { n, centers: classes, separation, layout: 'polygon', sd: 1 })
  return { x: d.x, y: Int32Array.from(toFlat(d.y!)) }
}

/**
 * A draw uniform on an interval.
 *
 * @param s The random stream; one uniform is drawn from it.
 * @param range The interval as `[lo, hi]`.
 * @returns A value uniform on $[\mathrm{lo}, \mathrm{hi})$.
 */
const between = (s: Stream, [lo, hi]: readonly [number, number]) => lo + (hi - lo) * uniform(s)

/** Votes of labelling functions with their true accuracies and coverages. */
export interface LabellingFunctions {
  /** Always `'dataset'`. */
  kind: 'dataset'
  /** The features, $n \times 2$, grouped by class. */
  x: Tensor
  /** True labels (int32), unseen by the label models. */
  y: Tensor
  /** Votes, $n \times m$ (int32): a class, or $-1$ for an abstention. */
  votes: Tensor
  /** Each function's accuracy $\alpha_j = P(\text{vote} = y \mid \text{voted})$, $m$ values. */
  accuracy: Tensor
  /** Each function's coverage $\beta_j = P(\text{voted})$, $m$ values. */
  coverage: Tensor
  /** The description and the stream key. */
  meta: DatasetMeta
}

/** Options of `labellingFunctions`. */
export interface LabellingFunctionOptions {
  /** Number of examples (default 400). */
  n?: number
  /** Number of classes $K$ (default 2). */
  classes?: number
  /** Labelling functions, not counting the copies (default 8). */
  functions?: number
  /** Each function's accuracy is drawn uniformly from this range (default `[0.55, 0.9]`). */
  accuracy?: readonly [number, number]
  /** Each function's coverage is drawn uniformly from this range (default `[0.2, 0.8]`). */
  coverage?: readonly [number, number]
  /** Extra copies of function 0 that repeat its votes (breaking conditional independence; default 0). */
  copies?: number
  /** Class separation of the features in standard deviations (default 3). */
  separation?: number
}

/**
 * Labelling functions in the data-programming model (Ratner et al., 2016): function $j$ votes on an example with
 * probability $\beta_j$ (its coverage) and, when it votes, names the true class with probability $\alpha_j$ and each
 * other class with probability $(1 - \alpha_j)/(K - 1)$. The accuracies and coverages are drawn per function and
 * returned, so a label model's estimates can be compared with the truth. `copies` duplicates function 0: the copies,
 * the last columns of `votes`, repeat its votes exactly, which double-counts it for any model that assumes independent
 * functions. Throws `DomainError` unless `n` is a non-negative integer.
 *
 * @param s The random stream: the features come from its child `x`, function $j$'s accuracy and coverage from
 *   `accuracy` $j$ and `coverage` $j$, its vote on example $i$ from `vote` $i$ $j$.
 * @param options The sizes, the accuracy and coverage ranges, the copies and the feature separation; see
 *   `LabellingFunctionOptions`.
 * @returns The features, true labels, the $n \times m$ votes ($m$ the functions plus the copies) and each function's
 *   accuracy and coverage.
 *
 * @example Votes and the accuracies behind them
 * const lf = labellingFunctions(stream(0), { n: 1000, functions: 4 })
 * print('x:', lf.x.shape, ' votes:', lf.votes.shape, ' first rows of votes:', toArray(lf.votes).slice(0, 3))
 * print('accuracy:', lf.accuracy)
 * // Function 0's share of votes cast, and the share of those that name the true class.
 * const y = toArray(lf.y)
 * const cast = toArray(lf.votes).map((row, i) => [row[0], y[i]]).filter(([v]) => v >= 0)
 * print('function 0 coverage:', cast.length / 1000, ' true:', toArray(lf.coverage)[0])
 * print('function 0 accuracy:', cast.filter(([v, t]) => v === t).length / cast.length)
 */
export function labellingFunctions(s: Stream, options: LabellingFunctionOptions = {}): LabellingFunctions {
  const {
    n = 400,
    classes = 2,
    functions = 8,
    accuracy: accRange = [0.55, 0.9],
    coverage: covRange = [0.2, 0.8],
    copies = 0,
    separation = 3,
  } = options
  checkCount(n, 'labellingFunctions')
  const { x, y } = features(child(s, 'x'), n, classes, separation)
  const m = functions + copies
  const acc = new Float64Array(m)
  const cov = new Float64Array(m)
  for (let j = 0; j < functions; j++) {
    acc[j] = between(child(s, 'accuracy', j), accRange)
    cov[j] = between(child(s, 'coverage', j), covRange)
  }
  for (let c = 0; c < copies; c++) {
    acc[functions + c] = acc[0]
    cov[functions + c] = cov[0]
  }
  const votes = new Int32Array(n * m)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      // A copy reads the stream of function 0, so it repeats its votes exactly.
      const r = child(s, 'vote', i, j < functions ? j : 0)
      if (uniform(child(r, 'cover')) >= cov[j]) votes[i * m + j] = -1
      else if (uniform(child(r, 'right')) < acc[j]) votes[i * m + j] = y[i]
      else {
        const wrong = integers(child(r, 'wrong'), classes - 1)
        votes[i * m + j] = wrong >= y[i] ? wrong + 1 : wrong
      }
    }
  return {
    kind: 'dataset',
    x,
    y: labels(y),
    votes: fromData(votes, [n, m]),
    accuracy: fromData(acc, [m]),
    coverage: fromData(cov, [m]),
    meta: {
      name: 'labelling functions',
      description: `${n} examples, ${classes} classes, ${functions} labelling functions${copies ? ` plus ${copies} copies of the first` : ''}.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      key: s.key,
    },
  }
}

/** Crowd labels with the workers' true confusion matrices. */
export interface CrowdLabels {
  /** The true class of each item (int32). */
  y: Tensor
  /** Votes, $n \times$ workers (int32), $-1$ where a worker did not label the item. */
  votes: Tensor
  /**
   * The confusion matrices, workers $\times K \times K$: entry $(w, k, l)$ is the probability that worker $w$ says
   * $l$ of an item of class $k$.
   */
  confusions: Tensor
  /** The description and the stream key. */
  meta: DatasetMeta
}

/** Options of `crowdLabels`. */
export interface CrowdLabelOptions {
  /** Number of items (default 200). */
  n?: number
  /** Number of classes $K$ (default 3). */
  classes?: number
  /** Number of workers (default 10). */
  workers?: number
  /** Workers per item, at most `workers` (default 3). */
  perItem?: number
  /** Each worker's accuracy is drawn from this range (default `[0.35, 0.95]`); errors favour one confusable class. */
  skill?: readonly [number, number]
}

/**
 * Crowdsourced labels in the Dawid–Skene model (Dawid and Skene, 1979): each item's class is uniform over the $K$
 * classes, and it is labelled by `perItem` workers drawn without replacement; worker $w$ answers from row $y$ of its
 * confusion matrix, whose diagonal is its skill $a_w$ and whose errors go to one confusable class per true class with
 * probability $0.7(1 - a_w)$ and are otherwise spread uniformly over the other classes (with two classes every error
 * names the other class). Throws `DomainError` unless `n` is a non-negative integer.
 *
 * @param s The random stream: worker $w$'s skill comes from its child `skill` $w$, the confusable classes from
 *   `confusable`, item $i$'s class from `y` $i$, its workers from `assign` and their answers from `answer`.
 * @param options The numbers of items, classes, workers and workers per item, and the skill range; see
 *   `CrowdLabelOptions`.
 * @returns The true classes, the votes ($-1$ where a worker did not label an item) and the true confusion matrices.
 *
 * @example Votes and a worker's confusion matrix
 * const c = crowdLabels(stream(0), { n: 600 })
 * print('votes:', c.votes.shape, ' first row:', toArray(c.votes)[0], ' its class:', toArray(c.y)[0])
 * print('labels per item:', toArray(c.votes)[0].filter((v) => v >= 0).length)
 * print('worker 0 confusion:', toArray(c.confusions)[0])
 * // Worker 0's share of correct answers, against the diagonal of its matrix.
 * const y = toArray(c.y)
 * const answers = toArray(c.votes).map((row, i) => [row[0], y[i]]).filter(([v]) => v >= 0)
 * print('worker 0 accuracy:', answers.filter(([v, t]) => v === t).length / answers.length)
 */
export function crowdLabels(s: Stream, options: CrowdLabelOptions = {}): CrowdLabels {
  const { n = 200, classes: K = 3, workers = 10, perItem = 3, skill = [0.35, 0.95] } = options
  checkCount(n, 'crowdLabels')
  const conf = new Float64Array(workers * K * K)
  for (let w = 0; w < workers; w++) {
    const a = between(child(s, 'skill', w), skill)
    for (let k = 0; k < K; k++) {
      const confusable = (k + 1 + integers(child(s, 'confusable', w, k), Math.max(1, K - 1))) % K
      for (let l = 0; l < K; l++) {
        const o = (w * K + k) * K + l
        if (l === k) conf[o] = a
        else if (K === 2) conf[o] = 1 - a
        else conf[o] = (1 - a) * ((l === confusable ? 0.7 : 0) + 0.3 / (K - 1))
      }
    }
  }
  const y = Int32Array.from({ length: n }, (_, i) => integers(child(s, 'y', i), K))
  const votes = new Int32Array(n * workers).fill(-1)
  for (let i = 0; i < n; i++) {
    const pool = Array.from({ length: workers }, (_, w) => w)
    for (let r = 0; r < Math.min(perItem, workers); r++) {
      const pick = integers(child(s, 'assign', i, r), pool.length)
      const w = pool.splice(pick, 1)[0]
      let u = uniform(child(s, 'answer', i, w))
      let l = 0
      for (; l < K - 1; l++) {
        u -= conf[(w * K + y[i]) * K + l]
        if (u < 0) break
      }
      votes[i * workers + w] = l
    }
  }
  return {
    y: labels(y),
    votes: fromData(votes, [n, workers]),
    confusions: fromData(conf, [workers, K, K]),
    meta: {
      name: 'crowd labels',
      description: `${n} items, ${K} classes, ${workers} workers, ${perItem} labels per item.`,
      task: 'classification',
      featureNames: [],
      key: s.key,
    },
  }
}

/** A positive–unlabelled sample. */
export interface PositiveUnlabelled {
  /** Always `'dataset'`. */
  kind: 'dataset'
  /** The features, $n \times 2$, the negatives first. */
  x: Tensor
  /** True classes (int32, 1 positive). */
  y: Tensor
  /** The observed labels $s$ (int32, 1 labelled positive, 0 unlabelled). */
  labelled: Tensor
  /** The true class prior $\pi$, the share of positives. */
  prior: number
  /** The label frequency $c = P(s = 1 \mid y = 1)$. */
  labelFrequency: number
  /** The description and the stream key. */
  meta: DatasetMeta
}

/**
 * Positive and unlabelled data under "selected completely at random": points from two Gaussian classes, exactly
 * $\mathrm{round}(n\pi)$ of them positive; each positive is labelled with probability $c$ independently of $\xvec$,
 * and every other point is unlabelled. Throws `DomainError` unless `n` is a non-negative integer.
 *
 * @param s The random stream: the features come from its child `x`, point $i$'s labelling from `label` $i$.
 * @param options `n`, the number of points (default 400); `prior`, the positive share $\pi$ (default 0.4);
 *   `labelFrequency`, the probability $c$ that a positive is labelled (default 0.3); `separation`, the distance
 *   between the class centres in standard deviations (default 3).
 * @returns The features, true classes, observed labels, and the prior and label frequency.
 *
 * @example Only positives are labelled
 * const pu = positiveUnlabelled(stream(0), { n: 2000 })
 * const y = toArray(pu.y)
 * const s = toArray(pu.labelled)
 * const positives = y.filter((v) => v === 1).length
 * const labelled = s.filter((v) => v === 1).length
 * print('x:', pu.x.shape, ' positives:', positives, ' labelled:', labelled)
 * print('labelled negatives:', s.filter((v, i) => v === 1 && y[i] === 0).length)
 * print('labelled share of positives:', labelled / positives, ' c:', pu.labelFrequency)
 */
export function positiveUnlabelled(
  s: Stream,
  options: { n?: number; prior?: number; labelFrequency?: number; separation?: number } = {},
): PositiveUnlabelled {
  const { n = 400, prior = 0.4, labelFrequency = 0.3, separation = 3 } = options
  checkCount(n, 'positiveUnlabelled')
  const pos = Math.round(n * prior)
  const d = blobs(child(s, 'x'), { n: [n - pos, pos], centers: 2, separation, layout: 'polygon', sd: 1 })
  const y = Int32Array.from(toFlat(d.y!))
  const lab = Int32Array.from(y, (v, i) => (v === 1 && uniform(child(s, 'label', i)) < labelFrequency ? 1 : 0))
  return {
    kind: 'dataset',
    x: d.x,
    y: labels(y),
    labelled: labels(lab),
    prior,
    labelFrequency,
    meta: {
      name: 'positive–unlabelled',
      description: `${n} points, positive prior ${prior}, each positive labelled with probability ${labelFrequency}.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      key: s.key,
    },
  }
}

/** Bags labelled by their class proportions. */
export interface ProportionBags {
  /** Always `'dataset'`. */
  kind: 'dataset'
  /** The features, $n \times 2$, bag by bag. */
  x: Tensor
  /** The true class of each instance (int32), hidden from a learner from proportions. */
  y: Tensor
  /** Each instance's bag (int32). */
  bag: Tensor
  /** Each bag's realised class proportions, $B \times K$. */
  proportions: Tensor
  /** The description and the stream key. */
  meta: DatasetMeta
}

/**
 * Bags for learning from label proportions: bag $b$ draws a class mixture from a symmetric Dirichlet with parameter
 * $\alpha$ = `concentration`, then `bagSize` labels from it and each instance's features from its class's Gaussian
 * (unit standard deviation, the centres on a polygon `separation` apart); only the bag's realised class proportions are
 * revealed. The Dirichlet is drawn as normalised Gamma draws, each a sum of $\max(1, \mathrm{round}(\alpha))$
 * exponentials, so a non-integer $\alpha$ is rounded and one below 1.5 acts as $\alpha = 1$. Throws `DomainError`
 * unless `bags` is a non-negative integer.
 *
 * @param s The random stream: bag $b$'s mixture comes from its child `mix` $b$, instance $i$'s class from `label` $i$
 *   and its features from `x` $i$.
 * @param options `bags`, the number of bags $B$ (default 20); `bagSize`, the instances per bag (default 20);
 *   `classes`, the number of classes $K$ (default 2); `concentration`, the Dirichlet parameter $\alpha$ (default 1);
 *   `separation`, the distance between neighbouring class centres (default 3).
 * @returns The features, true classes, each instance's bag and each bag's class proportions.
 *
 * @example Bags and their proportions
 * const b = proportionBags(stream(0), { bags: 4, bagSize: 10 })
 * print('x:', b.x.shape, ' bags of the first instances:', toArray(b.bag).slice(0, 12))
 * print('proportions:', b.proportions)
 * // The proportions are the bags' class shares: bag 0's share of class 1.
 * print('bag 0, class 1:', toArray(b.y).slice(0, 10).filter((v) => v === 1).length / 10)
 */
export function proportionBags(
  s: Stream,
  options: { bags?: number; bagSize?: number; classes?: number; concentration?: number; separation?: number } = {},
): ProportionBags {
  const { bags = 20, bagSize = 20, classes: K = 2, concentration = 1, separation = 3 } = options
  checkCount(bags, 'proportionBags')
  const n = bags * bagSize
  // Class centres on a polygon, as `blobs` places them.
  const centres = Array.from({ length: K }, (_, k) => {
    const r = K === 2 ? separation / 2 : separation / (2 * Math.sin(Math.PI / K))
    const t = (2 * Math.PI * k) / K
    return [r * Math.cos(t), r * Math.sin(t)]
  })
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  const bag = new Int32Array(n)
  const props = new Float64Array(bags * K)
  for (let b = 0; b < bags; b++) {
    // Dirichlet(α) through normalised Gamma draws; α = 1 by −log U.
    const g = Array.from({ length: K }, (_, k) => {
      if (concentration === 1) return -Math.log(1 - uniform(child(s, 'mix', b, k)))
      let sum = 0
      for (let t = 0; t < Math.max(1, Math.round(concentration)); t++)
        sum -= Math.log(1 - uniform(child(s, 'mix', b, k, t)))
      return sum
    })
    const total = g.reduce((a, v) => a + v, 0)
    for (let r = 0; r < bagSize; r++) {
      const i = b * bagSize + r
      let u = uniform(child(s, 'label', i)) * total
      let k = 0
      for (; k < K - 1; k++) {
        u -= g[k]
        if (u < 0) break
      }
      y[i] = k
      bag[i] = b
      props[b * K + k] += 1 / bagSize
      x[2 * i] = centres[k][0] + normal(child(s, 'x', i, 0))
      x[2 * i + 1] = centres[k][1] + normal(child(s, 'x', i, 1))
    }
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    bag: labels(bag),
    proportions: matrix(props, bags, K),
    meta: {
      name: 'label-proportion bags',
      description: `${bags} bags of ${bagSize} instances over ${K} classes, labelled only by their class proportions.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      key: s.key,
    },
  }
}

/** Multiple-instance bags. */
export interface InstanceBags {
  /** Always `'dataset'`. */
  kind: 'dataset'
  /** The instances' features, $n \times 2$, bag by bag. */
  x: Tensor
  /** Each instance's bag (int32). */
  bag: Tensor
  /** Each bag's label (int32, 1 positive), one per bag. */
  bagLabel: Tensor
  /** The instance labels (1 for witnesses, the positive instances); `instanceLabel` holds the same. */
  y: Tensor
  /** The instance labels, as `y`. */
  instanceLabel: Tensor
  /** The description and the stream key. */
  meta: DatasetMeta
}

/**
 * Bags for multiple-instance learning: instances come from a background of two broad Gaussians, at $(0.9, -0.6)$ and
 * $(-0.9, 0.6)$ with standard deviation 0.8; a positive bag's first one to three instances are instead witnesses, from
 * $\Gauss((1.6, 1.6), 0.25^2\Imat)$, a small region the background rarely reaches. A bag is positive exactly when it
 * holds a witness (the standard MIL assumption). Throws `DomainError` unless `bags` is a non-negative integer.
 *
 * @param s The random stream; bag $b$ is drawn from its child `bag` $b$.
 * @param options `bags`, the number of bags (default 40); `size`, the smallest and largest bag size, the size uniform
 *   between them inclusive (default `[5, 12]`); `positiveShare`, the probability that a bag is positive (default 0.5).
 * @returns The instances, their bags, the bag labels and the instance labels.
 *
 * @example A bag is positive exactly when it holds a witness
 * const mil = instanceBags(stream(0), { bags: 10 })
 * print('x:', mil.x.shape, ' bag labels:', mil.bagLabel)
 * const bag = toArray(mil.bag)
 * const witnesses = new Array(10).fill(0)
 * toArray(mil.y).forEach((w, i) => (witnesses[bag[i]] += w))
 * print('witnesses per bag:', witnesses)
 */
export function instanceBags(
  s: Stream,
  options: { bags?: number; size?: readonly [number, number]; positiveShare?: number } = {},
): InstanceBags {
  const { bags = 40, size = [5, 12], positiveShare = 0.5 } = options
  checkCount(bags, 'instanceBags')
  const x: number[] = []
  const bag: number[] = []
  const inst: number[] = []
  const bagLabel = new Int32Array(bags)
  for (let b = 0; b < bags; b++) {
    const bs = child(s, 'bag', b)
    const positive = uniform(child(bs, 'label')) < positiveShare
    bagLabel[b] = positive ? 1 : 0
    const m = size[0] + integers(child(bs, 'size'), size[1] - size[0] + 1)
    const witnesses = positive ? 1 + integers(child(bs, 'witnesses'), 3) : 0
    for (let r = 0; r < m; r++) {
      const w = r < witnesses
      const c = uniform(child(bs, 'side', r)) < 0.5 ? -1 : 1
      x.push(
        w ? 1.6 + 0.25 * normal(child(bs, 'x', r, 0)) : 0.9 * c + 0.8 * normal(child(bs, 'x', r, 0)),
        w ? 1.6 + 0.25 * normal(child(bs, 'x', r, 1)) : -0.6 * c + 0.8 * normal(child(bs, 'x', r, 1)),
      )
      bag.push(b)
      inst.push(w ? 1 : 0)
    }
  }
  return {
    kind: 'dataset',
    x: matrix(Float64Array.from(x), bag.length, 2),
    y: labels(inst),
    bag: labels(bag),
    bagLabel: labels(bagLabel),
    instanceLabel: labels(inst),
    meta: {
      name: 'multiple-instance bags',
      description: `${bags} bags of ${size[0]}–${size[1]} instances; positive bags hold witnesses near (1.6, 1.6).`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      key: s.key,
    },
  }
}

/**
 * Complementary labels: for each example, a class it is not, drawn uniformly from the other $K - 1$. The features are
 * $K$ Gaussian blobs, grouped by class. Throws `DomainError` unless `n` is a non-negative integer.
 *
 * @param s The random stream: the features come from its child `x`, example $i$'s complementary label from `bar` $i$.
 * @param options `n`, the number of examples (default 300); `classes`, the number of classes $K$ (default 3);
 *   `separation`, the distance between neighbouring class centres in standard deviations (default 4).
 * @returns The dataset: `x` ($n \times 2$), the true labels `y`, and `complementary`, a class each example is not
 *   (int32).
 *
 * @example A class each point is not
 * const c = complementaryLabels(stream(0), { n: 300 })
 * print('x:', c.x.shape)
 * print('first true labels:', toArray(c.y).slice(0, 4), ' complementary:', toArray(c.complementary).slice(0, 4))
 * const y = toArray(c.y)
 * const bar = toArray(c.complementary)
 * print('complementary equals true:', bar.filter((v, i) => v === y[i]).length)
 * print('complementary counts:', [0, 1, 2].map((k) => bar.filter((v) => v === k).length))
 */
export function complementaryLabels(
  s: Stream,
  options: { n?: number; classes?: number; separation?: number } = {},
): { kind: 'dataset'; x: Tensor; y: Tensor; complementary: Tensor; meta: DatasetMeta } {
  const { n = 300, classes = 3, separation = 4 } = options
  checkCount(n, 'complementaryLabels')
  const { x, y } = features(child(s, 'x'), n, classes, separation)
  const bar = Int32Array.from(y, (v, i) => {
    const r = integers(child(s, 'bar', i), classes - 1)
    return r >= v ? r + 1 : r
  })
  return {
    kind: 'dataset',
    x,
    y: labels(y),
    complementary: labels(bar),
    meta: {
      name: 'complementary labels',
      description: `${n} points over ${classes} classes, each with one class it does not belong to.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      key: s.key,
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'labellingFunctions',
    name: 'Labelling functions',
    summary: 'Votes of labelling functions with known accuracies and coverages, on Gaussian-blob features.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(10, 10000, { default: 400 }),
      classes: int(2, 10, { default: 2 }),
      functions: int(1, 50, { default: 8 }),
      copies: int(0, 20, { default: 0 }),
      separation: real(0, 10, { default: 3 }),
    }),
    truth: false,
    random: true,
    notes: ['data-programming', 'label-model-estimation', 'snorkel'],
    cite: ['ratner2016'],
  },
  labellingFunctions,
)
dataset(
  {
    key: 'crowdLabels',
    name: 'Crowd labels',
    summary: 'Items labelled by a few of many workers, each answering from a known confusion matrix.',
    task: 'classification',
    output: 'log',
    knobs: space({
      n: int(10, 10000, { default: 200 }),
      classes: int(2, 10, { default: 3 }),
      workers: int(1, 100, { default: 10 }),
      perItem: int(1, 20, { default: 3 }),
    }),
    truth: false,
    random: true,
    notes: ['dawid-skene-model', 'learning-from-crowds'],
    cite: ['dawid1979'],
  },
  crowdLabels,
)
dataset(
  {
    key: 'positiveUnlabelled',
    name: 'Positive–unlabelled sample',
    summary: 'Two Gaussian classes where only a random share c of the positives is labelled.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(10, 10000, { default: 400 }),
      prior: real(0.05, 0.95, { default: 0.4 }),
      labelFrequency: real(0.01, 1, { default: 0.3 }),
      separation: real(0, 10, { default: 3 }),
    }),
    truth: false,
    random: true,
    notes: ['positive-unlabelled-learning'],
    cite: ['elkan2008'],
  },
  positiveUnlabelled,
)
dataset(
  {
    key: 'proportionBags',
    name: 'Label-proportion bags',
    summary: 'Bags of Gaussian-class instances labelled only by their class proportions.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      bags: int(1, 1000, { default: 20 }),
      bagSize: int(1, 500, { default: 20 }),
      classes: int(2, 10, { default: 2 }),
      separation: real(0, 10, { default: 3 }),
    }),
    truth: false,
    random: true,
    notes: ['learning-from-label-proportions'],
  },
  proportionBags,
)
dataset(
  {
    key: 'instanceBags',
    name: 'Multiple-instance bags',
    summary: 'Bags that are positive exactly when they hold a witness instance from a small region.',
    task: 'classification',
    output: 'dataset',
    knobs: space({ bags: int(2, 1000, { default: 40 }), positiveShare: real(0, 1, { default: 0.5 }) }),
    truth: false,
    random: true,
    notes: ['multiple-instance-learning', 'attention-based-multiple-instance-learning'],
  },
  instanceBags,
)
dataset(
  {
    key: 'complementaryLabels',
    name: 'Complementary labels',
    summary: 'Gaussian classes where each point reveals only one class it does not belong to.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(10, 10000, { default: 300 }),
      classes: int(2, 10, { default: 3 }),
      separation: real(0, 10, { default: 4 }),
    }),
    truth: false,
    random: true,
    notes: ['complementary-labels'],
  },
  complementaryLabels,
)
