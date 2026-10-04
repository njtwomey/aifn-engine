/**
 * Data for weak supervision with known ground truth: labelling functions with known accuracies and coverages, crowd
 * workers with known confusion matrices, positive–unlabelled samples with a known label frequency, bags with known
 * class proportions, multiple-instance bags with known witnesses, and complementary labels. Features come from Gaussian
 * blobs (`blobs`), one per class, so the true labels are learnable from x.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { child, integers, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkCount, labels, matrix, type DatasetMeta } from '../types'
import { blobs } from './points'

/** Features and true labels from `classes` Gaussian blobs on a polygon, `separation` standard deviations apart. */
function features(s: Stream, n: number, classes: number, separation: number) {
  const d = blobs(s, { n, centers: classes, separation, layout: 'polygon', sd: 1 })
  return { x: d.x, y: Int32Array.from(toFlat(d.y!)) }
}

const between = (s: Stream, [lo, hi]: readonly [number, number]) => lo + (hi - lo) * uniform(s)

/** Votes of labelling functions with their true accuracies and coverages. */
export interface LabellingFunctions {
  kind: 'dataset'
  x: Tensor
  /** True labels (int32), unseen by the label models. */
  y: Tensor
  /** Votes [n, m] (int32): a class, or −1 for an abstention. */
  votes: Tensor
  /** P(vote = y | voted) and P(voted) of each function. */
  accuracy: Tensor
  coverage: Tensor
  meta: DatasetMeta
}

/** Options of `labellingFunctions`. */
export interface LabellingFunctionOptions {
  n?: number
  classes?: number
  /** Labelling functions m (default 8). */
  functions?: number
  /** Accuracies and coverages are drawn uniformly from these ranges (default [0.55, 0.9] and [0.2, 0.8]). */
  accuracy?: readonly [number, number]
  coverage?: readonly [number, number]
  /** Extra copies of function 0 that repeat its votes (breaking conditional independence; default 0). */
  copies?: number
  /** Class separation of the features in standard deviations (default 3). */
  separation?: number
}

/**
 * Labelling functions in the data-programming model (Ratner et al., 2016): function j votes on an example with
 * probability β_j (its coverage) and, when it votes, names the true class with probability α_j and each other class with
 * probability (1 − α_j)/(K − 1). The accuracies and coverages are drawn per function and returned, so a label model's
 * estimates can be compared with the truth. `copies` duplicates function 0, which double-counts it for any model that
 * assumes independent functions.
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
  y: Tensor
  /** Votes [n, workers] (int32), −1 where a worker did not label the item. */
  votes: Tensor
  /** P(worker says l | class k) [workers, K, K]. */
  confusions: Tensor
  meta: DatasetMeta
}

/** Options of `crowdLabels`. */
export interface CrowdLabelOptions {
  n?: number
  classes?: number
  workers?: number
  /** Workers per item (default 3). */
  perItem?: number
  /** Each worker's accuracy is drawn from this range (default [0.35, 0.95]); errors favour one confusable class. */
  skill?: readonly [number, number]
}

/**
 * Crowdsourced labels in the Dawid–Skene model (Dawid and Skene, 1979): each item is labelled by `perItem` workers
 * drawn without replacement; worker j answers from row y of its confusion matrix, whose diagonal is its skill and whose
 * errors go mostly (70%) to one confusable class per true class and otherwise uniformly.
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
  kind: 'dataset'
  x: Tensor
  /** True classes (1 positive) and the observed labels s (1 labelled positive, 0 unlabelled). */
  y: Tensor
  labelled: Tensor
  /** The true class prior and label frequency c = P(s = 1 | y = 1). */
  prior: number
  labelFrequency: number
  meta: DatasetMeta
}

/**
 * Positive and unlabelled data under "selected completely at random": points from two Gaussian classes with prior π
 * for the positive class; each positive is labelled with probability c independently of x, and every other point is
 * unlabelled.
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
  kind: 'dataset'
  x: Tensor
  y: Tensor
  /** Each instance's bag (int32) and each bag's class proportions [B, K]. */
  bag: Tensor
  proportions: Tensor
  meta: DatasetMeta
}

/**
 * Bags for learning from label proportions: bag b draws a class mixture from a symmetric Dirichlet(`concentration`),
 * then `bagSize` labels from it and each instance's features from its class's Gaussian; only the bag's realised class
 * proportions are revealed.
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
  kind: 'dataset'
  x: Tensor
  /** Each instance's bag, the bag labels, and which instances are witnesses (the positive ones). */
  bag: Tensor
  bagLabel: Tensor
  /** The instance labels (1 for witnesses); `y` holds the same. */
  y: Tensor
  instanceLabel: Tensor
  meta: DatasetMeta
}

/**
 * Bags for multiple-instance learning: instances come from a background of two broad Gaussians; a positive bag also
 * holds one to three witnesses from a small region the background rarely reaches. A bag is positive exactly when it
 * holds a witness (the standard MIL assumption).
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

/** Complementary labels: for each example, a class it is not, drawn uniformly from the other K − 1. */
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
