/**
 * Detection and segmentation metrics: box IoU and its GIoU, DIoU and CIoU variants, mask IoU and Dice, mean IoU and
 * generalised Dice over classes, matching detections to ground truth, average precision with the VOC and COCO
 * interpolations and mAP over IoU thresholds, panoptic quality, mask boundaries and the boundary F-score.
 *
 * Boxes are $[x_0, y_0, x_1, y_1]$ in continuous coordinates, so a box's area is $(x_1 - x_0)(y_1 - y_0)$ with no
 * $+1$ pixel convention. Masks and label maps are read flat, one entry per pixel, with any nonzero entry of a binary
 * mask inside it. Detection average precision follows PASCAL VOC (Everingham et al. 2010) and COCO (Lin et al. 2014):
 * detections are matched greedily in decreasing order of confidence, and the area under the interpolated
 * precision-recall curve is taken exactly or at 11 or 101 recall levels. A ratio with a zero denominator (two empty
 * masks, a class with no pixels) is NaN rather than an error.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  classesOf,
  compareLabels,
  denseMatrix as dense,
  divide,
  encodeLabels,
  labelList,
  matrix,
  sameLength,
  metricValues as values,
} from 'aifn-compute/learning/metrics'
import { defineMetric, type Data, type Label, type Labels, type Rows } from 'aifn-compute/learning/metrics'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * An axis-aligned box $[x_0, y_0, x_1, y_1]$ with $x_0 \le x_1$ and $y_0 \le y_1$: its corners, in continuous
 * coordinates (no $+1$ pixel convention).
 */
export type Box = readonly [number, number, number, number]

/**
 * The area $(x_1 - x_0)(y_1 - y_0)$ of a box, with a negative side taken as 0.
 *
 * @param b The box.
 * @returns Its area, at least 0.
 */
const area = (b: Box) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1])

/** Overlap of a predicted box with a ground-truth box, with the regression-loss variants of IoU. */
export type BoxOverlap = {
  /** The area of the intersection $\lvert B \cap G \rvert$. */
  intersection: number
  /** The area of the union $\lvert B \cup G \rvert$. */
  union: number
  /** The intersection over union $\lvert B \cap G \rvert / \lvert B \cup G \rvert$ (NaN when the union is 0). */
  iou: number
  /** The smallest box $C$ enclosing both. */
  enclosing: Box
  /** Generalised IoU, $\text{IoU} - \lvert C \setminus (B \cup G) \rvert / \lvert C \rvert$, in $[-1, 1]$. */
  giou: number
  /** Distance IoU, $\text{IoU} - \rho^2 / c^2$: $\rho$ the distance between the centres, $c$ the diagonal of $C$. */
  diou: number
  /** Complete IoU, $\text{DIoU} - \alpha v$. */
  ciou: number
  /**
   * CIoU's aspect-ratio term $v = \frac{4}{\pi^2} (\arctan(w_g / h_g) - \arctan(w_b / h_b))^2$, from the widths
   * and heights of the ground-truth and predicted boxes.
   */
  v: number
  /** CIoU's weight $\alpha = v / (1 - \text{IoU} + v)$ on $v$ (0 when $v = 0$). */
  alpha: number
}

/**
 * IoU of a predicted box $B$ against a ground-truth box $G$, with GIoU (Rezatofighi et al. 2019), DIoU and CIoU (Zheng
 * et al. 2020), the variants used as box-regression losses (intersection-over-union). Disjoint boxes have IoU 0, and
 * GIoU and DIoU then still say how far apart they are.
 *
 * @param b The predicted box $B$.
 * @param g The ground-truth box $G$. Only CIoU's aspect-ratio term $v$ tells the two apart.
 * @returns The intersection, union, IoU, enclosing box, GIoU, DIoU and CIoU, with CIoU's $v$ and $\alpha$.
 *
 * @example Two overlapping boxes, and two disjoint ones
 * const o = boxOverlap([0, 0, 2, 2], [1, 1, 3, 3])
 * print('IoU =', o.iou, ' GIoU =', o.giou, ' DIoU =', o.diou, ' CIoU =', o.ciou)
 * const far = boxOverlap([0, 0, 1, 1], [3, 0, 4, 1])
 * print('disjoint: IoU =', far.iou, ' GIoU =', far.giou)
 */
export function boxOverlap(b: Box, g: Box): BoxOverlap {
  const ix = Math.max(0, Math.min(b[2], g[2]) - Math.max(b[0], g[0]))
  const iy = Math.max(0, Math.min(b[3], g[3]) - Math.max(b[1], g[1]))
  const intersection = ix * iy
  const union = area(b) + area(g) - intersection
  const iou = divide(intersection, union)
  const enclosing: Box = [Math.min(b[0], g[0]), Math.min(b[1], g[1]), Math.max(b[2], g[2]), Math.max(b[3], g[3])]
  const c = area(enclosing)
  const giou = iou - divide(c - union, c)
  const rho2 = ((b[0] + b[2] - g[0] - g[2]) / 2) ** 2 + ((b[1] + b[3] - g[1] - g[3]) / 2) ** 2
  const diag2 = (enclosing[2] - enclosing[0]) ** 2 + (enclosing[3] - enclosing[1]) ** 2
  const diou = iou - divide(rho2, diag2)
  const v =
    (4 / Math.PI ** 2) * (Math.atan((g[2] - g[0]) / (g[3] - g[1])) - Math.atan((b[2] - b[0]) / (b[3] - b[1]))) ** 2
  const alpha = v > 0 ? v / (1 - iou + v) : 0
  return { intersection, union, iou, enclosing, giou, diou, ciou: diou - alpha * v, v, alpha }
}

/**
 * The intersection over union $\lvert A \cap B \rvert / \lvert A \cup B \rvert$ of two boxes (NaN when both have
 * zero area).
 *
 * @param groundTruth The ground-truth box.
 * @param prediction The predicted box.
 * @returns The IoU, in $[0, 1]$.
 *
 * @example Two boxes of area 4 overlapping in a unit square
 * print('IoU =', boxIou([0, 0, 2, 2], [1, 1, 3, 3]))
 * print('identical boxes:', boxIou([0, 0, 2, 2], [0, 0, 2, 2]))
 */
export const boxIou = defineMetric(
  {
    module: 'applied/evaluation/detection',
    key: 'boxIou',
    name: 'Box intersection over union',
    inputs: 'boxes',
    direction: 'higher',
    range: [0, 1],
    notes: ['intersection-over-union'],
  },
  (groundTruth: Box, prediction: Box): number => boxOverlap(prediction, groundTruth).iou,
)

/**
 * The $m \times n$ matrix of IoUs between two lists of boxes.
 *
 * @param a The first list, $m$ boxes (the rows).
 * @param b The second list, $n$ boxes (the columns).
 * @returns The IoU of `a[i]` and `b[j]` at row `i`, column `j`.
 *
 * @example Two predictions against three ground-truth boxes
 * const predictions = [[0, 0, 2, 2], [5, 5, 7, 7]]
 * const truth = [[0, 0, 2, 3], [1, 1, 3, 3], [5, 5, 7, 7]]
 * print(boxIouMatrix(predictions, truth))
 */
export function boxIouMatrix(a: readonly Box[], b: readonly Box[]): Tensor {
  const out = new Float64Array(a.length * b.length)
  a.forEach((p, i) => b.forEach((g, j) => (out[i * b.length + j] = boxOverlap(p, g).iou)))
  return matrix(out, a.length, b.length)
}

// ── Masks ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The pixel counts of two binary masks: true positives (inside both), false positives (inside the prediction only)
 * and false negatives (inside the truth only). Masks of different sizes throw.
 *
 * @param truth The true mask, any shape, read flat; nonzero is inside.
 * @param prediction The predicted mask, with as many entries as `truth`.
 * @returns The counts `tp`, `fp` and `fn`.
 */
function maskCounts(truth: Data, prediction: Data) {
  const t = values(truth)
  const p = values(prediction)
  sameLength(t, p, 'mask metric')
  let tp = 0
  let fp = 0
  let fn = 0
  for (let i = 0; i < t.length; i++) {
    const a = t[i] !== 0
    const b = p[i] !== 0
    if (a && b) tp++
    else if (b) fp++
    else if (a) fn++
  }
  return { tp, fp, fn }
}

/**
 * IoU of two binary masks, $\text{TP} / (\text{TP} + \text{FP} + \text{FN})$ over pixels. NaN when both are empty.
 *
 * @param truth The true mask: an array, typed array or tensor of any shape, read flat; nonzero is inside.
 * @param prediction The predicted mask, with as many entries as `truth`.
 * @returns The IoU, in $[0, 1]$.
 *
 * @example Two $3 \times 3$ masks sharing two pixels
 * const truth = tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
 * const prediction = tensor([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
 * print('mask IoU =', maskIou(truth, prediction))
 * print('Dice =', dice(truth, prediction))
 */
export const maskIou = defineMetric(
  {
    module: 'applied/evaluation/detection',
    key: 'maskIou',
    name: 'Mask intersection over union',
    inputs: 'masks',
    direction: 'higher',
    range: [0, 1],
    notes: ['intersection-over-union'],
  },
  (truth: Data, prediction: Data): number => {
    const { tp, fp, fn } = maskCounts(truth, prediction)
    return divide(tp, tp + fp + fn)
  },
)

/**
 * The Dice coefficient $2\lvert A \cap B \rvert / (\lvert A \rvert + \lvert B \rvert)$ of two binary masks, the
 * $F_1$ of the pixel labels (Dice 1945). NaN when both are empty.
 *
 * @param truth The true mask: an array, typed array or tensor of any shape, read flat; nonzero is inside.
 * @param prediction The predicted mask, with as many entries as `truth`.
 * @returns The Dice coefficient, in $[0, 1]$; it is $2J/(1 + J)$ for the IoU $J$.
 *
 * @example Dice beside IoU on flat masks
 * const truth = [1, 1, 1, 1, 0, 0]
 * const prediction = [0, 1, 1, 1, 1, 0]
 * print('Dice =', dice(truth, prediction))
 * print('IoU =', maskIou(truth, prediction))
 */
export const dice = defineMetric(
  {
    module: 'applied/evaluation/detection',
    key: 'dice',
    name: 'Dice coefficient',
    inputs: 'masks',
    direction: 'higher',
    range: [0, 1],
    notes: ['dice-and-segmentation-overlap'],
  },
  (truth: Data, prediction: Data): number => {
    const { tp, fp, fn } = maskCounts(truth, prediction)
    return divide(2 * tp, 2 * tp + fp + fn)
  },
)

/**
 * Per-class IoU of two label maps (pixels as cases), pooled over all pixels, for the classes in either map (or
 * `classes`); pixels whose truth is `ignore` are left out. A class in neither map has IoU NaN. Label maps of different
 * sizes throw.
 *
 * @param truth The true class of each pixel, read flat.
 * @param prediction The predicted class of each pixel, matching `truth`.
 * @param options The classes to score, and the label to ignore.
 * @param options.classes The classes to report, in this order. Left out, every class in either map (except `ignore`),
 *   sorted. A pixel whose label is not in the list counts for no class.
 * @param options.ignore A true label whose pixels are skipped entirely, as the "void" label of a segmentation
 *   benchmark.
 * @returns `classes`, the classes scored, and `iou`, a vector of their IoUs in the same order.
 *
 * @example Three classes on a six-pixel map, with an ignored pixel
 * const truth = [0, 0, 1, 1, 2, 255]
 * const prediction = [0, 1, 1, 1, 2, 0]
 * print(iouPerClass(truth, prediction, { ignore: 255 }))
 */
export function iouPerClass(
  truth: Labels,
  prediction: Labels,
  options: { classes?: readonly Label[]; ignore?: Label } = {},
): { classes: Label[]; iou: Tensor } {
  const t = labelList(truth)
  const p = labelList(prediction)
  sameLength(t, p, 'iouPerClass')
  const classes = options.classes
    ? [...options.classes]
    : classesOf(t, p).filter((c) => options.ignore === undefined || compareLabels(c, options.ignore) !== 0)
  const ti = encodeLabels(t, classes)
  const pi = encodeLabels(p, classes)
  const K = classes.length
  const inter = new Float64Array(K)
  const union = new Float64Array(K)
  for (let i = 0; i < t.length; i++) {
    if (options.ignore !== undefined && t[i] === options.ignore) continue
    if (ti[i] >= 0 && ti[i] === pi[i]) {
      inter[ti[i]]++
      union[ti[i]]++
    } else {
      if (ti[i] >= 0) union[ti[i]]++
      if (pi[i] >= 0) union[pi[i]]++
    }
  }
  return { classes, iou: fromData(Float64Array.from(inter, (v, k) => divide(v, union[k]))) }
}

/**
 * Mean IoU (dice-and-segmentation-overlap): per-class IoU over the whole set, as `iouPerClass`, averaged over the
 * classes whose IoU is defined.
 *
 * @param truth The true class of each pixel, read flat.
 * @param prediction The predicted class of each pixel, matching `truth`.
 * @param options The classes to score, and the label to ignore.
 * @param options.classes The classes to average over (default every class in either map).
 * @param options.ignore A true label whose pixels are skipped.
 * @returns The mean of the per-class IoUs, skipping NaN ones, in $[0, 1]$.
 *
 * @example The mean of the per-class IoUs $1/2$, $2/3$ and 1
 * print('mean IoU =', meanIou([0, 0, 1, 1, 2, 255], [0, 1, 1, 1, 2, 0], { ignore: 255 }))
 */
export const meanIou = defineMetric(
  {
    module: 'applied/evaluation/detection',
    key: 'meanIou',
    name: 'Mean IoU',
    inputs: 'masks',
    direction: 'higher',
    range: [0, 1],
    notes: ['dice-and-segmentation-overlap'],
  },
  (truth: Labels, prediction: Labels, options: { classes?: readonly Label[]; ignore?: Label } = {}): number => {
    const v = iouPerClass(truth, prediction, options).iou.data as Float64Array
    let s = 0
    let k = 0
    for (const x of v)
      if (!Number.isNaN(x)) {
        s += x
        k++
      }
    return divide(s, k)
  },
)

/**
 * Generalised Dice (Sudre et al. 2017), $2 \sum_l w_l \sum_n r_{ln} p_{ln} / \sum_l w_l \sum_n (r_{ln} + p_{ln})$
 * with $w_l = 1 / (\sum_n r_{ln})^2$, which weights each class by the inverse square of its volume so that small
 * classes count. Classes absent from the truth would have infinite weight and are skipped. Matrices of different
 * shapes throw `ShapeError`.
 *
 * @param truth The one-hot truth $r_{ln}$, an $N \times L$ matrix: pixels as rows, classes as columns.
 * @param prediction The predicted labels (one-hot) or probabilities $p_{ln}$, of the same shape.
 * @returns The generalised Dice, in $[0, 1]$.
 *
 * @example A small class half missed weighs as much as a large one
 * const truth = [[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]
 * const prediction = [[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0]]
 * print('generalised Dice =', generalisedDice(truth, prediction))
 * print('pixel Dice of class 1 =', dice([0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0]))
 */
export const generalisedDice = defineMetric(
  {
    module: 'applied/evaluation/detection',
    key: 'generalisedDice',
    name: 'Generalised Dice',
    inputs: 'masks',
    direction: 'higher',
    range: [0, 1],
    notes: ['dice-and-segmentation-overlap'],
  },
  (truth: Rows, prediction: Rows): number => {
    const R = dense(truth, 'generalisedDice')
    const P = dense(prediction, 'generalisedDice')
    if (R.rows !== P.rows || R.cols !== P.cols)
      throw new ShapeError('metrics', 'metrics: generalisedDice: shapes differ')
    let num = 0
    let den = 0
    for (let l = 0; l < R.cols; l++) {
      let vol = 0
      let inter = 0
      let sum = 0
      for (let n = 0; n < R.rows; n++) {
        const r = R.data[n * R.cols + l]
        const p = P.data[n * R.cols + l]
        vol += r
        inter += r * p
        sum += r + p
      }
      if (vol === 0) continue
      const w = 1 / (vol * vol)
      num += w * inter
      den += w * sum
    }
    return divide(2 * num, den)
  },
)

// ── Detection average precision ──────────────────────────────────────────────────────────────────────────────────────

/**
 * A ground-truth object: the `image` it is in (any id), its `box`, its class `label` (optional), and `ignore`, which
 * marks VOC "difficult" or COCO "crowd" objects, which count neither way.
 */
export type GroundTruthObject = { image: number | string; box: Box; label?: Label; ignore?: boolean }

/** A detection: the `image` it is in, its `box`, its confidence `score` (higher is surer) and its class `label`. */
export type Detection = { image: number | string; box: Box; score: number; label?: Label }

/** Each detection's outcome after matching, in decreasing order of score. */
export type Match = {
  /** The index of the detection in the input list. */
  detection: number
  /** The detection's confidence. */
  score: number
  /** True positive, false positive, or ignored (matched an ignored object). */
  outcome: 'tp' | 'fp' | 'ignored'
  /** Index of the matched ground-truth object, or $-1$. */
  object: number
  /** The IoU with the matched object (0 for a false positive). */
  iou: number
}

/**
 * Match detections of one class to ground truth in decreasing order of confidence
 * (mean-average-precision-for-detection): each detection takes the unmatched ground-truth object in its image with the
 * highest IoU $\ge$ `iouThreshold` (COCO's rule). An ignored object is taken only when no other qualifies, and it may
 * absorb several detections; a detection that matches nothing is a false positive. Labels are ignored here;
 * `detectionAveragePrecision` filters by class first.
 *
 * @param groundTruth The ground-truth objects, of every image.
 * @param detections The detections, of every image, in any order. Ties in score keep their input order.
 * @param options `iouThreshold`, the smallest IoU that counts as a match (default 0.5).
 * @returns One `Match` per detection, in decreasing order of score.
 *
 * @example Two detections of one object: the surer one takes it
 * const truth = [{ image: 0, box: [0, 0, 10, 10] }]
 * const detections = [
 *   { image: 0, box: [1, 1, 10, 10], score: 0.6 },
 *   { image: 0, box: [0, 0, 10, 9], score: 0.9 },
 *   { image: 0, box: [20, 20, 30, 30], score: 0.7 },
 * ]
 * for (const m of matchDetections(truth, detections)) print(m)
 */
export function matchDetections(
  groundTruth: readonly GroundTruthObject[],
  detections: readonly Detection[],
  options: { iouThreshold?: number } = {},
): Match[] {
  const t = options.iouThreshold ?? 0.5
  const order = detections.map((_, i) => i).sort((a, b) => detections[b].score - detections[a].score || a - b)
  const taken = new Uint8Array(groundTruth.length)
  const byImage = new Map<number | string, number[]>()
  groundTruth.forEach((g, j) => byImage.set(g.image, [...(byImage.get(g.image) ?? []), j]))
  return order.map((i) => {
    const d = detections[i]
    let best = -1
    let bestIou = -1
    // Unmatched real objects first; an ignored object (which may absorb several detections) only when none qualifies.
    for (const pass of [false, true]) {
      for (const j of byImage.get(d.image) ?? []) {
        if (!!groundTruth[j].ignore !== pass || (!pass && taken[j])) continue
        const iou = boxOverlap(d.box, groundTruth[j].box).iou
        if (iou >= t && iou > bestIou) {
          best = j
          bestIou = iou
        }
      }
      if (best >= 0) break
    }
    if (best < 0) return { detection: i, score: d.score, outcome: 'fp', object: -1, iou: 0 }
    if (groundTruth[best].ignore)
      return { detection: i, score: d.score, outcome: 'ignored', object: best, iou: bestIou }
    taken[best] = 1
    return { detection: i, score: d.score, outcome: 'tp', object: best, iou: bestIou }
  })
}

/**
 * How the precision envelope is integrated: at every distinct recall (`all-points`, VOC 2010 on), or as the mean at
 * 11 (`11-point`, VOC 2007) or 101 (`101-point`, COCO) equally spaced recall levels.
 */
export type ApInterpolation = 'all-points' | '11-point' | '101-point'

/**
 * Average precision from a list of detections already sorted by decreasing confidence, each a true or false positive,
 * and the number of ground-truth objects (mean-average-precision-for-detection): the area under the interpolated
 * precision $P_{\text{interp}}(r) = \max_{j : R_j \ge r} P_j$, exactly (`all-points`) or sampled at 11 or 101 recall
 * levels.
 *
 * @param isTruePositive For each detection, in decreasing order of confidence, whether it is a true positive.
 * @param groundTruthCount The number of ground-truth objects, the denominator of recall. With 0 the AP is NaN.
 * @param interpolation How the precision envelope is integrated (see `ApInterpolation`).
 * @returns `averagePrecision`, and the vectors `precision` and `recall` after each detection (not interpolated).
 *
 * @example Three hits and a miss among four detections of four objects
 * const { averagePrecision, precision, recall } = averagePrecisionFromMatches([true, true, false, true], 4)
 * print('AP =', averagePrecision)
 * print('precision =', precision)
 * print('recall =', recall)
 * print('11-point AP =', averagePrecisionFromMatches([true, true, false, true], 4, '11-point').averagePrecision)
 */
export function averagePrecisionFromMatches(
  isTruePositive: ArrayLike<boolean | number>,
  groundTruthCount: number,
  interpolation: ApInterpolation = 'all-points',
): { averagePrecision: number; precision: Tensor; recall: Tensor } {
  const n = isTruePositive.length
  const precision = new Float64Array(n)
  const recall = new Float64Array(n)
  let tp = 0
  for (let j = 0; j < n; j++) {
    if (isTruePositive[j]) tp++
    precision[j] = tp / (j + 1)
    recall[j] = divide(tp, groundTruthCount)
  }
  // The envelope: the best precision at this recall or beyond, a running maximum from the end.
  const envelope = Float64Array.from(precision)
  for (let j = n - 2; j >= 0; j--) envelope[j] = Math.max(envelope[j], envelope[j + 1])
  const interpolated = (r: number) => {
    for (let j = 0; j < n; j++) if (recall[j] >= r - 1e-12) return envelope[j]
    return 0
  }
  let ap = 0
  if (groundTruthCount === 0) ap = NaN
  else if (interpolation === 'all-points') {
    let prev = 0
    for (let j = 0; j < n; j++)
      if (recall[j] > prev) {
        ap += (recall[j] - prev) * envelope[j]
        prev = recall[j]
      }
  } else {
    const m = interpolation === '11-point' ? 11 : 101
    for (let i = 0; i < m; i++) ap += interpolated(i / (m - 1)) / m
  }
  return { averagePrecision: ap, precision: fromData(precision), recall: fromData(recall) }
}

/**
 * Detection AP for one class (or for all objects when `label` is left out), at one IoU threshold: the detections are
 * matched by `matchDetections`, those matching an ignored object dropped, and the rest scored by
 * `averagePrecisionFromMatches` against the number of objects not ignored.
 *
 * @param groundTruth The ground-truth objects, of every image.
 * @param detections The detections, of every image.
 * @param options The class, the IoU threshold and the interpolation.
 * @param options.iouThreshold The smallest IoU that counts as a match (default 0.5).
 * @param options.interpolation How the precision envelope is integrated (default `all-points`).
 * @param options.label The class to score: only objects and detections with this `label` are used. Left out, every
 *   object and detection is used, whatever its label.
 * @returns `averagePrecision`, and the vectors `precision` and `recall` after each matched detection.
 *
 * @example AP of one class across two images
 * const truth = [
 *   { image: 'a', box: [0, 0, 10, 10], label: 'cat' },
 *   { image: 'b', box: [5, 5, 15, 15], label: 'cat' },
 * ]
 * const detections = [
 *   { image: 'a', box: [0, 0, 10, 11], score: 0.9, label: 'cat' },
 *   { image: 'b', box: [30, 30, 40, 40], score: 0.8, label: 'cat' },
 *   { image: 'b', box: [5, 6, 15, 15], score: 0.4, label: 'cat' },
 * ]
 * print('AP@0.5 =', detectionAveragePrecision(truth, detections, { label: 'cat' }).averagePrecision)
 */
export function detectionAveragePrecision(
  groundTruth: readonly GroundTruthObject[],
  detections: readonly Detection[],
  options: { iouThreshold?: number; interpolation?: ApInterpolation; label?: Label } = {},
): { averagePrecision: number; precision: Tensor; recall: Tensor } {
  const g = options.label === undefined ? groundTruth : groundTruth.filter((o) => o.label === options.label)
  const d = options.label === undefined ? detections : detections.filter((o) => o.label === options.label)
  const matches = matchDetections(g, d, options).filter((m) => m.outcome !== 'ignored')
  const count = g.filter((o) => !o.ignore).length
  return averagePrecisionFromMatches(
    matches.map((m) => m.outcome === 'tp'),
    count,
    options.interpolation ?? 'all-points',
  )
}

/** The ten COCO IoU thresholds $0.50, 0.55, \dots, 0.95$. */
export const cocoIouThresholds: readonly number[] = Array.from({ length: 10 }, (_, i) => 0.5 + 0.05 * i)

/**
 * Mean average precision for detection: AP averaged over classes (those with a ground-truth object not ignored) and
 * over IoU thresholds. Defaults are COCO's: thresholds $0.50, 0.55, \dots, 0.95$ and 101-point interpolation (Lin et
 * al. 2014). For PASCAL VOC use `{ iouThresholds: [0.5], interpolation: 'all-points' }` (Everingham et al. 2010). An
 * object or detection with no `label` is in class 0.
 *
 * @param groundTruth The ground-truth objects, of every image and class.
 * @param detections The detections, of every image and class.
 * @param options The IoU thresholds and the interpolation.
 * @param options.iouThresholds The IoU thresholds to average over (default `cocoIouThresholds`).
 * @param options.interpolation How the precision envelope is integrated (default `101-point`).
 * @returns The mean of the AP over every class and threshold, in $[0, 1]$.
 *
 * @example COCO-style and VOC-style mAP of a few detections
 * const truth = [
 *   { image: 0, box: [0, 0, 10, 10], label: 'cat' },
 *   { image: 0, box: [20, 0, 30, 10], label: 'dog' },
 *   { image: 1, box: [0, 0, 8, 8], label: 'dog' },
 * ]
 * const detections = [
 *   { image: 0, box: [0, 0, 10, 9], score: 0.9, label: 'cat' },
 *   { image: 0, box: [21, 0, 30, 11], score: 0.8, label: 'dog' },
 *   { image: 1, box: [1, 1, 9, 9], score: 0.6, label: 'dog' },
 *   { image: 1, box: [20, 20, 25, 25], score: 0.7, label: 'cat' },
 * ]
 * print('mAP (COCO) =', detectionMeanAveragePrecision(truth, detections))
 * const voc = { iouThresholds: [0.5], interpolation: 'all-points' }
 * print('mAP (VOC) =', detectionMeanAveragePrecision(truth, detections, voc))
 */
export const detectionMeanAveragePrecision = defineMetric(
  {
    module: 'applied/evaluation/detection',
    key: 'detectionMeanAveragePrecision',
    name: 'Mean average precision (detection)',
    inputs: 'boxes',
    direction: 'higher',
    range: [0, 1],
    notes: ['mean-average-precision-for-detection'],
    capability: 'score',
  },
  (
    groundTruth: readonly GroundTruthObject[],
    detections: readonly Detection[],
    options: { iouThresholds?: readonly number[]; interpolation?: ApInterpolation } = {},
  ): number => {
    const thresholds = options.iouThresholds ?? cocoIouThresholds
    const labels = classesOf(groundTruth.filter((o) => !o.ignore).map((o) => o.label ?? 0))
    let s = 0
    for (const t of thresholds)
      for (const label of labels) {
        const g = groundTruth.filter((o) => (o.label ?? 0) === label)
        const d = detections.filter((o) => (o.label ?? 0) === label)
        s += detectionAveragePrecision(g, d, {
          iouThreshold: t,
          interpolation: options.interpolation ?? '101-point',
        }).averagePrecision
      }
    return s / (thresholds.length * labels.length)
  },
)

// ── Panoptic quality ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Panoptic quality $\text{PQ} = \text{SQ} \times \text{RQ}$ of one class from its matched IoUs and unmatched counts
 * (Kirillov et al. 2019): $\text{SQ}$ is the mean IoU of the matches and
 * $\text{RQ} = \text{TP} / (\text{TP} + \text{FP}/2 + \text{FN}/2)$, an $F_1$ of the segments.
 *
 * @param matchedIous The IoU of each matched pair of segments (the true positives); its length is $\text{TP}$.
 * @param falsePositives The number of predicted segments matched to nothing.
 * @param falseNegatives The number of ground-truth segments matched to nothing.
 * @returns `pq`, `sq` (NaN with no match) and `rq` (NaN when all three counts are 0).
 *
 * @example Three matches, one spurious and one missed segment
 * print(panopticFromMatches([0.9, 0.8, 0.7], 1, 1))
 */
export function panopticFromMatches(
  matchedIous: ArrayLike<number>,
  falsePositives: number,
  falseNegatives: number,
): { pq: number; sq: number; rq: number } {
  const tp = matchedIous.length
  let sum = 0
  for (let i = 0; i < tp; i++) sum += matchedIous[i]
  const sq = divide(sum, tp)
  const rq = divide(tp, tp + falsePositives / 2 + falseNegatives / 2)
  return { pq: divide(sum, tp + falsePositives / 2 + falseNegatives / 2), sq, rq }
}

/**
 * Panoptic quality of two segment maps (panoptic-quality): every pixel carries a segment id, and `truthClass` and
 * `predictionClass` map a segment id to its class (default: one class). Within each class, a predicted and a
 * ground-truth segment match when IoU $> 0.5$, which makes the matching unique;
 * $\text{PQ} = \sum \text{IoU} / (\text{TP} + \text{FP}/2 + \text{FN}/2)$ per class, as `panopticFromMatches`,
 * averaged over classes. Pixels whose truth is `voidId` are removed from the predictions first, and predictions more
 * than half void are ignored. Maps of different sizes throw.
 *
 * @param truth The ground-truth segment id of each pixel, read flat.
 * @param prediction The predicted segment id of each pixel, matching `truth`.
 * @param options How segments map to classes, and the void label.
 * @param options.truthClass The class of a ground-truth segment id (default: every segment in class 0).
 * @param options.predictionClass The class of a predicted segment id (default: every segment in class 0).
 * @param options.voidId A ground-truth segment id for unlabelled pixels, which count for no segment.
 * @returns `pq`, `sq` and `rq` averaged over the classes that have a match or an unmatched segment (a class with no
 *   match adds 0 to `sq`), and `perClass`, each class's own values.
 *
 * @example Two true segments: one found, one split in two
 * const truth = [1, 1, 1, 1, 2, 2, 2, 2]
 * const prediction = [7, 7, 7, 0, 8, 8, 9, 9]
 * const { pq, sq, rq } = panopticQuality(truth, prediction)
 * print('PQ =', pq, ' SQ =', sq, ' RQ =', rq)
 */
export function panopticQuality(
  truth: Labels,
  prediction: Labels,
  options: {
    truthClass?: (segment: Label) => Label
    predictionClass?: (segment: Label) => Label
    voidId?: Label
  } = {},
): { pq: number; sq: number; rq: number; perClass: Map<Label, { pq: number; sq: number; rq: number }> } {
  const t = labelList(truth)
  const p = labelList(prediction)
  sameLength(t, p, 'panopticQuality')
  const tc = options.truthClass ?? (() => 0)
  const pc = options.predictionClass ?? (() => 0)
  const area = new Map<string, number>()
  const key = (a: Label, b: Label) => `${String(a)}\u0000${String(b)}`
  const tArea = new Map<Label, number>()
  const pArea = new Map<Label, number>()
  const pVoid = new Map<Label, number>()
  for (let i = 0; i < t.length; i++) {
    if (options.voidId !== undefined && t[i] === options.voidId) {
      pVoid.set(p[i], (pVoid.get(p[i]) ?? 0) + 1)
      continue
    }
    tArea.set(t[i], (tArea.get(t[i]) ?? 0) + 1)
    pArea.set(p[i], (pArea.get(p[i]) ?? 0) + 1)
    area.set(key(t[i], p[i]), (area.get(key(t[i], p[i])) ?? 0) + 1)
  }
  const matchedT = new Set<Label>()
  const matchedP = new Set<Label>()
  const perClassIous = new Map<Label, number[]>()
  for (const [k, inter] of area) {
    const [ts, ps] = k.split('\u0000')
    const tSeg = [...tArea.keys()].find((s) => String(s) === ts)!
    const pSeg = [...pArea.keys()].find((s) => String(s) === ps)!
    if (tc(tSeg) !== pc(pSeg)) continue
    const iou = inter / (tArea.get(tSeg)! + pArea.get(pSeg)! - inter)
    if (iou > 0.5) {
      matchedT.add(tSeg)
      matchedP.add(pSeg)
      const c = tc(tSeg)
      perClassIous.set(c, [...(perClassIous.get(c) ?? []), iou])
    }
  }
  const fp = new Map<Label, number>()
  const fn = new Map<Label, number>()
  for (const s of tArea.keys()) if (!matchedT.has(s)) fn.set(tc(s), (fn.get(tc(s)) ?? 0) + 1)
  for (const [s, a] of pArea) {
    if (matchedP.has(s)) continue
    const voidPixels = pVoid.get(s) ?? 0
    if (voidPixels / (a + voidPixels) > 0.5) continue
    fp.set(pc(s), (fp.get(pc(s)) ?? 0) + 1)
  }
  const classes = new Set<Label>([...perClassIous.keys(), ...fp.keys(), ...fn.keys()])
  const perClass = new Map<Label, { pq: number; sq: number; rq: number }>()
  let pq = 0
  let sq = 0
  let rq = 0
  for (const c of classes) {
    const r = panopticFromMatches(perClassIous.get(c) ?? [], fp.get(c) ?? 0, fn.get(c) ?? 0)
    perClass.set(c, r)
    pq += r.pq
    sq += Number.isNaN(r.sq) ? 0 : r.sq
    rq += r.rq
  }
  const K = classes.size
  return { pq: divide(pq, K), sq: divide(sq, K), rq: divide(rq, K), perClass }
}

// ── Boundaries ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The boundary pixels of a binary mask of $\text{height} \times \text{width}$ (row-major): pixels inside the mask
 * with a 4-neighbour outside it or on the image edge. Returned as (row, column) points, a $k \times 2$ matrix. A flat
 * mask whose length is not a multiple of `width` throws `ShapeError`.
 *
 * @param mask The mask, nonzero inside: a matrix (rows of numbers or a rank-2 tensor), or a flat array read row-major
 *   with `width`.
 * @param options `width`, the number of columns of a flat mask (default the matrix's column count; a flat array read
 *   without it is one column).
 * @returns The $k$ boundary pixels as rows `[row, column]`, in row-major order.
 *
 * @example The boundary of a filled $3 \times 3$ square in a $5 \times 5$ image
 * const mask = [
 *   [0, 0, 0, 0, 0],
 *   [0, 1, 1, 1, 0],
 *   [0, 1, 1, 1, 0],
 *   [0, 1, 1, 1, 0],
 *   [0, 0, 0, 0, 0],
 * ]
 * print(maskBoundary(mask))
 */
export function maskBoundary(mask: Data | Rows, options: { width?: number } = {}): Tensor {
  const m = dense(mask as Rows, 'maskBoundary')
  const width = options.width ?? m.cols
  const flat = m.data
  const height = flat.length / width
  if (!Number.isInteger(height))
    throw new ShapeError('metrics', 'metrics: maskBoundary: the mask does not fill whole rows')
  const inside = (r: number, c: number) => r >= 0 && r < height && c >= 0 && c < width && flat[r * width + c] !== 0
  const pts: number[] = []
  for (let r = 0; r < height; r++)
    for (let c = 0; c < width; c++)
      if (inside(r, c) && !(inside(r - 1, c) && inside(r + 1, c) && inside(r, c - 1) && inside(r, c + 1)))
        pts.push(r, c)
  return matrix(Float64Array.from(pts), pts.length / 2, 2)
}

/**
 * The boundary F-score (Csurka et al. 2013; Perazzi et al. 2016): precision is the fraction of predicted boundary
 * pixels within `tolerance` pixels (Euclidean) of a true boundary pixel, recall the reverse, and the score their
 * harmonic mean. Boundaries are found by `maskBoundary`. NaN when either mask is empty.
 *
 * @param truth The true mask, $\text{height} \times \text{width}$, as a matrix or flat with `width`; nonzero inside.
 * @param prediction The predicted mask, laid out as `truth`.
 * @param options The width of flat masks, and the tolerance.
 * @param options.width The number of columns of a flat mask (default the matrix's column count).
 * @param options.tolerance The largest distance, in pixels, at which a boundary pixel counts as found (default 2).
 * @returns The boundary F-score, in $[0, 1]$.
 *
 * @example A square shifted by one pixel, at tolerances 0 and 2
 * const truth = [
 *   [1, 1, 1, 0, 0],
 *   [1, 1, 1, 0, 0],
 *   [1, 1, 1, 0, 0],
 *   [0, 0, 0, 0, 0],
 * ]
 * const prediction = [
 *   [0, 1, 1, 1, 0],
 *   [0, 1, 1, 1, 0],
 *   [0, 1, 1, 1, 0],
 *   [0, 0, 0, 0, 0],
 * ]
 * print('tolerance 0:', boundaryF1(truth, prediction, { tolerance: 0 }))
 * print('tolerance 2:', boundaryF1(truth, prediction))
 */
export const boundaryF1 = defineMetric(
  {
    module: 'applied/evaluation/detection',
    key: 'boundaryF1',
    name: 'Boundary F-score',
    inputs: 'masks',
    direction: 'higher',
    range: [0, 1],
    notes: ['hausdorff-distance'],
  },
  (truth: Data | Rows, prediction: Data | Rows, options: { width?: number; tolerance?: number } = {}): number => {
    const T = boundaryPoints(truth, options.width)
    const P = boundaryPoints(prediction, options.width)
    const tol2 = (options.tolerance ?? 2) ** 2
    const within = (a: number[][], b: number[][]) =>
      a.filter(([r, c]) => b.some(([s, d]) => (r - s) ** 2 + (c - d) ** 2 <= tol2)).length
    const precision = divide(within(P, T), P.length)
    const recall = divide(within(T, P), T.length)
    return divide(2 * precision * recall, precision + recall)
  },
)

/**
 * The boundary pixels of a mask, as `maskBoundary` finds them, as an array of `[row, column]` pairs.
 *
 * @param mask The mask, as a matrix or flat with `width`; nonzero inside.
 * @param width The number of columns of a flat mask (default the matrix's column count).
 * @returns The boundary pixels, in row-major order.
 */
function boundaryPoints(mask: Data | Rows, width?: number): number[][] {
  const b = maskBoundary(mask, { width })
  const d = b.data as Float64Array
  return Array.from({ length: b.shape[0] }, (_, i) => [d[2 * i], d[2 * i + 1]])
}
