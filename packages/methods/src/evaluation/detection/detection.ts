/**
 * Detection and segmentation metrics: box IoU and its GIoU, DIoU and CIoU variants, mask IoU and Dice, mean IoU and
 * generalised Dice over classes, matching detections to ground truth, average precision with the VOC and COCO
 * interpolations and mAP over IoU thresholds, panoptic quality, mask boundaries and the boundary F-score.
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

/** An axis-aligned box [x₀, y₀, x₁, y₁] with x₀ ≤ x₁ and y₀ ≤ y₁ (continuous coordinates, no +1 pixel convention). */
export type Box = readonly [number, number, number, number]

const area = (b: Box) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1])

/** Overlap of a predicted box with a ground-truth box, with the regression-loss variants. */
export type BoxOverlap = {
  intersection: number
  union: number
  iou: number
  /** Smallest box enclosing both. */
  enclosing: Box
  giou: number
  diou: number
  ciou: number
  /** CIoU's aspect-ratio term v and its weight α. */
  v: number
  alpha: number
}

/**
 * IoU of a prediction b against ground truth g, and GIoU (Rezatofighi et al. 2019), DIoU and CIoU (Zheng et al. 2020)
 * (intersection-over-union).
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

/** IoU |A ∩ B|/|A ∪ B| of two boxes. */
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

/** The m × n matrix of IoUs between two lists of boxes. */
export function boxIouMatrix(a: readonly Box[], b: readonly Box[]): Tensor {
  const out = new Float64Array(a.length * b.length)
  a.forEach((p, i) => b.forEach((g, j) => (out[i * b.length + j] = boxOverlap(p, g).iou)))
  return matrix(out, a.length, b.length)
}

// ── Masks ────────────────────────────────────────────────────────────────────────────────────────────────────────────

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

/** IoU of two binary masks (any shape, nonzero = inside): TP/(TP + FP + FN). NaN when both are empty. */
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

/** The Dice coefficient 2|A ∩ B|/(|A| + |B|) of two binary masks, the F₁ of the pixel labels (Dice 1945). */
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
 * `classes`); `ignore` pixels in the truth are left out.
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

/** Mean IoU: per-class IoU over the whole set, averaged over classes (dice-and-segmentation-overlap). */
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
 * Generalised Dice (Sudre et al. 2017): 2 Σₗ wₗ Σₙ rₗₙpₗₙ / Σₗ wₗ Σₙ (rₗₙ + pₗₙ) with wₗ = 1/(Σₙ rₗₙ)², from one-hot
 * truth and predicted labels or probabilities, both as N × L matrices (pixels as rows, classes as columns). Classes
 * absent from the truth have infinite weight and are skipped.
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

/** A ground-truth object. `ignore` marks VOC "difficult" or COCO "crowd" objects, which count neither way. */
export type GroundTruthObject = { image: number | string; box: Box; label?: Label; ignore?: boolean }

/** A detection with its confidence. */
export type Detection = { image: number | string; box: Box; score: number; label?: Label }

/** Each detection's outcome after matching, in decreasing order of score. */
export type Match = {
  detection: number
  score: number
  /** True positive, false positive, or ignored (matched an ignored object). */
  outcome: 'tp' | 'fp' | 'ignored'
  /** Index of the matched ground-truth object, or −1. */
  object: number
  iou: number
}

/**
 * Match detections of one class to ground truth in decreasing order of confidence (mean-average-precision-for-
 * detection): each detection takes the unmatched ground-truth object in its image with the highest IoU ≥ `iouThreshold`
 * (COCO's rule), preferring non-ignored objects; a detection that matches nothing is a false positive. Labels are
 * ignored here; `detectionAveragePrecision` filters by class first.
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

/** How the precision envelope is integrated: every distinct recall (VOC 2010+), 11 points (VOC 2007) or 101 (COCO). */
export type ApInterpolation = 'all-points' | '11-point' | '101-point'

/**
 * Average precision from a list of detections already sorted by decreasing confidence, each a true or false positive,
 * and the number of ground-truth objects (mean-average-precision-for-detection): the area under the interpolated
 * precision P_interp(r) = max_{j: Rⱼ ≥ r} Pⱼ, exactly (`all-points`) or sampled at 11 or 101 recall levels.
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

/** Detection AP for one class (or for all objects when nothing is labelled), at one IoU threshold. */
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

/** The COCO IoU thresholds 0.50, 0.55, …, 0.95. */
export const cocoIouThresholds: readonly number[] = Array.from({ length: 10 }, (_, i) => 0.5 + 0.05 * i)

/**
 * Mean average precision for detection: AP averaged over classes (those with ground truth) and over IoU thresholds.
 * Defaults are COCO's: thresholds 0.50:0.05:0.95 and 101-point interpolation (Lin et al. 2014). For PASCAL VOC use
 * `{ iouThresholds: [0.5], interpolation: 'all-points' }` (Everingham et al. 2010).
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

/** PQ = SQ × RQ from matched IoUs and unmatched counts (Kirillov et al. 2019). */
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
 * Panoptic quality of two segment maps (panoptic-quality): every pixel carries a segment id, and `classOf` maps a
 * segment id to its class (default: one class). Within each class, a predicted and a ground-truth segment match when
 * IoU > 0.5, which makes the matching unique; PQ = Σ IoU/(TP + FP/2 + FN/2) per class, averaged over classes. Pixels
 * whose truth is `voidId` are removed from the predictions first, and predictions more than half void are ignored.
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
 * The boundary pixels of a binary mask of `height` × `width` (row-major): pixels inside the mask with a 4-neighbour
 * outside it or on the image edge. Returned as (row, column) points, a k × 2 matrix.
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
 * pixels within `tolerance` pixels (Euclidean, default 2) of a true boundary pixel, recall the reverse, and the score
 * their harmonic mean. Masks are `height × width`, as matrices or flat with `width`.
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

function boundaryPoints(mask: Data | Rows, width?: number): number[][] {
  const b = maskBoundary(mask, { width })
  const d = b.data as Float64Array
  return Array.from({ length: b.shape[0] }, (_, i) => [d[2 * i], d[2 * i + 1]])
}
