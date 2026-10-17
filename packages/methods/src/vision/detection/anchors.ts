/**
 * The box geometry of dense detectors: anchor grids, box offsets, anchor labelling by IoU, anchor-free (FCOS)
 * targets and the feature-pyramid level of a box.
 *
 * `anchorGrid` places one anchor per (size, aspect ratio) at the centre of every cell of a feature map of stride $s$.
 * `encodeBoxes` turns a box into the offsets $t_x = (x - x_a)/w_a$, $t_y = (y - y_a)/h_a$, $t_w = \log(w/w_a)$,
 * $t_h = \log(h/h_a)$ relative to an anchor of centre $(x_a, y_a)$ and size $w_a \times h_a$, and `decodeBoxes`
 * inverts it. `matchAnchors` labels each anchor positive, negative or ignored by its best IoU with the ground-truth
 * boxes, and by default also makes each ground-truth box's best anchor positive (Faster R-CNN, RetinaNet).
 * `anchorFreeTargets` gives, for points, FCOS's distances $(l, t, r, b)$ to the edges of the smallest box containing
 * the point, the regression-range test of the level, and the centre-ness
 * $\sqrt{\frac{\min(l, r)}{\max(l, r)} \frac{\min(t, b)}{\max(t, b)}}$. `pyramidLevel` is FPN's level
 * $k = \lfloor k_0 + \log_2(\sqrt{wh}/224) \rfloor$, clamped to the pyramid.
 *
 * Boxes are corners $[x_0, y_0, x_1, y_1]$ in continuous pixel coordinates, the `Box` of
 * `aifn-methods/evaluation/detection` (whose `boxIouMatrix` scores the overlaps). Cell $(i, j)$ of a stride-$s$ map is
 * centred at $((j + 1/2)s, (i + 1/2)s)$. Sources: Ren et al. (2015), Faster R-CNN, §3.1 (anchors, offsets, 0.7 and
 * 0.3 labels); Lin et al. (2017), FPN, eq. 1 and §4.1; Lin et al. (2017), RetinaNet, §4 (0.5 and 0.4 labels, an
 * ignored band); Tian et al. (2019), FCOS, §3.1 to §3.3. Box offsets carry no variance scaling. Plain arrays of boxes,
 * cheap enough to rematch a few thousand anchors on every drag.
 */

import { boxIouMatrix, type Box } from 'aifn-methods/evaluation/detection'
import { toFlat } from 'aifn-compute/foundation/tensor'

export type { Box }

/** Offsets $(t_x, t_y, t_w, t_h)$ of a box relative to an anchor. */
export type Offsets = readonly [number, number, number, number]

/** Options of `anchorGrid`. */
export type AnchorGridOptions = {
  /** Feature-map rows and columns. */
  featureShape: readonly [number, number]
  /** Input pixels per cell, $s$. */
  stride: number
  /** Square-root areas $\sqrt{wh}$ of the anchors, in pixels. */
  sizes: readonly number[]
  /** Aspect ratios $h/w$. */
  ratios: readonly number[]
}

/** The anchors of a feature map and where each comes from. */
export type AnchorGrid = {
  /** Anchors, cell by cell in row-major order, then size, then ratio. */
  anchors: Box[]
  /** For each anchor: its cell row, column, size index and ratio index. */
  index: { row: number; col: number; size: number; ratio: number }[]
}

/**
 * Anchors of every size and ratio centred on every cell of a stride-$s$ feature map: an anchor of size $a$ and ratio
 * $q$ is $a/\sqrt{q}$ wide and $a\sqrt{q}$ high, so its area is $a^2$.
 *
 * @param options The feature map's shape and stride, and the anchors' sizes and ratios.
 * @param options.featureShape The feature map's rows and columns.
 * @param options.stride Input pixels per cell, $s$.
 * @param options.sizes Square-root areas $\sqrt{wh}$ of the anchors, in pixels.
 * @param options.ratios Aspect ratios $h/w$.
 * @returns The anchors in row-major cell order (then size, then ratio), with each one's cell and indices.
 *
 * @example Three ratios at one size on a 1 x 2 map of stride 16
 * const g = anchorGrid({ featureShape: [1, 2], stride: 16, sizes: [32], ratios: [0.5, 1, 2] })
 * print('anchors =', g.anchors)
 * print('index =', g.index)
 */
export function anchorGrid({ featureShape, stride, sizes, ratios }: AnchorGridOptions): AnchorGrid {
  const anchors: Box[] = []
  const index: AnchorGrid['index'] = []
  const [rows, cols] = featureShape
  for (let row = 0; row < rows; row++)
    for (let col = 0; col < cols; col++) {
      const cx = (col + 0.5) * stride
      const cy = (row + 0.5) * stride
      sizes.forEach((a, size) =>
        ratios.forEach((q, ratio) => {
          const w = a / Math.sqrt(q)
          const h = a * Math.sqrt(q)
          anchors.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
          index.push({ row, col, size, ratio })
        }),
      )
    }
  return { anchors, index }
}

/**
 * A box as its centre and size.
 *
 * @param box The corners $[x_0, y_0, x_1, y_1]$.
 * @returns $[x, y, w, h]$.
 */
const centreSize = ([x0, y0, x1, y1]: Box) => [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0] as const

/**
 * The offsets of each box from the anchor at the same index (Ren et al., 2015, §3.1.2).
 *
 * @param anchors The anchors.
 * @param boxes The boxes, one per anchor.
 * @returns The offsets $(t_x, t_y, t_w, t_h)$, one per anchor.
 *
 * @example An anchor at (100, 100), 64 x 128, and a box at (110, 90), 80 x 120
 * print('offsets =', encodeBoxes([[68, 36, 132, 164]], [[70, 30, 150, 150]]))
 */
export function encodeBoxes(anchors: readonly Box[], boxes: readonly Box[]): Offsets[] {
  return anchors.map((a, i) => {
    const [xa, ya, wa, ha] = centreSize(a)
    const [x, y, w, h] = centreSize(boxes[i])
    return [(x - xa) / wa, (y - ya) / ha, Math.log(w / wa), Math.log(h / ha)] as const
  })
}

/**
 * Boxes from offsets relative to the anchor at the same index: the inverse of `encodeBoxes`.
 *
 * @param anchors The anchors.
 * @param offsets The offsets $(t_x, t_y, t_w, t_h)$, one per anchor.
 * @returns The boxes.
 *
 * @example Encoding then decoding gives the box back
 * const anchor = [68, 36, 132, 164]
 * print('decoded =', decodeBoxes([anchor], encodeBoxes([anchor], [[70, 30, 150, 150]])))
 */
export function decodeBoxes(anchors: readonly Box[], offsets: readonly Offsets[]): Box[] {
  return anchors.map((a, i) => {
    const [xa, ya, wa, ha] = centreSize(a)
    const [tx, ty, tw, th] = offsets[i]
    const x = xa + tx * wa
    const y = ya + ty * ha
    const w = wa * Math.exp(tw)
    const h = ha * Math.exp(th)
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2] as const
  })
}

/** Options of `matchAnchors`. */
export type MatchOptions = {
  /** IoU at or above which an anchor is positive (Faster R-CNN 0.7, RetinaNet 0.5). */
  positive: number
  /** IoU below which an anchor is negative (Faster R-CNN 0.3, RetinaNet 0.4); between the two it is ignored. */
  negative: number
  /** Also make each ground-truth box's highest-IoU anchor positive, so no object is left unmatched. Default true. */
  bestForEachTruth?: boolean
}

/** The labels of anchors matched against ground-truth boxes. */
export type AnchorMatch = {
  /** 1 positive, 0 negative, $-1$ ignored, per anchor. */
  labels: number[]
  /** Index of the ground-truth box each anchor overlaps most ($-1$ when there are none). */
  matched: number[]
  /** That IoU, per anchor. */
  iou: number[]
  /** For each ground-truth box, its highest-IoU anchor. */
  bestAnchor: number[]
}

/**
 * Label anchors against ground-truth boxes by IoU: positive at or above `positive`, negative below `negative`,
 * ignored between, and (by default) each ground-truth box's best anchor positive whatever its IoU, as Faster R-CNN
 * and RetinaNet.
 *
 * @param anchors The anchors.
 * @param truths The ground-truth boxes.
 * @param options The two IoU thresholds and the best-anchor rule.
 * @returns The labels, the matched box and IoU of every anchor, and every box's best anchor.
 *
 * @example Two anchors against one box
 * const m = matchAnchors([[0, 0, 10, 10], [5, 0, 15, 10]], [[0, 0, 10, 10]], { positive: 0.7, negative: 0.3 })
 * print('labels =', m.labels, ', iou =', m.iou)
 */
export function matchAnchors(anchors: readonly Box[], truths: readonly Box[], options: MatchOptions): AnchorMatch {
  const n = anchors.length
  const m = truths.length
  const labels = Array<number>(n).fill(0)
  const matched = Array<number>(n).fill(-1)
  const iou = Array<number>(n).fill(0)
  const bestAnchor = Array<number>(m).fill(-1)
  if (m === 0 || n === 0) return { labels, matched, iou, bestAnchor }
  const flat = toFlat(boxIouMatrix(anchors, truths))
  const best = Array<number>(m).fill(-Infinity)
  for (let i = 0; i < n; i++)
    for (let g = 0; g < m; g++) {
      const v = flat[i * m + g]
      if (v > iou[i] || matched[i] < 0) {
        iou[i] = v
        matched[i] = g
      }
      if (v > best[g]) {
        best[g] = v
        bestAnchor[g] = i
      }
    }
  for (let i = 0; i < n; i++) labels[i] = iou[i] >= options.positive ? 1 : iou[i] < options.negative ? 0 : -1
  if (options.bestForEachTruth !== false)
    bestAnchor.forEach((i, g) => {
      if (i >= 0 && best[g] > 0) {
        labels[i] = 1
        matched[i] = g
        iou[i] = best[g]
      }
    })
  return { labels, matched, iou, bestAnchor }
}

/** The FCOS target of one point. */
export type PointTarget = {
  /** Index of the ground-truth box the point regresses (the smallest containing it), or $-1$. */
  matched: number
  /** Distances to the left, top, right and bottom edges of that box. */
  ltrb: readonly [number, number, number, number]
  /** True when the point is inside a box and $\max(l, t, r, b)$ lies in the level's range. */
  positive: boolean
  /** FCOS centre-ness in $[0, 1]$ (0 when unmatched). */
  centreness: number
}

/**
 * FCOS targets for points (Tian et al., 2019, §3.1 to §3.3): each point strictly inside one or more boxes takes the
 * smallest such box, and is positive on this level when $m_{\text{lo}} \le \max(l, t, r, b) \le m_{\text{hi}}$.
 *
 * @param points The points $(x, y)$, usually the cell centres of one pyramid level.
 * @param truths The ground-truth boxes.
 * @param range The level's regression range $[m_{\text{lo}}, m_{\text{hi}}]$, in pixels.
 * @returns One target per point.
 *
 * @example A point in a wide box: its distances and centre-ness
 * print(anchorFreeTargets([[120, 88]], [[100, 60, 200, 100]], [0, 64]))
 */
export function anchorFreeTargets(
  points: readonly (readonly [number, number])[],
  truths: readonly Box[],
  range: readonly [number, number],
): PointTarget[] {
  return points.map(([x, y]) => {
    let matched = -1
    let area = Infinity
    truths.forEach(([x0, y0, x1, y1], g) => {
      const a = (x1 - x0) * (y1 - y0)
      if (x > x0 && x < x1 && y > y0 && y < y1 && a < area) {
        area = a
        matched = g
      }
    })
    if (matched < 0) return { matched, ltrb: [0, 0, 0, 0] as const, positive: false, centreness: 0 }
    const [x0, y0, x1, y1] = truths[matched]
    const ltrb = [x - x0, y - y0, x1 - x, y1 - y] as const
    const reach = Math.max(...ltrb)
    const [l, t, r, b] = ltrb
    const centreness = Math.sqrt((Math.min(l, r) / Math.max(l, r)) * (Math.min(t, b) / Math.max(t, b)))
    return { matched, ltrb, positive: reach >= range[0] && reach <= range[1], centreness }
  })
}

/**
 * FPN's pyramid level for a box of width $w$ and height $h$, $\lfloor k_0 + \log_2(\sqrt{wh}/224) \rfloor$
 * (Lin et al., 2017, eq. 1), clamped to the pyramid's levels.
 *
 * @param w The box's width, in pixels.
 * @param h The box's height, in pixels.
 * @param options The level of a canonical box and its size, and the levels of the pyramid.
 * @param options.k0 The level of a box of the canonical size.
 * @param options.canonical The canonical size $\sqrt{wh}$, ImageNet's 224.
 * @param options.min The finest level.
 * @param options.max The coarsest level.
 * @returns The level $k$.
 *
 * @example Halving the box moves it one level finer
 * for (const s of [448, 224, 112, 32]) print(`${s} x ${s}: P${pyramidLevel(s, s)}`)
 */
export function pyramidLevel(
  w: number,
  h: number,
  { k0 = 4, canonical = 224, min = 2, max = 5 }: { k0?: number; canonical?: number; min?: number; max?: number } = {},
): number {
  const k = Math.floor(k0 + Math.log2(Math.sqrt(w * h) / canonical))
  return Math.min(max, Math.max(min, k))
}
