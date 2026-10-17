/**
 * The registry of `aifn-methods/vision/detection`: anchor grids, box offsets, anchor matching, anchor-free targets
 * and pyramid levels, as functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as anchors from './anchors'

const fn = definer<FunctionInfo>('function', 'vision/detection')
const HEADS = 'detection-heads-and-anchors'

fn(
  {
    key: 'anchorGrid',
    name: 'Anchor grid',
    summary: 'Anchors of every size and aspect ratio centred on every cell of a stride-s feature map.',
    role: 'construction',
    notes: [HEADS, 'feature-pyramid-network'],
    cite: ['ren2015', 'lin2017fpn'],
  },
  anchors.anchorGrid,
)
fn(
  {
    key: 'encodeBoxes',
    name: 'Box offsets from anchors',
    summary:
      'The offsets (tx, ty, tw, th) of each box from its anchor: centre shifts in anchor units, log size ratios.',
    tex: 't_x = (x - x_a)/w_a,\\ t_w = \\log(w/w_a)',
    role: 'transform',
    notes: [HEADS],
    cite: ['ren2015'],
  },
  anchors.encodeBoxes,
)
fn(
  {
    key: 'decodeBoxes',
    name: 'Boxes from anchor offsets',
    summary: 'The inverse of encodeBoxes: boxes from offsets relative to their anchors.',
    tex: 'x = x_a + t_x w_a,\\ w = w_a e^{t_w}',
    role: 'transform',
    notes: [HEADS],
    cite: ['ren2015'],
  },
  anchors.decodeBoxes,
)
fn(
  {
    key: 'matchAnchors',
    name: 'Anchor matching by IoU',
    summary: 'Anchors labelled positive, negative or ignored by their best IoU, with each object’s best anchor kept.',
    role: 'transform',
    notes: [HEADS],
    cite: ['ren2015', 'lin2017'],
  },
  anchors.matchAnchors,
)
fn(
  {
    key: 'anchorFreeTargets',
    name: 'FCOS point targets',
    summary:
      'Distances to the edges of the smallest containing box, the level’s range test and centre-ness, per point.',
    tex: '\\sqrt{\\frac{\\min(l, r)}{\\max(l, r)} \\frac{\\min(t, b)}{\\max(t, b)}}',
    role: 'transform',
    notes: [HEADS],
    cite: ['tian2019'],
  },
  anchors.anchorFreeTargets,
)
fn(
  {
    key: 'pyramidLevel',
    name: 'Feature-pyramid level of a box',
    summary: 'FPN’s level for a w × h box, ⌊k₀ + log₂(√(wh)/224)⌋, clamped to the pyramid.',
    tex: 'k = \\lfloor k_0 + \\log_2(\\sqrt{wh}/224) \\rfloor',
    role: 'property',
    notes: ['feature-pyramid-network'],
    cite: ['lin2017fpn'],
  },
  anchors.pyramidLevel,
)

/** The registered functions of `aifn-methods/vision/detection`, keyed by `info.key`. */
export const anchorFunctions: Readonly<Record<string, Entry<unknown, FunctionInfo>>> = entries('function', anchors)
