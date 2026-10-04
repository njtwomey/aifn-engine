/** The functions of `aifn-methods/evaluation/detection` besides its metrics. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as detection from './detection'

const fn = definer<FunctionInfo>('function', 'evaluation/detection')
const MAP = ['mean-average-precision-for-detection']

fn(
  { key: 'boxIouMatrix', name: 'Box IoU matrix', role: 'estimator', notes: ['intersection-over-union'] },
  detection.boxIouMatrix,
)
fn(
  { key: 'boxOverlap', name: 'Box overlap', role: 'estimator', notes: ['intersection-over-union'] },
  detection.boxOverlap,
)
fn(
  {
    key: 'iouPerClass',
    name: 'IoU per class',
    role: 'estimator',
    notes: ['intersection-over-union', 'dice-and-segmentation-overlap'],
  },
  detection.iouPerClass,
)
fn(
  {
    key: 'matchDetections',
    name: 'Match detections to ground truth',
    summary: 'Greedy matching by score at an IoU threshold, as PASCAL VOC and COCO.',
    role: 'solver',
    notes: MAP,
    cite: ['everingham2010', 'lin2014'],
  },
  detection.matchDetections,
)
fn(
  { key: 'averagePrecisionFromMatches', name: 'Average precision from matches', role: 'estimator', notes: MAP },
  detection.averagePrecisionFromMatches,
)
fn(
  {
    key: 'detectionAveragePrecision',
    name: 'Detection average precision',
    role: 'estimator',
    notes: MAP,
    cite: ['lin2014'],
  },
  detection.detectionAveragePrecision,
)
fn(
  {
    key: 'panopticFromMatches',
    name: 'Panoptic quality from matches',
    role: 'estimator',
    notes: ['panoptic-quality'],
    cite: ['kirillov2019'],
  },
  detection.panopticFromMatches,
)
fn(
  {
    key: 'panopticQuality',
    name: 'Panoptic quality with its parts',
    role: 'estimator',
    notes: ['panoptic-quality'],
    cite: ['kirillov2019'],
  },
  detection.panopticQuality,
)
fn(
  { key: 'maskBoundary', name: 'Mask boundary', role: 'transform', notes: ['hausdorff-distance'] },
  detection.maskBoundary,
)

/** The functions of the module that are not metrics, keyed by name. */
export const detectionFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', detection) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
