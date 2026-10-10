/**
 * `aifn-methods/evaluation/detection`: detection and segmentation metrics: boxes, masks and panoptic quality.
 *
 * - Boxes: `boxIou` of two boxes, `boxIouMatrix` between two lists, and `boxOverlap`, which adds the GIoU, DIoU and
 *   CIoU variants used as regression losses.
 * - Masks and label maps: `maskIou` and `dice` of binary masks; `iouPerClass`, `meanIou` and `generalisedDice` of
 *   multi-class maps, the last weighting each class by the inverse square of its volume.
 * - Detection: `matchDetections` matches detections greedily by confidence at an IoU threshold;
 *   `averagePrecisionFromMatches` integrates the precision envelope (`ApInterpolation`: all points, 11 or 101);
 *   `detectionAveragePrecision` scores one class, and `detectionMeanAveragePrecision` averages over classes and the
 *   `cocoIouThresholds`.
 * - Panoptic segmentation: `panopticQuality` of two segment maps, and `panopticFromMatches` from matched IoUs and
 *   counts.
 * - Boundaries: `maskBoundary` lists a mask's boundary pixels, and `boundaryF1` scores two masks by them within a
 *   tolerance.
 *
 * Boxes (`Box`) are $[x_0, y_0, x_1, y_1]$ in continuous coordinates; masks are read flat, nonzero inside. A ratio
 * with a zero denominator is NaN. The metrics are collected in `evaluationMetricRegistry` of `aifn-methods/evaluation`,
 * and `detectionFunctions` registers the other functions.
 */

export {
  averagePrecisionFromMatches,
  boundaryF1,
  boxIou,
  boxIouMatrix,
  boxOverlap,
  cocoIouThresholds,
  detectionAveragePrecision,
  detectionMeanAveragePrecision,
  dice,
  generalisedDice,
  iouPerClass,
  maskBoundary,
  maskIou,
  matchDetections,
  meanIou,
  panopticFromMatches,
  panopticQuality,
  type ApInterpolation,
  type Box,
  type BoxOverlap,
  type Detection,
  type GroundTruthObject,
  type Match,
} from './detection'
export { detectionFunctions } from './registry'
