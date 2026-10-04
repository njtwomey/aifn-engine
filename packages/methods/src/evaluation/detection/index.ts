/**
 * `aifn-methods/evaluation/detection`: detection and segmentation metrics: boxes, masks and panoptic quality.
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
