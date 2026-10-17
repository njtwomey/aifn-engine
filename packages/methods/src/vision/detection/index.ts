/**
 * `aifn-methods/vision/detection`: the box geometry of dense object detectors, from anchors to training targets.
 *
 * - Anchors: `anchorGrid` places anchors of every size and aspect ratio on every cell of a feature map.
 * - Box offsets: `encodeBoxes` and its inverse `decodeBoxes`, the $(t_x, t_y, t_w, t_h)$ parameterisation of Faster
 *   R-CNN.
 * - Training targets: `matchAnchors` labels anchors positive, negative or ignored by IoU (with each object's best
 *   anchor kept positive), and `anchorFreeTargets` gives FCOS's edge distances, level test and centre-ness for
 *   points.
 * - Feature pyramids: `pyramidLevel`, FPN's level for a box of a given size.
 *
 * Boxes are the `Box` of `aifn-methods/evaluation/detection`, corners $[x_0, y_0, x_1, y_1]$ in pixels, and IoUs
 * are its `boxIouMatrix`; the detection metrics (average precision, matching detections to truth) are that module's.
 * Everything is plain arrays of boxes. `anchorFunctions` is the module's registry.
 */

export {
  anchorFreeTargets,
  anchorGrid,
  decodeBoxes,
  encodeBoxes,
  matchAnchors,
  pyramidLevel,
  type AnchorGrid,
  type AnchorGridOptions,
  type AnchorMatch,
  type Box,
  type MatchOptions,
  type Offsets,
  type PointTarget,
} from './anchors'
export { anchorFunctions } from './registry'
