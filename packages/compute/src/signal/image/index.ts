/**
 * `aifn-compute/signal/image`: classical image processing on greyscale images ([height, width] tensors, row 0 at the top), on
 * the 2-D filtering of `aifn-compute/foundation/convolution`.
 *
 * - Linear filters: `gaussianKernel`, `gaussianBlur`, `gradients` (Sobel, Scharr, Prewitt), `sobel`,
 *   `gaussianLaplace` (the Laplacian of Gaussian, as scipy), `structureTensor`.
 * - Edges: `canny` (every stage: smoothed, gradient, suppressed, strong, weak, edges).
 * - Corners: `harrisResponse`, `shiTomasiResponse`, `imagePeaks` (non-maximum suppression), `detectCorners`.
 * - Blobs: `blobsLog` (scale-normalised LoG), `blobsDog` (difference of Gaussians).
 * - Hough: `houghLines` + `houghLinePeaks`, `houghCircles` + `houghCirclePeaks`.
 * - Morphology: `erode`, `dilate`, `opening`, `closing`, `morphologicalGradient`, `topHat`; `squareElement`,
 *   `discElement`.
 * - Pyramids: `pyramidReduce`, `pyramidExpand`, `gaussianPyramid`, `laplacianPyramid`, `reconstructLaplacian`.
 */

export {
  gaussianBlur,
  gaussianKernel,
  gradients,
  gaussianLaplace,
  sobel,
  structureTensor,
  type GradientOperator,
  type Gradients,
  type StructureTensor,
} from './filters'
export { canny, type Canny, type CannyOptions } from './edges'
export {
  detectCorners,
  harrisResponse,
  imagePeaks,
  shiTomasiResponse,
  type CornerOptions,
  type ImagePeak,
  type PeakOptions,
} from './corners'
export { blobsDog, blobsLog, type Blob, type BlobOptions } from './blobs'
export {
  houghCirclePeaks,
  houghCircles,
  houghLinePeaks,
  houghLines,
  type HoughCircle,
  type HoughLine,
  type HoughLines,
  type HoughPeakOptions,
} from './hough'
export {
  closing,
  dilate,
  discElement,
  erode,
  morphologicalGradient,
  opening,
  squareElement,
  topHat,
  type MorphologyOptions,
  type StructuringElement,
} from './morphology'
export {
  gaussianPyramid,
  laplacianPyramid,
  pyramidExpand,
  pyramidReduce,
  reconstructLaplacian,
  type PyramidOptions,
} from './pyramids'
export { imageFunctions } from './registry'
