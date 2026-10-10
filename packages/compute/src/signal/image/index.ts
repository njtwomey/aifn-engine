/**
 * `aifn-compute/signal/image`: classical image processing on greyscale images, as scipy.ndimage and scikit-image.
 *
 * - Linear filters: `gaussianKernel`, `gaussianBlur`, `gradients` (Sobel, Scharr, Prewitt), `sobel`,
 *   `gaussianLaplace` (the Laplacian of Gaussian, as scipy), and `structureTensor`, the second-moment matrix the
 *   corner detectors read.
 * - Edges: `canny`, which returns every stage (smoothed, gradient, suppressed, strong, weak, edges).
 * - Corners: `harrisResponse` and `shiTomasiResponse` from the structure tensor, `imagePeaks` (non-maximum suppression)
 *   and `detectCorners`, the two together.
 * - Blobs: `blobsLog` (scale-normalised LoG, finer scales) and `blobsDog` (difference of Gaussians, faster).
 * - Hough: `houghLines` then `houghLinePeaks`, `houghCircles` then `houghCirclePeaks`.
 * - Morphology with a flat structuring element (`squareElement`, `discElement`): `erode`, `dilate`, `opening`,
 *   `closing`, `morphologicalGradient`, `topHat`.
 * - Pyramids: `pyramidReduce` and `pyramidExpand`, `gaussianPyramid`, `laplacianPyramid` and its exact inverse
 *   `reconstructLaplacian`.
 * - `imageFunctions`: the module's functions with the notes and citations that define them.
 *
 * An image is an $h \times w$ tensor or an array of rows, row 0 at the top; positions are (row, column), and $x$
 * means the column. Filters correlate on the 2-D filtering of `aifn-compute/foundation/convolution` and read beyond the
 * edge with scipy.ndimage's border modes. Detections (peaks, blobs, lines, circles) are listed strongest first.
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
