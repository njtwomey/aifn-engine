/** The functions of `aifn-compute/signal/image`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as blobs from './blobs'
import * as corners from './corners'
import * as edges from './edges'
import * as filters from './filters'
import * as hough from './hough'
import * as morphology from './morphology'
import * as pyramids from './pyramids'

const fn = definer<FunctionInfo>('function', 'signal/image')
const FILTER = ['image-filtering']
const EDGE = ['edge-detection']
const FEATURE = ['feature-detection-and-descriptors']
const HOUGH = ['hough-transform']
const MORPH = ['morphological-operations']
const PYR = ['image-pyramids-and-scale-space']

// ── Linear filters ───────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'gaussianKernel', name: 'Gaussian kernel', role: 'construction', notes: FILTER, cite: ['lindeberg1994'] },
  filters.gaussianKernel,
)
fn(
  {
    key: 'gaussianBlur',
    name: 'Gaussian blur',
    role: 'transform',
    notes: [...FILTER, ...PYR],
    cite: ['lindeberg1994'],
  },
  filters.gaussianBlur,
)
fn(
  {
    key: 'gradients',
    name: 'Image gradients (Sobel, Scharr, Prewitt)',
    role: 'transform',
    notes: [...EDGE, ...FILTER],
    cite: ['gonzalez1985'],
  },
  filters.gradients,
)
fn(
  { key: 'sobel', name: 'Sobel gradient', role: 'transform', notes: [...EDGE, ...FILTER], cite: ['gonzalez1985'] },
  filters.sobel,
)
fn(
  {
    key: 'structureTensor',
    name: 'Structure tensor',
    summary: 'The Gaussian-weighted second-moment matrix of the image gradient at each pixel.',
    role: 'transform',
    notes: FEATURE,
    cite: ['harris1988'],
  },
  filters.structureTensor,
)
fn(
  {
    key: 'gaussianLaplace',
    name: 'Laplacian of Gaussian (filter)',
    role: 'transform',
    notes: [...EDGE, ...PYR],
    cite: ['marr1980'],
  },
  filters.gaussianLaplace,
)

// ── Features ─────────────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'canny',
    name: 'Canny edge detector',
    summary:
      'Smooth, differentiate, keep local maxima across the edge, then link by hysteresis between two thresholds.',
    role: 'transform',
    notes: EDGE,
    cite: ['canny1986'],
  },
  edges.canny,
)
fn(
  {
    key: 'harrisResponse',
    name: 'Harris corner response',
    tex: 'R = \\det M - k\\,(\\operatorname{tr} M)^2',
    role: 'transform',
    notes: FEATURE,
    cite: ['harris1988'],
  },
  corners.harrisResponse,
)
fn(
  {
    key: 'shiTomasiResponse',
    name: 'Shi–Tomasi corner response',
    tex: '\\lambda_{\\min}(M)',
    role: 'transform',
    notes: FEATURE,
    cite: ['shi1994'],
  },
  corners.shiTomasiResponse,
)
fn(
  { key: 'imagePeaks', name: 'Image peaks (non-maximum suppression)', role: 'estimator', notes: FEATURE },
  corners.imagePeaks,
)
fn(
  {
    key: 'detectCorners',
    name: 'Corner detection',
    role: 'estimator',
    notes: FEATURE,
    cite: ['harris1988', 'shi1994'],
  },
  corners.detectCorners,
)
fn(
  {
    key: 'blobsLog',
    name: 'Blobs by the Laplacian of Gaussian',
    summary: 'Local maxima of the scale-normalised LoG over position and scale.',
    role: 'estimator',
    notes: [...FEATURE, ...PYR],
    cite: ['lindeberg1998'],
  },
  blobs.blobsLog,
)
fn(
  {
    key: 'blobsDog',
    name: 'Blobs by the difference of Gaussians',
    role: 'estimator',
    notes: [...FEATURE, ...PYR],
    cite: ['lowe2004'],
  },
  blobs.blobsDog,
)
fn(
  {
    key: 'houghLines',
    name: 'Hough transform for lines',
    tex: '\\rho = x\\cos\\theta + y\\sin\\theta',
    role: 'transform',
    notes: HOUGH,
    cite: ['duda1972', 'hough1962'],
  },
  hough.houghLines,
)
fn(
  { key: 'houghLinePeaks', name: 'Hough line peaks', role: 'estimator', notes: HOUGH, cite: ['duda1972'] },
  hough.houghLinePeaks,
)
fn(
  { key: 'houghCircles', name: 'Hough transform for circles', role: 'transform', notes: HOUGH, cite: ['duda1972'] },
  hough.houghCircles,
)
fn({ key: 'houghCirclePeaks', name: 'Hough circle peaks', role: 'estimator', notes: HOUGH }, hough.houghCirclePeaks)

// ── Morphology ───────────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'erode', name: 'Erosion', role: 'transform', notes: MORPH, cite: ['serra1982', 'soille2004'] },
  morphology.erode,
)
fn(
  { key: 'dilate', name: 'Dilation', role: 'transform', notes: MORPH, cite: ['serra1982', 'soille2004'] },
  morphology.dilate,
)
fn({ key: 'opening', name: 'Opening', role: 'transform', notes: MORPH, cite: ['soille2004'] }, morphology.opening)
fn({ key: 'closing', name: 'Closing', role: 'transform', notes: MORPH, cite: ['soille2004'] }, morphology.closing)
fn(
  {
    key: 'morphologicalGradient',
    name: 'Morphological gradient',
    role: 'transform',
    notes: MORPH,
    cite: ['soille2004'],
  },
  morphology.morphologicalGradient,
)
fn({ key: 'topHat', name: 'White top-hat', role: 'transform', notes: MORPH, cite: ['soille2004'] }, morphology.topHat)
fn(
  { key: 'squareElement', name: 'Square structuring element', role: 'construction', notes: MORPH },
  morphology.squareElement,
)
fn({ key: 'discElement', name: 'Disc structuring element', role: 'construction', notes: MORPH }, morphology.discElement)

// ── Pyramids ─────────────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'pyramidReduce', name: 'Pyramid reduce', role: 'transform', notes: PYR, cite: ['burt1983'] },
  pyramids.pyramidReduce,
)
fn(
  { key: 'pyramidExpand', name: 'Pyramid expand', role: 'transform', notes: PYR, cite: ['burt1983'] },
  pyramids.pyramidExpand,
)
fn(
  { key: 'gaussianPyramid', name: 'Gaussian pyramid', role: 'transform', notes: PYR, cite: ['burt1983'] },
  pyramids.gaussianPyramid,
)
fn(
  { key: 'laplacianPyramid', name: 'Laplacian pyramid', role: 'transform', notes: PYR, cite: ['burt1983'] },
  pyramids.laplacianPyramid,
)
fn(
  {
    key: 'reconstructLaplacian',
    name: 'Reconstruct from a Laplacian pyramid',
    role: 'transform',
    notes: PYR,
    cite: ['burt1983'],
  },
  pyramids.reconstructLaplacian,
)

/** The functions of the module, keyed by name. */
export const imageFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', filters, edges, corners, blobs, hough, morphology, pyramids) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
