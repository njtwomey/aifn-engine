/** The registry of `aifn-methods/vision/two-view`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as scene from './scene'

const fn = definer<FunctionInfo>('function', 'vision/two-view')

fn(
  {
    key: 'twoViewScene',
    name: 'Synthetic two-view scene',
    summary:
      'Two pinhole cameras, scene points on a plane or in depth, noisy correspondences with outliers, and the true H and F.',
    role: 'simulation',
    random: true,
    notes: ['epipolar-geometry-and-stereo', 'homography', 'pinhole-camera-model'],
    cite: ['hartley2004'],
  },
  scene.twoViewScene,
)
fn(
  {
    key: 'homographyProblem',
    name: 'Homography RANSAC problem',
    role: 'construction',
    notes: ['homography'],
    cite: ['fischler1981'],
  },
  scene.homographyProblem,
)
fn(
  {
    key: 'fundamentalProblem',
    name: 'Fundamental-matrix RANSAC problem',
    role: 'construction',
    notes: ['epipolar-geometry-and-stereo'],
    cite: ['fischler1981', 'hartley1997'],
  },
  scene.fundamentalProblem,
)

/** The functions of the module, keyed by name. */
export const twoViewFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', scene) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
