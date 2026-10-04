/** The functions and algorithms of `aifn-compute/numerics/robust`, registered with the notes they serve. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as ransac from './ransac'

const NOTES = ['homography', 'epipolar-geometry-and-stereo', 'structure-from-motion']

const algorithm = definer<AlgorithmInfo>('algorithm', 'numerics/robust')
algorithm(
  {
    key: 'ransac',
    name: 'RANSAC',
    summary: 'Fit minimal random samples, keep the model with most inliers, stop at the adaptive sample count.',
    problem: 'least-squares',
    state: { iterate: 'best', objective: 'inlierCount', flags: ['terminated'] },
    random: true,
    notes: NOTES,
    cite: ['fischler1981'],
  },
  ransac.ransac,
)

const fn = definer<FunctionInfo>('function', 'numerics/robust')
fn({ key: 'ransacFit', name: 'RANSAC fit', role: 'fit', notes: NOTES, cite: ['fischler1981'] }, ransac.ransacFit)
fn(
  {
    key: 'ransacTrials',
    name: 'RANSAC sample count',
    tex: 'N = \\left\\lceil \\frac{\\log(1 - p)}{\\log(1 - w^s)} \\right\\rceil',
    role: 'property',
    notes: NOTES,
    cite: ['fischler1981', 'hartley2004'],
  },
  ransac.ransacTrials,
)

/** The functions of the module, keyed by name. */
export const robustFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', ransac) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

/** The algorithms of the module, keyed by factory name. */
export const robustAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', ransac) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
