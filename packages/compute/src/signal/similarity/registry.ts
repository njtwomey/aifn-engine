/** The functions and algorithms of `aifn-compute/signal/similarity`, registered with the notes they serve. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as dtw from './dtw'
import * as mp from './matrix-profile'
import * as profile from './profile'
import * as sax from './sax'

const fn = definer<FunctionInfo>('function', 'signal/similarity')
const MP = ['matrix-profile', 'matrix-profile-algorithms']
const DP = ['distance-profile-and-mass']
const DTW = ['dynamic-time-warping']
const SAX = ['symbolic-aggregate-approximation']

fn(
  {
    key: 'distanceProfile',
    name: 'Distance profile (MASS)',
    tex: 'd_{i}^2 = 2m\\left(1 - \\frac{QT_i - m\\mu_q\\mu_i}{m\\sigma_q\\sigma_i}\\right)',
    summary: 'The z-normalised distance from a query to every subsequence, from FFT sliding dot products.',
    role: 'transform',
    notes: [...DP, 'matrix-profile'],
    cite: ['mueen-mass', 'yeh2016'],
  },
  profile.distanceProfile,
)
fn(
  {
    key: 'slidingDotProduct',
    name: 'Sliding dot product',
    summary: 'The dot product of a query with every window of a series, by one FFT convolution.',
    role: 'transform',
    notes: DP,
    cite: ['mueen-mass'],
  },
  profile.slidingDotProduct,
)
fn(
  { key: 'slidingMeanStd', name: 'Sliding mean and standard deviation', role: 'transform', notes: DP },
  profile.slidingMeanStd,
)
fn(
  {
    key: 'zDistance',
    name: 'z-normalised distance from a dot product',
    role: 'property',
    notes: DP,
    cite: ['yeh2016'],
  },
  profile.zDistance,
)
fn({ key: 'zNormalise', name: 'z-normalisation', role: 'transform', notes: [...DP, ...SAX] }, profile.zNormalise)
fn(
  {
    key: 'matrixProfile',
    name: 'Matrix profile (STOMP)',
    summary: 'Every subsequence’s nearest-neighbour distance and index, exact in O(n²) by diagonal updates.',
    role: 'solver',
    notes: MP,
    cite: ['yeh2016', 'zhu2016'],
  },
  mp.matrixProfile,
)
fn(
  {
    key: 'motifs',
    name: 'Matrix-profile motifs',
    summary: 'The closest pairs of subsequences, each excluding its neighbourhood from later picks.',
    role: 'property',
    notes: ['matrix-profile', 'motif-discovery-variants'],
    cite: ['yeh2016'],
  },
  mp.motifs,
)
fn(
  {
    key: 'discords',
    name: 'Matrix-profile discords',
    summary: 'The subsequences farthest from their nearest neighbours: the most unusual shapes.',
    role: 'property',
    notes: ['matrix-profile', 'discord-discovery-at-scale'],
    cite: ['yeh2016'],
  },
  mp.discords,
)
fn({ key: 'scrimpProfile', name: 'Matrix profile of a SCRIMP++ state', role: 'property', notes: MP }, mp.scrimpProfile)
fn(
  {
    key: 'dtw',
    name: 'Dynamic time warping',
    tex: 'D(i,j) = c(x_i, y_j) + \\min\\{D(i-1,j), D(i,j-1), D(i-1,j-1)\\}',
    summary: 'The cheapest monotone alignment of two series, with a Sakoe–Chiba band and the warping path.',
    role: 'solver',
    notes: DTW,
    cite: ['sakoe1978'],
  },
  dtw.dtw,
)
fn(
  {
    key: 'dtwProgram',
    name: 'DTW as a dynamic program',
    summary: 'The accumulated-cost recursion of DTW on the shared DP engine, so the table can be filled row by row.',
    role: 'construction',
    notes: DTW,
    cite: ['sakoe1978'],
  },
  dtw.dtwProgram,
)
fn(
  { key: 'keoghEnvelope', name: 'Keogh envelope', role: 'transform', notes: DTW, cite: ['keogh2005'] },
  dtw.keoghEnvelope,
)
fn(
  {
    key: 'lbKeogh',
    name: 'LB_Keogh',
    summary: 'A lower bound on banded DTW: the query’s distance outside the candidate’s envelope.',
    role: 'property',
    notes: DTW,
    cite: ['keogh2005', 'rakthanmanon2012'],
  },
  dtw.lbKeogh,
)
fn(
  {
    key: 'lbKim',
    name: 'LB_Kim',
    summary: 'An O(1) lower bound on DTW from the first, last, largest and smallest values.',
    role: 'property',
    notes: DTW,
    cite: ['rakthanmanon2012'],
  },
  dtw.lbKim,
)
fn(
  {
    key: 'paa',
    name: 'Piecewise aggregate approximation',
    summary: 'The means of w equal frames of a series.',
    role: 'transform',
    notes: SAX,
    cite: ['lin2003'],
  },
  sax.paa,
)
fn(
  { key: 'saxBreakpoints', name: 'SAX breakpoints', role: 'construction', notes: SAX, cite: ['lin2003'] },
  sax.saxBreakpoints,
)
fn(
  {
    key: 'sax',
    name: 'Symbolic aggregate approximation',
    summary: 'z-normalise, take the PAA, and map each mean to one of a equiprobable Gaussian regions.',
    role: 'transform',
    notes: SAX,
    cite: ['lin2003'],
  },
  sax.sax,
)
fn(
  {
    key: 'saxMinDist',
    name: 'SAX MINDIST',
    summary: 'A lower bound on the Euclidean distance between z-normalised series from their SAX words.',
    role: 'property',
    notes: SAX,
    cite: ['lin2003'],
  },
  sax.saxMinDist,
)

const algorithm = definer<AlgorithmInfo>('algorithm', 'signal/similarity')
algorithm(
  {
    key: 'scrimpSteps',
    name: 'SCRIMP++',
    summary: 'An anytime matrix profile: PreSCRIMP’s approximation, then exact diagonals in random order.',
    problem: 'map',
    state: { iterate: 'profile', flags: ['converged'] },
    random: true,
    notes: MP,
    cite: ['zhu2018'],
  },
  mp.scrimpSteps,
)

type Table<I extends FunctionInfo | AlgorithmInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>

/** The functions of the module, keyed by name. */
export const similarityFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  profile,
  mp,
  dtw,
  sax,
) as Table<FunctionInfo>
/** The algorithms of the module, keyed by factory name. */
export const similarityAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  mp,
) as Table<AlgorithmInfo>
