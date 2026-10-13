/**
 * A Type I error study of the tests that compare two learning algorithms, after Dietterich (1998): two learners of
 * equal expected accuracy are compared on many simulated data sets, and each test's rejection rate is its false
 * positive rate.
 *
 * The data are $n$ rows of two standard normal inputs $\xvec$, labelled 1 when $\lVert \xvec \rVert_2 + 0.3\,e$ ($e$
 * standard normal) exceeds $\sqrt{2 \ln 2}$, the median radius, so the class boundary is a circle and the classes are
 * about balanced. Learner A is a depth-4 decision tree on the inputs; learner B is the same tree on the inputs rotated
 * by $45^\circ$, plus Gaussian noise of standard deviation `handicap` on each rotated input. A circle looks the same
 * after rotation, so with no handicap the two learners have the same expected accuracy at every training size and the
 * null hypothesis of every test holds exactly; a handicap makes B worse, and the rates become power.
 *
 * On each data set the study runs McNemar's test on a random hold-out third (Dietterich, 1998), the paired $t$-test on
 * the $J = 10r$ accuracy differences of $r$ repeats of 10-fold cross-validation (naive) and on the first repeat alone,
 * the corrected resampled $t$-test of Nadeau and Bengio (2003) on the same differences, and Dietterich's $5 \times 2$cv
 * $t$-test with Alpaydin's (1999) combined $F$ test. Both learners always see the same splits.
 */

import { child, normals, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset, type Supervised } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold, repeated, shuffleSplit, type Splitter } from 'aifn-compute/learning/validate'
import { DomainError } from 'aifn-compute/foundation/errors'
import { correctedResampledTTest, fiveByTwoCvTest, mcnemarTest, oneSampleTTest } from 'aifn-compute/probability/tests'
import { decisionTree } from 'aifn-methods/learning/trees-and-ensembles'

/** The median of the radius of a standard bivariate normal, $\sqrt{2 \ln 2}$, so the classes are about balanced. */
const MEDIAN_RADIUS = 1.1774
const RADIUS_NOISE = 0.3
const FOLDS = 10
const FRESH = 2000
const learner = decisionTree({ maxDepth: 4 })

/** The tests of the study, in display order. */
export const TYPE_I_TESTS = ['mcnemar', 'cvT', 'resampledT', 'corrected', 'fiveByTwoT', 'combinedF'] as const

/**
 * A test of the study: `mcnemar` (McNemar on a hold-out third), `cvT` (paired $t$ on one 10-fold cross-validation),
 * `resampledT` (paired $t$ on $r$ repeats of it), `corrected` (the corrected resampled $t$ on the $r$ repeats),
 * `fiveByTwoT` ($5 \times 2$cv $t$) and `combinedF` ($5 \times 2$cv combined $F$).
 */
export type TypeITest = (typeof TYPE_I_TESTS)[number]

/** The tests' display labels, in the order of `TYPE_I_TESTS`. */
const LABELS = ['McNemar', 't, 10-fold', 't, r×10-fold', 'corrected t', '5×2cv t', '5×2cv F']

/** Options of `typeIErrorReplicate` and `typeIErrorStudy`. */
export interface TypeIErrorOptions {
  /** Rows per data set (default 100). */
  n?: number
  /** Repeats $r$ of 10-fold cross-validation, from 1 to 10 (default 10). */
  repeats?: number
  /** The noise standard deviation on B's inputs; 0 (the default) makes the learners equally good. */
  handicap?: number
}

/** One simulated data set: every test's $p$-value. */
export interface TypeIErrorReplicate {
  /** The $p$-value of each test. */
  p: Record<TypeITest, number>
  /** The naive paired $t$-test's $p$-value after $1, \dots, r$ repeats. */
  naiveByRepeats: number[]
  /** The corrected resampled $t$-test's $p$-value after $1, \dots, r$ repeats. */
  correctedByRepeats: number[]
  /** A's minus B's accuracy on 2000 fresh rows, each learner fitted to all $n$ rows. */
  trueDifference: number
}

/** The two learners' views of one data set. */
interface Pair {
  /** The data as learner A sees them. */
  a: Supervised<Tensor, Tensor>
  /** The same rows as learner B sees them. */
  b: Supervised<Tensor, Tensor>
}

/**
 * $n$ rows of the circle problem, as learners A and B see them.
 *
 * @param s The stream: inputs from its child `x`, radius noise from `e`, B's input noise from `z`.
 * @param n The number of rows.
 * @param handicap The noise standard deviation on B's rotated inputs.
 * @returns Both views, with the same labels.
 */
function drawPair(s: Stream, n: number, handicap: number): Pair {
  const x = toFlat(normals(child(s, 'x'), [n, 2]))
  const e = toFlat(normals(child(s, 'e'), [n]))
  const z = toFlat(normals(child(s, 'z'), [n, 2]))
  const y = Float64Array.from({ length: n }, (_, i) =>
    Math.hypot(x[2 * i], x[2 * i + 1]) + RADIUS_NOISE * e[i] > MEDIAN_RADIUS ? 1 : 0,
  )
  const r = Math.SQRT1_2
  const rotated = Float64Array.from({ length: 2 * n }, (_, t) => {
    const i = Math.floor(t / 2)
    const u = t % 2 === 0 ? r * (x[2 * i] - x[2 * i + 1]) : r * (x[2 * i] + x[2 * i + 1])
    return u + handicap * z[t]
  })
  const labels = fromData(y)
  return {
    a: dataset(fromData(Float64Array.from(x), [n, 2]), labels),
    b: dataset(fromData(rotated, [n, 2]), labels),
  }
}

/**
 * A's minus B's accuracy on each split of a splitter; both learners see the same splits.
 *
 * @param pair The data as each learner sees them.
 * @param splitter The resampling scheme.
 * @param s The stream the splits are drawn from (the same for both learners).
 * @returns One difference per split, in the splitter's order.
 */
function differences(pair: Pair, splitter: Splitter, s: Stream): number[] {
  const a = toFlat(crossValidate(learner, pair.a, splitter, [accuracy], { stream: s }).scores.accuracy)
  const b = toFlat(crossValidate(learner, pair.b, splitter, [accuracy], { stream: s }).scores.accuracy)
  return Array.from(a, (v, i) => v - b[i])
}

/**
 * McNemar's test of A against B on one random hold-out third of the rows, both trained on the other two thirds.
 *
 * @param pair The data as each learner sees them.
 * @param s The stream the split is drawn from.
 * @returns McNemar's test result.
 */
function holdoutMcnemar(pair: Pair, s: Stream) {
  const split = shuffleSplit({ splits: 1, testSize: 1 / 3 })
  const a = crossValidate(learner, pair.a, split, [accuracy], { stream: s })
  const b = crossValidate(learner, pair.b, split, [accuracy], { stream: s })
  const test = Array.from(toFlat(a.splits[0].test))
  const pa = toFlat(a.outOfFold.decide!)
  const pb = toFlat(b.outOfFold.decide!)
  const y = toFlat(pair.a.y)
  return mcnemarTest(
    test.map((i) => pa[i]),
    test.map((i) => pb[i]),
    test.map((i) => y[i]),
  )
}

/**
 * Check the options and fill in the defaults.
 *
 * @param options The options as given.
 * @param where The caller's name, for error messages.
 * @returns The rows, repeats and handicap.
 */
function resolve(options: TypeIErrorOptions, where: string): Required<TypeIErrorOptions> {
  const { n = 100, repeats = 10, handicap = 0 } = options
  if (!(Number.isInteger(n) && n >= 20)) throw new DomainError(where, `${where}: n must be an integer >= 20, got ${n}`)
  if (!(Number.isInteger(repeats) && repeats >= 1 && repeats <= 10))
    throw new DomainError(where, `${where}: repeats must be an integer from 1 to 10, got ${repeats}`)
  if (!(handicap >= 0)) throw new DomainError(where, `${where}: handicap must be non-negative, got ${handicap}`)
  return { n, repeats, handicap }
}

/**
 * Every test on one simulated data set (see the file's introduction): the data from `child(s, 'data')`, the $r$
 * repeats of 10-fold cross-validation from `child(s, 'cv')`, the $5 \times 2$ folds from `child(s, '5x2')`, McNemar's
 * hold-out third from `child(s, 'holdout')` and the fresh rows from `child(s, 'fresh')`. Costs $20r + 24$ tree fits.
 * Throws `DomainError` for a bad option.
 *
 * @param s The stream; not advanced.
 * @param options The rows, repeats and handicap (see `TypeIErrorOptions`).
 * @returns The $p$-values, the naive and corrected $p$-values by number of repeats, and the true difference.
 *
 * @example One data set: the p-values of the six tests
 * const r = typeIErrorReplicate(stream(0), { n: 60, repeats: 2 })
 * print('p-values:', r.p)
 * print('true accuracy difference:', r.trueDifference)
 */
export function typeIErrorReplicate(s: Stream, options: TypeIErrorOptions = {}): TypeIErrorReplicate {
  const { n, repeats, handicap } = resolve(options, 'typeIErrorReplicate')
  const pair = drawPair(child(s, 'data'), n, handicap)
  const trainSize = (n * (FOLDS - 1)) / FOLDS
  const testSize = n / FOLDS
  const cv = differences(pair, repeated(kFold({ k: FOLDS, shuffle: true }), repeats), child(s, 'cv'))
  const naiveByRepeats: number[] = []
  const correctedByRepeats: number[] = []
  for (let r = 1; r <= repeats; r++) {
    const d = cv.slice(0, r * FOLDS)
    naiveByRepeats.push(oneSampleTTest(d).pValue)
    correctedByRepeats.push(correctedResampledTTest(d, { trainSize, testSize }).pValue)
  }
  const f = differences(pair, repeated(kFold({ k: 2, shuffle: true }), 5), child(s, '5x2'))
  const five = fiveByTwoCvTest([0, 1, 2, 3, 4].map((i) => [f[2 * i], f[2 * i + 1]]))
  const fresh = drawPair(child(s, 'fresh'), FRESH, handicap)
  const accA = accuracy(fresh.a.y, learner.fit(pair.a).decide(fresh.a.x))
  const accB = accuracy(fresh.b.y, learner.fit(pair.b).decide(fresh.b.x))
  return {
    p: {
      mcnemar: holdoutMcnemar(pair, child(s, 'holdout')).pValue,
      cvT: naiveByRepeats[0],
      resampledT: naiveByRepeats[repeats - 1],
      corrected: correctedByRepeats[repeats - 1],
      fiveByTwoT: five.pValue,
      combinedF: five.combinedF.pValue,
    },
    naiveByRepeats,
    correctedByRepeats,
    trueDifference: accA - accB,
  }
}

/** The study over many data sets. */
export interface TypeIErrorStudy {
  /** The number of simulated data sets. */
  replicates: number
  /** The level $\alpha$ of the rejection rates. */
  alpha: number
  /** The tests, in display order (`TYPE_I_TESTS`). */
  tests: TypeITest[]
  /** Their display labels, in the same order. */
  labels: string[]
  /** Each test's $p$-value on every data set. */
  pValues: Record<TypeITest, Float64Array>
  /** How many data sets each test rejects on ($p \le \alpha$). */
  rejections: Record<TypeITest, number>
  /** Each test's rejection rate: `rejections` over `replicates`. */
  rates: Record<TypeITest, number>
  /**
   * The rejection rates of the naive paired $t$-test and of the corrected resampled $t$-test after
   * $1, \dots, r$ repeats of 10-fold cross-validation.
   */
  byRepeats: { repeats: number[]; naive: Float64Array; corrected: Float64Array }
  /** A's minus B's accuracy on fresh data, on every data set. */
  trueDifferences: Float64Array
  /** Their mean: 0 in expectation with no handicap. */
  trueDifference: number
}

/** Options of `typeIErrorStudy`. */
export interface TypeIErrorStudyOptions extends TypeIErrorOptions {
  /** The number of simulated data sets (default 40). */
  replicates?: number
  /** The level $\alpha$ (default 0.05). */
  alpha?: number
}

/**
 * The study's summary over the replicates run so far.
 *
 * @param rows The replicates run so far.
 * @param repeats The repeats $r$.
 * @param alpha The level $\alpha$.
 * @returns The study over `rows`.
 */
function summarise(rows: TypeIErrorReplicate[], repeats: number, alpha: number): TypeIErrorStudy {
  const count = rows.length
  const pValues = {} as Record<TypeITest, Float64Array>
  const rejections = {} as Record<TypeITest, number>
  const rates = {} as Record<TypeITest, number>
  for (const t of TYPE_I_TESTS) {
    pValues[t] = Float64Array.from(rows, (r) => r.p[t])
    rejections[t] = rows.filter((r) => r.p[t] <= alpha).length
    rates[t] = rejections[t] / count
  }
  const rs = Array.from({ length: repeats }, (_, i) => i + 1)
  const trueDifferences = Float64Array.from(rows, (r) => r.trueDifference)
  return {
    replicates: count,
    alpha,
    tests: [...TYPE_I_TESTS],
    labels: [...LABELS],
    pValues,
    rejections,
    rates,
    byRepeats: {
      repeats: rs,
      naive: Float64Array.from(rs, (r) => rows.filter((row) => row.naiveByRepeats[r - 1] <= alpha).length / count),
      corrected: Float64Array.from(
        rs,
        (r) => rows.filter((row) => row.correctedByRepeats[r - 1] <= alpha).length / count,
      ),
    },
    trueDifferences,
    trueDifference: trueDifferences.reduce((a, b) => a + b, 0) / count,
  }
}

/**
 * The study one data set at a time: yields the summary after each replicate, so a worker can stream it (a page shows
 * the rates fill in). Replicate $i$ draws from `child(s, i)`, as `typeIErrorStudy`.
 *
 * @param s The stream; not advanced.
 * @param options The rows, repeats, handicap, number of data sets and level (see `TypeIErrorStudyOptions`).
 * @returns A generator of the study over the first $1, 2, \dots$ replicates; its last value is `typeIErrorStudy`'s.
 *
 * @example The rates after each of three data sets
 * for (const study of typeIErrorStudySteps(stream(0), { n: 60, repeats: 1, replicates: 3 }))
 *   print(study.replicates, 'data sets: McNemar rejects', study.rejections.mcnemar)
 */
export function* typeIErrorStudySteps(s: Stream, options: TypeIErrorStudyOptions = {}): Generator<TypeIErrorStudy> {
  const { replicates = 40, alpha = 0.05 } = options
  const { repeats } = resolve(options, 'typeIErrorStudy')
  if (!(Number.isInteger(replicates) && replicates >= 1))
    throw new DomainError(
      'typeIErrorStudy',
      `typeIErrorStudy: replicates must be a positive integer, got ${replicates}`,
    )
  const rows: TypeIErrorReplicate[] = []
  for (let r = 0; r < replicates; r++) {
    rows.push(typeIErrorReplicate(child(s, r), options))
    yield summarise(rows, repeats, alpha)
  }
}

/**
 * The Type I error study (Dietterich, 1998) over many simulated data sets (see the file's introduction): each test's
 * $p$-value on every data set, its rejection rate at level $\alpha$, the naive and corrected rates by number of
 * repeats, and the true accuracy difference. Replicate $i$ draws from `child(s, i)`. Costs about $(20r + 24)$ tree fits
 * per data set. With no handicap a valid test rejects at most a share $\alpha$ of the data sets, up to Monte Carlo
 * error; the naive $t$-test on $r$ repeats rejects more the larger $r$.
 *
 * @param s The stream; not advanced.
 * @param options The rows, repeats, handicap, number of data sets and level (see `TypeIErrorStudyOptions`).
 * @returns The study (see `TypeIErrorStudy`).
 *
 * @example Six tests on four data sets of equally good learners
 * const study = typeIErrorStudy(stream(0), { n: 60, repeats: 2, replicates: 4 })
 * print('rejection rates:', study.rates)
 * print('naive and corrected by repeats:', study.byRepeats.naive, study.byRepeats.corrected)
 */
export function typeIErrorStudy(s: Stream, options: TypeIErrorStudyOptions = {}): TypeIErrorStudy {
  let last: TypeIErrorStudy | undefined
  for (const study of typeIErrorStudySteps(s, options)) last = study
  return last!
}
