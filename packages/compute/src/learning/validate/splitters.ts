/**
 * Splitters (plan §5.5): pure functions of a dataset's size, labels and groups and a stream, returning train and test
 * index sets. Non-random splitters agree with scikit-learn's `KFold`, `StratifiedKFold`, `GroupKFold`, `LeaveOneOut`
 * and `TimeSeriesSplit` on the same indices; random ones draw only from the stream they are given.
 */

import { rowCount, type Column, type Dataset, type Features } from 'aifn-compute/learning/estimators'
import { child, permutation, shuffle, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** What a splitter reads: the number of rows, and labels (stratified) or groups (grouped) where it needs them. */
export interface SplitInput {
  n: number
  /** Class labels, for stratification. */
  y?: Column
  /** Group labels: rows of a group are never split between train and test. */
  groups?: Column
}

/** One split: sorted int32 row indices of the training and test sets. Rows in neither are unused. */
export interface Split {
  train: Tensor
  test: Tensor
}

/** A splitter: a name, whether it needs a stream, and the split function. */
export interface Splitter {
  readonly name: string
  /** True when `split` draws from its stream (and throws without one). */
  readonly randomised: boolean
  split(data: SplitInput | Dataset<Features, Column>, s?: Stream): Split[]
}

/** The `assignment` code of a training row. */
export const TRAIN = 0
/** The `assignment` code of a test row. */
export const TEST = 1
/** The `assignment` code of a row a split leaves out. */
export const UNUSED = -1

function inputOf(data: SplitInput | Dataset<Features, Column>): SplitInput {
  if ('n' in data && typeof data.n === 'number') return data as SplitInput
  const d = data as Dataset<Features, Column>
  return { n: rowCount(d.x), y: d.y, groups: d.groups }
}

function labelsOf(c: Column, n: number, what: string): (string | number)[] {
  const out = Array.isArray(c) ? Array.from(c) : toFlat(c as Tensor)
  if (out.length !== n) throw new ShapeError('splitter', `splitter: ${out.length} ${what} for ${n} rows`)
  return out
}

function sorted(indices: number[]): Tensor {
  return fromData(Int32Array.from(indices).sort(), [indices.length])
}

/** Split from a test-fold number per row (−1: never in a test set); train is every other assigned row. */
function fromTestFolds(testFold: Int32Array, k: number): Split[] {
  const n = testFold.length
  return Array.from({ length: k }, (_, f) => {
    const test: number[] = []
    const train: number[] = []
    for (let i = 0; i < n; i++) (testFold[i] === f ? test : train).push(i)
    return { train: sorted(train), test: sorted(test) }
  })
}

/** Fold sizes of n rows in k folds: the first n mod k folds get one extra row (scikit-learn's convention). */
function foldSizes(n: number, k: number): number[] {
  return Array.from({ length: k }, (_, f) => Math.floor(n / k) + (f < n % k ? 1 : 0))
}

function checkK(k: number, n: number, name: string) {
  if (!(Number.isInteger(k) && k >= 2)) throw new DomainError(name, `${name}: k must be a whole number ≥ 2`)
  if (k > n) throw new DomainError(name, `${name}: k = ${k} folds for ${n} rows`)
}

function needStream(s: Stream | undefined, name: string): Stream {
  if (!s) throw new DomainError(name, `${name}: a shuffled split needs a stream`)
  return s
}

/**
 * k-fold: k contiguous test blocks (the first n mod k one row larger), each tested once; with `shuffle`, the rows are
 * permuted by the stream first. Matches scikit-learn's `KFold(k)` without shuffling.
 */
export function kFold({ k = 5, shuffle: shuffled = false }: { k?: number; shuffle?: boolean } = {}): Splitter {
  const name = `k-fold(${k}${shuffled ? ', shuffled' : ''})`
  return {
    name,
    randomised: shuffled,
    split(data, s) {
      const { n } = inputOf(data)
      checkK(k, n, 'kFold')
      const order = shuffled
        ? Int32Array.from(permutation(needStream(s, 'kFold'), n).data)
        : Int32Array.from({ length: n }, (_, i) => i)
      const testFold = new Int32Array(n)
      let start = 0
      foldSizes(n, k).forEach((size, f) => {
        for (let r = start; r < start + size; r++) testFold[order[r]] = f
        start += size
      })
      return fromTestFolds(testFold, k)
    },
  }
}

/**
 * Stratified k-fold: each fold's class proportions match the whole set's as nearly as possible. Classes are taken in
 * order of first appearance; the sorted labels are dealt round-robin to the folds to decide how many of each class
 * each fold tests, and each class's rows are assigned to folds in order (shuffled by the stream with `shuffle`). This
 * is scikit-learn's `StratifiedKFold` algorithm, and matches it without shuffling.
 */
export function stratifiedKFold({
  k = 5,
  shuffle: shuffled = false,
}: { k?: number; shuffle?: boolean } = {}): Splitter {
  return {
    name: `stratified-k-fold(${k}${shuffled ? ', shuffled' : ''})`,
    randomised: shuffled,
    split(data, s) {
      const { n, y } = inputOf(data)
      if (!y) throw new DomainError('stratifiedKFold', 'stratifiedKFold: needs labels y')
      checkK(k, n, 'stratifiedKFold')
      const labels = labelsOf(y, n, 'labels')
      // Encode classes by order of first appearance.
      const code = new Map<string | number, number>()
      const encoded = Int32Array.from(labels, (v) => {
        if (!code.has(v)) code.set(v, code.size)
        return code.get(v)!
      })
      const classes = code.size
      const counts = new Int32Array(classes)
      for (const c of encoded) counts[c]++
      if (Math.max(...counts) < k)
        throw new DomainError('stratifiedKFold', `stratifiedKFold: no class has ${k} members`)
      // allocation[f][c]: rows of class c tested in fold f, from dealing the sorted codes round-robin.
      const order = Int32Array.from(encoded).sort()
      const allocation = Array.from({ length: k }, () => new Int32Array(classes))
      order.forEach((c, r) => allocation[r % k][c]++)
      const testFold = new Int32Array(n)
      const stream = shuffled ? needStream(s, 'stratifiedKFold') : undefined
      for (let c = 0; c < classes; c++) {
        const folds: number[] = []
        for (let f = 0; f < k; f++) for (let r = 0; r < allocation[f][c]; r++) folds.push(f)
        if (stream) shuffle(child(stream, 'class', c), folds)
        let next = 0
        for (let i = 0; i < n; i++) if (encoded[i] === c) testFold[i] = folds[next++]
      }
      return fromTestFolds(testFold, k)
    },
  }
}

/**
 * Grouped k-fold: every group lies wholly in one test fold. Groups are taken largest first and each goes to the fold
 * with the fewest rows so far (the first of ties), balancing fold sizes, as scikit-learn's `GroupKFold`. Ties in group
 * size are broken by the sorted order of the group labels.
 */
export function groupKFold({ k = 5 }: { k?: number } = {}): Splitter {
  return {
    name: `group-k-fold(${k})`,
    randomised: false,
    split(data) {
      const { n, groups } = inputOf(data)
      if (!groups) throw new DomainError('groupKFold', 'groupKFold: needs groups')
      const labels = labelsOf(groups, n, 'group labels')
      const unique = Array.from(new Set(labels)).sort((a, b) =>
        typeof a === 'number' && typeof b === 'number'
          ? a - b
          : String(a) < String(b)
            ? -1
            : String(a) > String(b)
              ? 1
              : 0,
      )
      if (unique.length < k) throw new DomainError('groupKFold', `groupKFold: ${unique.length} groups for ${k} folds`)
      const index = new Map(unique.map((g, j) => [g, j]))
      const sizes = new Int32Array(unique.length)
      for (const g of labels) sizes[index.get(g)!]++
      const byWeight = Array.from(unique.keys()).sort((a, b) => sizes[b] - sizes[a] || a - b)
      const load = new Int32Array(k)
      const foldOf = new Int32Array(unique.length)
      for (const g of byWeight) {
        let lightest = 0
        for (let f = 1; f < k; f++) if (load[f] < load[lightest]) lightest = f
        load[lightest] += sizes[g]
        foldOf[g] = lightest
      }
      return fromTestFolds(
        Int32Array.from(labels, (g) => foldOf[index.get(g)!]),
        k,
      )
    },
  }
}

/** Leave-one-out: n splits, each testing one row. */
export function leaveOneOut(): Splitter {
  return {
    name: 'leave-one-out',
    randomised: false,
    split(data) {
      const { n } = inputOf(data)
      if (n < 2) throw new DomainError('leaveOneOut', 'leaveOneOut: needs at least two rows')
      return fromTestFolds(
        Int32Array.from({ length: n }, (_, i) => i),
        n,
      )
    },
  }
}

/**
 * Repeat a randomised splitter `repeats` times on independent substreams `stream.child('repeat', r)`, e.g. repeated
 * (stratified) k-fold. The splits of all repeats are concatenated.
 */
export function repeated(splitter: Splitter, repeats: number): Splitter {
  return {
    name: `repeated(${splitter.name}, ${repeats})`,
    randomised: true,
    split(data, s) {
      const stream = needStream(s, 'repeated')
      return Array.from({ length: repeats }, (_, r) => splitter.split(data, child(stream, 'repeat', r))).flat()
    },
  }
}

/**
 * Shuffle-split: `splits` independent random splits, each testing ⌈testSize·n⌉ rows (a fraction below 1, or a count)
 * and training on ⌊trainSize·n⌋ (default: the rest). Rows may be unused, and a row may be tested in several splits.
 */
export function shuffleSplit({
  splits = 10,
  testSize = 0.1,
  trainSize,
}: { splits?: number; testSize?: number; trainSize?: number } = {}): Splitter {
  return {
    name: `shuffle-split(${splits})`,
    randomised: true,
    split(data, s) {
      const { n } = inputOf(data)
      const stream = needStream(s, 'shuffleSplit')
      const nTest = testSize < 1 ? Math.ceil(testSize * n) : testSize
      const nTrain = trainSize === undefined ? n - nTest : trainSize < 1 ? Math.floor(trainSize * n) : trainSize
      if (nTest + nTrain > n || nTest < 1 || nTrain < 1)
        throw new DomainError('shuffleSplit', 'shuffleSplit: sizes do not fit the rows')
      return Array.from({ length: splits }, (_, r) => {
        const p = Array.from(permutation(child(stream, 'split', r), n).data)
        return { test: sorted(p.slice(0, nTest)), train: sorted(p.slice(nTest, nTest + nTrain)) }
      })
    },
  }
}

/**
 * Expanding-window time-series splits, as scikit-learn's `TimeSeriesSplit`: `splits` consecutive test blocks of
 * `testSize` rows (default ⌊n / (splits + 1)⌋) ending at the last row; each trains on every earlier row (at most
 * `maxTrainSize` of the latest), leaving `gap` rows out before the test block. No training row comes after a test row.
 */
export function expandingWindow({
  splits = 5,
  testSize,
  gap = 0,
  maxTrainSize,
}: { splits?: number; testSize?: number; gap?: number; maxTrainSize?: number } = {}): Splitter {
  return {
    name: `expanding-window(${splits})`,
    randomised: false,
    split(data) {
      const { n } = inputOf(data)
      const size = testSize ?? Math.floor(n / (splits + 1))
      const firstTest = n - splits * size
      if (size < 1 || firstTest - gap < 1)
        throw new DomainError('expandingWindow', `expandingWindow: ${n} rows are too few for ${splits} splits`)
      return Array.from({ length: splits }, (_, f) => {
        const start = firstTest + f * size
        const trainEnd = start - gap
        const trainStart = maxTrainSize === undefined ? 0 : Math.max(0, trainEnd - maxTrainSize)
        return {
          train: fromData(Int32Array.from({ length: trainEnd - trainStart }, (_, i) => trainStart + i)),
          test: fromData(Int32Array.from({ length: size }, (_, i) => start + i)),
        }
      })
    },
  }
}

/**
 * Rolling-origin evaluation with a fixed window (Tashman, 2000, "Out-of-sample tests of forecasting accuracy",
 * International Journal of Forecasting 16): train on the `window` rows before the origin, test the next `horizon` rows
 * after a `gap`, then move the origin forward by `step` (default `horizon`) until the test block would pass the end.
 * The first origin is at `window`. With a growing window use `expandingWindow`.
 */
export function rollingOrigin({
  window,
  horizon = 1,
  step,
  gap = 0,
}: {
  window: number
  horizon?: number
  step?: number
  gap?: number
}): Splitter {
  const by = step ?? horizon
  return {
    name: `rolling-origin(window ${window}, horizon ${horizon})`,
    randomised: false,
    split(data) {
      const { n } = inputOf(data)
      const out: Split[] = []
      for (let origin = window; origin + gap + horizon <= n; origin += by) {
        out.push({
          train: fromData(Int32Array.from({ length: window }, (_, i) => origin - window + i)),
          test: fromData(Int32Array.from({ length: horizon }, (_, i) => origin + gap + i)),
        })
      }
      if (out.length === 0)
        throw new DomainError('rollingOrigin', `rollingOrigin: ${n} rows are too few for a window of ${window}`)
      return out
    },
  }
}

/**
 * The fold assignment matrix [splits, n], int32: `TEST` (1) where row i is tested in split f, `TRAIN` (0) where it is
 * trained on, `UNUSED` (−1) otherwise. A figure draws it directly (e.g. a categorical heatmap).
 */
export function assignment(splits: readonly Split[], n: number): Tensor {
  const out = new Int32Array(splits.length * n).fill(UNUSED)
  splits.forEach(({ train, test }, f) => {
    for (const i of train.data) out[f * n + i] = TRAIN
    for (const i of test.data) out[f * n + i] = TEST
  })
  return fromData(out, [splits.length, n])
}
