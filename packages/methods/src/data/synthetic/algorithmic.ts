/**
 * Algorithmic datasets for small transformers and for grokking.
 *
 * `sequenceTasks` writes one of seven toy problems as strings over a fixed vocabulary: copy, reverse, sort, Dyck-1 and
 * Dyck-2 bracket completion, addition and associative recall (the induction-head task). Each example reads
 * `^ prompt = answer .`; a model is trained to predict the answer tokens and the end mark after the separator. Train
 * and test examples are drawn at different lengths when asked, for length generalisation (Anil et al., 2022, "Exploring
 * length generalization in large language models"). The truth answers any prompt and scores any output exactly.
 *
 * `modularArithmetic` is the full table of a ∘ b mod p for ∘ ∈ {+, −, ×, ÷}, split into train and test pairs by
 * fraction (Power, Burda, Edwards, Babuschkin and Misra, 2022, "Grokking: generalization beyond overfitting on small
 * algorithmic datasets"). Its truth carries the operation and the real Fourier basis of ℤ_p, against which a learned
 * embedding's spectrum is measured: networks that generalise on modular addition represent a and b by a few
 * frequencies (Nanda, Chan, Lieberum, Smith and Steinhardt, 2023, "Progress measures for grokking via mechanistic
 * interpretability").
 */

import type { DatasetInfo, Size, Truth as TruthContract } from 'aifn-compute/foundation/contracts'
import { Categorical } from 'aifn-compute/probability/distributions'
import { child, integers, permutation, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { rfft } from 'aifn-compute/foundation/fourier'
import { labels, matrix, type Dataset, type DatasetMeta } from '../types'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// ── Sequence tasks ───────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The fixed vocabulary of every sequence task: padding `_`, the start mark `^`, the end mark `.`, the separator `=`,
 * `+`, two bracket pairs, the digits and eight letters.
 */
export const SEQUENCE_VOCABULARY: readonly string[] = [
  '_',
  '^',
  '.',
  '=',
  '+',
  '(',
  ')',
  '[',
  ']',
  ...'0123456789',
  ...'abcdefgh',
]
const ID = new Map(SEQUENCE_VOCABULARY.map((t, i) => [t, i]))
const PAD = 0
const START = 1
const END = 2
const SEPARATOR = 3
const LETTERS = SEQUENCE_VOCABULARY.slice(-8)

/** The sequence tasks. */
export const SEQUENCE_TASKS = ['copy', 'reverse', 'sort', 'dyck1', 'dyck2', 'addition', 'induction'] as const
export type SequenceTaskName = (typeof SEQUENCE_TASKS)[number]

/** Token ids of a string over `SEQUENCE_VOCABULARY` (one character per token). */
export function encodeSequence(text: string): number[] {
  return [...text].map((c) => {
    const id = ID.get(c)
    if (id === undefined) throw new DomainError('encodeSequence', `encodeSequence: '${c}' is not in the vocabulary`)
    return id
  })
}

/** The string of token ids (padding dropped). */
export function decodeSequence(ids: readonly number[]): string {
  return ids
    .filter((i) => i !== PAD)
    .map((i) => SEQUENCE_VOCABULARY[i] ?? '?')
    .join('')
}

/** The answer to a prompt, as a string (without the end mark). */
function answerOf(task: SequenceTaskName, prompt: string): string {
  switch (task) {
    case 'copy':
      return prompt
    case 'reverse':
      return [...prompt].reverse().join('')
    case 'sort':
      return [...prompt].sort().join('')
    case 'dyck1':
    case 'dyck2': {
      const stack: string[] = []
      for (const c of prompt) {
        if (c === '(' || c === '[') stack.push(c === '(' ? ')' : ']')
        else if (stack.pop() !== c)
          throw new DomainError('sequenceTasks', `sequenceTasks: '${prompt}' is not a balanced prefix`)
      }
      return stack.reverse().join('')
    }
    case 'addition': {
      const [a, b] = prompt.split('+')
      return String(Number(a) + Number(b))
    }
    case 'induction': {
      // The prompt is distinct symbols then a query among them: the answer is the symbol after the query's first use.
      const query = prompt.at(-1)!
      const at = prompt.indexOf(query)
      if (at < 0 || at >= prompt.length - 2)
        throw new DomainError('sequenceTasks', `sequenceTasks: '${prompt}' has no answer`)
      return prompt[at + 1]
    }
  }
}

/** One prompt of `length` (symbols, or digits per operand for addition) drawn from `s`. */
function drawPrompt(s: Stream, task: SequenceTaskName, length: Size, symbols: Size): string {
  const pick = (k: Size, n: Size) => Array.from(toFlat(integers(s, n, { shape: [k] })))
  switch (task) {
    case 'copy':
    case 'reverse':
    case 'sort':
      return pick(length, symbols)
        .map((i) => LETTERS[i])
        .join('')
    case 'dyck1':
    case 'dyck2': {
      // A random walk on the bracket depth: open with probability ½ (always at depth 0), close the innermost otherwise.
      const kinds = task === 'dyck1' ? 1 : 2
      const coins = pick(length, 2)
      const which = pick(length, kinds)
      const stack: string[] = []
      let out = ''
      for (let i = 0; i < length; i++) {
        if (stack.length === 0 || coins[i] === 0) {
          const open = which[i] === 0 ? '(' : '['
          stack.push(open)
          out += open
        } else out += stack.pop() === '(' ? ')' : ']'
      }
      return out
    }
    case 'addition': {
      const operand = () => {
        const digits = pick(length, 10)
        if (length > 1 && digits[0] === 0) digits[0] = 1 + pick(1, 9)[0]
        return digits.join('')
      }
      return `${operand()}+${operand()}`
    }
    case 'induction': {
      // `length` distinct symbols (at most eight), then a query among all but the last.
      const n = Math.min(length, LETTERS.length)
      const order = Array.from(toFlat(permutation(child(s, 'symbols'), LETTERS.length))).slice(0, n)
      const query = order[pick(1, n - 1)[0]]
      return [...order, query].map((i) => LETTERS[i]).join('')
    }
  }
}

/**
 * Examples of a sequence task as padded token rows: `tokens` [n, L] (`^ prompt = answer .`, then padding), the
 * next-token `targets` [n, L] (tokens shifted left by one, padding at the end), and `weights` [n, L], 1 where the
 * target is an answer token or the end mark and 0 elsewhere, so a loss weighted by them scores only the answer.
 */
export type SequenceExamples = {
  readonly tokens: Tensor
  readonly targets: Tensor
  readonly weights: Tensor
  /** Each example's prompt and answer, as strings. */
  readonly prompts: readonly string[]
  readonly answers: readonly string[]
  /** Each example's length knob (symbols, or digits per operand). */
  readonly lengths: readonly number[]
}

/** What a sequence-task output scores: whether it is exactly the answer, and the share of answer positions right. */
export type SequenceScore = { readonly exact: boolean; readonly tokenAccuracy: number }

/** The truth of a sequence task: the answer to any prompt, and the exact score of any output. */
export interface SequenceTaskTruth {
  readonly task: SequenceTaskName
  /** The answer (without the end mark) to a prompt. */
  answer(prompt: string): string
  /** Score an output against a prompt's answer; text after the first end mark is ignored. */
  score(prompt: string, output: string): SequenceScore
}

/** Train and test examples of a sequence task, with its vocabulary and truth. */
export interface SequenceTaskData {
  readonly task: SequenceTaskName
  readonly vocabulary: readonly string[]
  /** Row length L of `tokens`, shared by both splits (the longest example of either). */
  readonly width: Size
  readonly train: SequenceExamples
  readonly test: SequenceExamples
  readonly truth: SequenceTaskTruth
  readonly meta: DatasetMeta
}

/** Options of `sequenceTasks`. */
export interface SequenceTaskOptions {
  /** Default `reverse`. */
  task?: SequenceTaskName
  /** Training examples (default 2000) and test examples (default 200). */
  n?: Size
  testN?: Size
  /** Training lengths (symbols, or digits per operand for addition), inclusive (defaults 2 and 6). */
  minLength?: Size
  maxLength?: Size
  /**
   * The longest test length. Above `maxLength`, test lengths run from `maxLength + 1` to it (length generalisation);
   * otherwise the test set has the training lengths. Default `maxLength`.
   */
  testLength?: Size
  /** Distinct letters for copy, reverse and sort (2–8; default 8). */
  symbols?: Size
}

/** The truth of a sequence task. */
export function sequenceTaskTruth(task: SequenceTaskName): SequenceTaskTruth {
  return {
    task,
    answer: (prompt) => answerOf(task, prompt),
    score: (prompt, output) => {
      const want = answerOf(task, prompt) + '.'
      const cut = output.indexOf('.')
      const got = cut < 0 ? output : output.slice(0, cut + 1)
      let right = 0
      for (let i = 0; i < want.length; i++) if (got[i] === want[i]) right++
      return { exact: got === want, tokenAccuracy: right / want.length }
    },
  }
}

function examples(
  s: Stream,
  task: SequenceTaskName,
  n: Size,
  lengths: readonly [Size, Size],
  symbols: Size,
): { prompts: string[]; answers: string[]; lengths: number[] } {
  const prompts: string[] = []
  const answers: string[] = []
  const lens: number[] = []
  for (let i = 0; i < n; i++) {
    const si = child(s, i)
    const len = lengths[0] + integers(child(si, 'length'), lengths[1] - lengths[0] + 1)
    const prompt = drawPrompt(si, task, len, symbols)
    prompts.push(prompt)
    answers.push(answerOf(task, prompt))
    lens.push(len)
  }
  return { prompts, answers, lengths: lens }
}

function rows(drawn: { prompts: string[]; answers: string[]; lengths: number[] }, width: Size): SequenceExamples {
  const n = drawn.prompts.length
  const tokens = new Int32Array(n * width)
  const targets = new Int32Array(n * width)
  const weights = new Float64Array(n * width)
  drawn.prompts.forEach((prompt, i) => {
    const ids = [START, ...encodeSequence(prompt), SEPARATOR, ...encodeSequence(drawn.answers[i]), END]
    const answerFrom = prompt.length + 2 // the index of the first answer token in `ids`
    ids.forEach((t, j) => {
      tokens[i * width + j] = t
      if (j > 0) targets[i * width + j - 1] = t
      if (j >= answerFrom) weights[i * width + j - 1] = 1
    })
    for (let j = ids.length; j < width; j++) tokens[i * width + j] = PAD
  })
  return {
    tokens: fromData(tokens, [n, width]),
    targets: fromData(targets, [n, width]),
    weights: fromData(weights, [n, width]),
    ...drawn,
  }
}

/**
 * Seeded examples of an algorithmic sequence task (see the module comment): `^ prompt = answer .` over
 * `SEQUENCE_VOCABULARY`, with train and test sets drawn from `child(s, 'train')` and `child(s, 'test')`.
 */
export function sequenceTasks(s: Stream, options: SequenceTaskOptions = {}): SequenceTaskData {
  const { task = 'reverse', n = 2000, testN = 200, minLength = 2, maxLength = 6, symbols = 8 } = options
  const testLength = options.testLength ?? maxLength
  if (!SEQUENCE_TASKS.includes(task)) throw new DomainError('sequenceTasks', `sequenceTasks: unknown task '${task}'`)
  if (!(minLength >= 1 && maxLength >= minLength))
    throw new DomainError('sequenceTasks', 'sequenceTasks: need 1 ≤ minLength ≤ maxLength')
  if (task === 'induction' && minLength < 3)
    throw new DomainError('sequenceTasks', 'sequenceTasks: induction needs minLength ≥ 3')
  if (task === 'induction' && testLength > LETTERS.length)
    throw new DomainError(
      'sequenceTasks',
      `sequenceTasks: induction draws distinct letters, so lengths are at most ${LETTERS.length}`,
    )
  if (!(symbols >= 2 && symbols <= LETTERS.length))
    throw new DomainError('sequenceTasks', 'sequenceTasks: symbols must be in 2–8')
  const testRange: [Size, Size] = testLength > maxLength ? [maxLength + 1, testLength] : [minLength, maxLength]
  const train = examples(child(s, 'train'), task, n, [minLength, maxLength], symbols)
  const test = examples(child(s, 'test'), task, testN, testRange, symbols)
  const longest = (d: typeof train) => d.prompts.reduce((m, p, i) => Math.max(m, p.length + d.answers[i].length), 0)
  const width = Math.max(longest(train), longest(test)) + 3
  const lengthWord = task === 'addition' ? 'digits per operand' : 'symbols'
  return {
    task,
    vocabulary: SEQUENCE_VOCABULARY,
    width,
    train: rows(train, width),
    test: rows(test, width),
    truth: sequenceTaskTruth(task),
    meta: {
      name: `${task} task`,
      description: `${n} training and ${testN} test examples of the ${task} task as "^ prompt = answer .", with ${minLength}–${maxLength} ${lengthWord} in training and ${testRange[0]}–${testRange[1]} in test.`,
      task: 'sequence',
      featureNames: ['token'],
      key: s.key,
    },
  }
}

// ── Modular arithmetic ───────────────────────────────────────────────────────────────────────────────────────────────

/** The operations of `modularArithmetic`. */
export const MODULAR_OPERATIONS = ['+', '-', '*', '/'] as const
export type ModularOperation = (typeof MODULAR_OPERATIONS)[number]

const isPrime = (p: number) => {
  if (p < 2) return false
  for (let k = 2; k * k <= p; k++) if (p % k === 0) return false
  return true
}

/** b⁻¹ mod p for prime p, by Fermat's little theorem (b^(p−2)). */
function inverse(b: number, p: number): number {
  let result = 1
  let base = b % p
  for (let e = p - 2; e > 0; e >>= 1) {
    if (e & 1) result = (result * base) % p
    base = (base * base) % p
  }
  return result
}

/** a ∘ b mod p (division by the inverse; b ≠ 0). */
export function modularValue(op: ModularOperation, a: number, b: number, p: number): number {
  switch (op) {
    case '+':
      return (a + b) % p
    case '-':
      return (((a - b) % p) + p) % p
    case '*':
      return (a * b) % p
    case '/':
      return (a * inverse(b, p)) % p
  }
}

/**
 * The truth of a modular-arithmetic table, a classification truth whose Bayes rule is the operation itself (Bayes
 * risk 0), plus the real Fourier basis of ℤ_p: the functions 1, cos(2πka/p) and sin(2πka/p) for k = 1, …, ⌊p/2⌋.
 */
export interface ModularTruth extends TruthContract {
  readonly task: 'classification'
  readonly op: ModularOperation
  readonly p: Size
  /** a ∘ b mod p for pairs x [n, 2] (columns a and b): int32 [n]. */
  decide(x: Tensor): Tensor
  /** The frequencies k = 0, …, ⌊p/2⌋ of the basis. */
  readonly frequencies: readonly number[]
  /** The orthonormal real Fourier basis [p, p]: column 0 constant, then cos and sin of each frequency in turn. */
  readonly fourierBasis: Tensor
  /**
   * The share of a table's power [p, d] (one row per residue, e.g. an embedding) at each frequency k = 0, …, ⌊p/2⌋,
   * summed over its columns: |DFT|² along the residues, folded over ±k, normalised to sum to one.
   */
  spectrum(table: Tensor): Float64Array
}

/** The truth of a ∘ b mod p. */
export function modularTruth(op: ModularOperation, p: Size): ModularTruth {
  const half = Math.floor(p / 2)
  const basis = new Float64Array(p * p)
  for (let a = 0; a < p; a++) {
    basis[a * p] = 1 / Math.sqrt(p)
    for (let k = 1; k <= half; k++) {
      const angle = (2 * Math.PI * k * a) / p
      // For even p the frequency p/2 has only its cosine, scaled to unit norm.
      basis[a * p + 2 * k - 1] = (2 * k === p ? 1 / Math.sqrt(p) : Math.sqrt(2 / p)) * Math.cos(angle)
      if (2 * k < p) basis[a * p + 2 * k] = Math.sqrt(2 / p) * Math.sin(angle)
    }
  }
  const decide = (x: Tensor) => {
    const v = toFlat(x)
    return labels(Array.from({ length: v.length / 2 }, (_, i) => modularValue(op, v[2 * i], v[2 * i + 1], p)))
  }
  return {
    kind: 'model',
    task: 'classification',
    op,
    p,
    decide,
    // The answer is certain: all mass on a ∘ b.
    predictive: (x) => {
      const y = toFlat(decide(x))
      const probs = new Float64Array(y.length * p)
      y.forEach((v, i) => (probs[i * p + v] = 1))
      return Categorical(fromData(probs, [y.length, p]))
    },
    expect: (x, f = (v) => v) => fromData(Float64Array.from(toFlat(decide(x)), f)),
    bayesRisk: 0,
    frequencies: Array.from({ length: half + 1 }, (_, k) => k),
    fourierBasis: fromData(basis, [p, p]),
    spectrum: (table) => {
      const [rowsN, d] = table.shape
      if (rowsN !== p) throw new ShapeError('spectrum', `spectrum: the table has ${rowsN} rows, not p = ${p}`)
      const v = toFlat(table)
      const power = new Float64Array(half + 1)
      for (let j = 0; j < d; j++) {
        const column = Array.from({ length: p }, (_, a) => v[a * d + j])
        const f = toFlat(rfft(column) as Tensor) // interleaved re, im for k = 0 … ⌊p/2⌋
        for (let k = 0; k <= half; k++) power[k] += f[2 * k] ** 2 + f[2 * k + 1] ** 2
      }
      const total = power.reduce((x, y) => x + y, 0)
      return total > 0 ? power.map((x) => x / total) : power
    },
  }
}

/** Pairs of residues: x [n, 2] (a and b as float64), labels y int32 [n], and the table rows they are (`rows`). */
export type ModularPart = Dataset & { readonly y: Tensor; readonly rows: Tensor }

/** The full table and its random train and test parts, with the truth beside them. */
export interface ModularArithmeticData {
  readonly op: ModularOperation
  readonly p: Size
  readonly table: ModularPart
  readonly train: ModularPart
  readonly test: ModularPart
  readonly truth: ModularTruth
}

/** Options of `modularArithmetic`. */
export interface ModularArithmeticOptions {
  /** The modulus (default 31; prime for division). */
  p?: Size
  /** Default `+`. */
  op?: ModularOperation
  /** The share of the table used for training (default 0.5). */
  fraction?: number
}

/**
 * The table of a ∘ b mod p over every pair (a, b) ∈ ℤ_p² (b ≠ 0 for division), split at random into a training share
 * `fraction` of its rows and the test rest, drawn from `child(s, 'split')`. Features are the pair (a, b), labels
 * a ∘ b; the truth (`ModularTruth`) is beside the parts rather than in their metadata, since dataset modifiers do not
 * apply to a finite table.
 */
export function modularArithmetic(s: Stream, options: ModularArithmeticOptions = {}): ModularArithmeticData {
  const { p = 31, op = '+', fraction = 0.5 } = options
  if (!(Number.isInteger(p) && p >= 2))
    throw new DomainError('modularArithmetic', `modularArithmetic: p must be an integer ≥ 2, got ${p}`)
  if (!MODULAR_OPERATIONS.includes(op))
    throw new DomainError('modularArithmetic', `modularArithmetic: unknown operation '${op}'`)
  if (op === '/' && !isPrime(p))
    throw new DomainError('modularArithmetic', `modularArithmetic: division needs a prime p, got ${p}`)
  if (!(fraction > 0 && fraction < 1))
    throw new DomainError('modularArithmetic', 'modularArithmetic: fraction must be in (0, 1)')
  const pairs: [number, number][] = []
  for (let a = 0; a < p; a++) for (let b = op === '/' ? 1 : 0; b < p; b++) pairs.push([a, b])
  const truth = modularTruth(op, p)
  const name = `a ${op} b mod ${p}`
  const part = (ids: readonly number[], which: string): ModularPart => ({
    kind: 'dataset',
    x: matrix(Float64Array.from(ids.flatMap((i) => pairs[i])), ids.length, 2),
    y: labels(ids.map((i) => modularValue(op, pairs[i][0], pairs[i][1], p))),
    rows: labels(ids),
    meta: {
      name: which === 'table' ? name : `${name} (${which})`,
      description: `${which === 'table' ? 'Every pair' : `The ${which} pairs (${ids.length} of ${pairs.length})`} of residues (a, b) with the label a ${op} b mod ${p}.`,
      task: 'classification',
      featureNames: ['a', 'b'],
      labelNames: Array.from({ length: p }, (_, k) => String(k)),
      source: 'Power et al. (2022), "Grokking: generalization beyond overfitting on small algorithmic datasets"',
      key: s.key,
    },
  })
  const order = Array.from(toFlat(permutation(child(s, 'split'), pairs.length)))
  const nTrain = Math.max(1, Math.min(pairs.length - 1, Math.round(fraction * pairs.length)))
  const sorted = (ids: number[]) => ids.sort((x, y) => x - y)
  return {
    op,
    p,
    table: part(
      pairs.map((_, i) => i),
      'table',
    ),
    train: part(sorted(order.slice(0, nTrain)), 'train'),
    test: part(sorted(order.slice(nTrain)), 'test'),
    truth,
  }
}

// ── Registration ─────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'sequenceTasks',
    name: 'Algorithmic sequence tasks',
    summary:
      'Copy, reverse, sort, Dyck bracket completion, addition and associative recall as "^ prompt = answer ." strings, split by length.',
    task: 'sequence',
    output: 'sequence',
    knobs: space({
      task: oneOf(SEQUENCE_TASKS, { default: 'reverse' }),
      n: int(1, 100000, { default: 2000 }),
      testN: int(1, 100000, { default: 200 }),
      minLength: int(1, 12, { default: 2 }),
      maxLength: int(1, 12, { default: 6 }),
      testLength: int(1, 16, { default: 6 }),
      symbols: int(2, 8, { default: 8 }),
    }),
    truth: true,
    random: true,
    notes: ['transformer', 'attention-head-analysis'],
    cite: ['olsson2022'],
  },
  sequenceTasks,
)

dataset(
  {
    key: 'modularArithmetic',
    name: 'Modular arithmetic',
    summary: 'The table of a ∘ b mod p for +, −, × or ÷, split into train and test pairs by fraction.',
    task: 'classification',
    output: 'split',
    knobs: space({
      p: int(2, 113, { default: 31 }),
      op: oneOf(MODULAR_OPERATIONS, { default: '+' }),
      fraction: real(0.05, 0.95, { default: 0.5 }),
    }),
    truth: true,
    random: true,
    notes: ['double-descent', 'decoupled-weight-decay', 'discrete-fourier-transform'],
  },
  modularArithmetic,
)
