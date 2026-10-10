/**
 * Algorithmic datasets for small transformers and for grokking.
 *
 * `sequenceTasks` writes one of seven toy problems as strings over a fixed vocabulary: copy, reverse, sort, Dyck-1 and
 * Dyck-2 bracket completion, addition and associative recall (the induction-head task). Each example reads
 * `^ prompt = answer .`; a model is trained to predict the answer tokens and the end mark after the separator. Train
 * and test examples are drawn at different lengths when asked, for length generalisation (Anil et al., 2022, "Exploring
 * length generalization in large language models"). The truth answers any prompt and scores any output exactly.
 *
 * `modularArithmetic` is the full table of $a \circ b \bmod p$ for $\circ \in \{+, -, \times, \div\}$, split into
 * train and test pairs by fraction (Power, Burda, Edwards, Babuschkin and Misra, 2022, "Grokking: generalization beyond
 * overfitting on small algorithmic datasets"). Its truth carries the operation and the real Fourier basis of
 * $\integers_p$, against which a learned embedding's spectrum is measured: networks that generalise on modular addition
 * represent $a$ and $b$ by a few frequencies (Nanda, Chan, Lieberum, Smith and Steinhardt, 2023, "Progress measures for
 * grokking via mechanistic interpretability").
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
/** The name of a sequence task, one of `SEQUENCE_TASKS`. */
export type SequenceTaskName = (typeof SEQUENCE_TASKS)[number]

/**
 * Token ids of a string over `SEQUENCE_VOCABULARY` (one character per token). Throws `DomainError` for a character
 * outside the vocabulary.
 *
 * @param text The string, one token per character.
 * @returns The token ids, indices into `SEQUENCE_VOCABULARY`.
 *
 * @example A reverse example, encoded and decoded
 * const ids = encodeSequence('^abc=cba.')
 * print('ids:', ids)
 * print('back:', decodeSequence(ids))
 */
export function encodeSequence(text: string): number[] {
  return [...text].map((c) => {
    const id = ID.get(c)
    if (id === undefined) throw new DomainError('encodeSequence', `encodeSequence: '${c}' is not in the vocabulary`)
    return id
  })
}

/**
 * The string of token ids (padding dropped); an id outside the vocabulary shows as `?`.
 *
 * @param ids Token ids, indices into `SEQUENCE_VOCABULARY`.
 * @returns The string, one character per token that is not padding.
 *
 * @example A padded row
 * print(decodeSequence([1, 19, 20, 3, 20, 19, 2, 0, 0]))
 * print(decodeSequence(encodeSequence('^12+34=46.')))
 */
export function decodeSequence(ids: readonly number[]): string {
  return ids
    .filter((i) => i !== PAD)
    .map((i) => SEQUENCE_VOCABULARY[i] ?? '?')
    .join('')
}

/**
 * The answer to a prompt, as a string (without the end mark): the prompt itself, reversed or sorted; the closing
 * brackets that complete a Dyck prefix; the sum; or, for induction, the symbol after the query's first use. Throws
 * `DomainError` for a Dyck prompt that is not a balanced prefix, or an induction prompt with no answer.
 *
 * @param task The task.
 * @param prompt The prompt, a string over the vocabulary.
 * @returns The answer.
 */
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

/**
 * One prompt of `length` (symbols, or digits per operand for addition) drawn from `s`: random letters (copy, reverse,
 * sort); a random walk on the bracket depth (Dyck), which may end at any depth; two operands without leading zeros
 * (addition); or $\min(\text{length}, 8)$ distinct letters followed by a query among all but the last (induction).
 *
 * @param s The random stream; induction also uses its child `symbols`.
 * @param task The task.
 * @param length The prompt's length: symbols, digits per operand (addition) or distinct letters before the query
 *   (induction).
 * @param symbols How many letters copy, reverse and sort draw from, the first of `abcdefgh`.
 * @returns The prompt.
 */
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
 * Examples of a sequence task as padded token rows: `tokens` ($n \times L$, `^ prompt = answer .`, then padding), the
 * next-token `targets` ($n \times L$, tokens shifted left by one, padding at the end), and `weights` ($n \times L$), 1
 * where the target is an answer token or the end mark and 0 elsewhere, so a loss weighted by them scores only the
 * answer.
 */
export type SequenceExamples = {
  /** The token rows (int32), $n \times L$. */
  readonly tokens: Tensor
  /** The next token at each position (int32), $n \times L$. */
  readonly targets: Tensor
  /** 1 where the target is an answer token or the end mark, else 0 (float64), $n \times L$. */
  readonly weights: Tensor
  /** Each example's prompt, as a string. */
  readonly prompts: readonly string[]
  /** Each example's answer, as a string without the end mark. */
  readonly answers: readonly string[]
  /** Each example's length knob (symbols, or digits per operand). */
  readonly lengths: readonly number[]
}

/**
 * What a sequence-task output scores: `exact`, whether it is exactly the answer, and `tokenAccuracy`, the share of
 * answer positions (the end mark included) that are right.
 */
export type SequenceScore = { readonly exact: boolean; readonly tokenAccuracy: number }

/** The truth of a sequence task: the answer to any prompt, and the exact score of any output. */
export interface SequenceTaskTruth {
  /** The task. */
  readonly task: SequenceTaskName
  /** The answer (without the end mark) to a prompt. */
  answer(prompt: string): string
  /** Score an output against a prompt's answer; text after the first end mark is ignored. */
  score(prompt: string, output: string): SequenceScore
}

/** Train and test examples of a sequence task, with its vocabulary and truth. */
export interface SequenceTaskData {
  /** The task. */
  readonly task: SequenceTaskName
  /** The vocabulary, `SEQUENCE_VOCABULARY`. */
  readonly vocabulary: readonly string[]
  /** Row length $L$ of `tokens`, shared by both splits (the longest example of either). */
  readonly width: Size
  /** The training examples. */
  readonly train: SequenceExamples
  /** The test examples. */
  readonly test: SequenceExamples
  /** The task's truth, as `sequenceTaskTruth` gives it. */
  readonly truth: SequenceTaskTruth
  /** The description and the stream key. */
  readonly meta: DatasetMeta
}

/** Options of `sequenceTasks`. */
export interface SequenceTaskOptions {
  /** Default `reverse`. */
  task?: SequenceTaskName
  /** Training examples (default 2000). */
  n?: Size
  /** Test examples (default 200). */
  testN?: Size
  /**
   * The shortest training length (symbols, or digits per operand for addition), at least 1, and at least 3 for
   * induction (default 2).
   */
  minLength?: Size
  /** The longest training length, inclusive (default 6). */
  maxLength?: Size
  /**
   * The longest test length. Above `maxLength`, test lengths run from `maxLength + 1` to it (length generalisation);
   * otherwise the test set has the training lengths. Default `maxLength`.
   */
  testLength?: Size
  /** Distinct letters for copy, reverse and sort, from 2 to 8 (default 8). */
  symbols?: Size
}

/**
 * The truth of a sequence task: the answer to any prompt, and the score of any output against it.
 *
 * @param task The task.
 * @returns The truth: `answer(prompt)` and `score(prompt, output)`.
 *
 * @example Answers and scores
 * const t = sequenceTaskTruth('dyck2')
 * print('answer to ([(:', t.answer('([('))
 * print('score of )]).:', t.score('([(', ')]).'))
 * print('score of ))):', t.score('([(', ')))'))
 */
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

/**
 * Draw $n$ prompts of a task, each with a length uniform on the given range, and their answers. Example $i$ is drawn
 * from the child $i$ of `s`, its length from that stream's child `length`.
 *
 * @param s The random stream.
 * @param task The task.
 * @param n The number of examples.
 * @param lengths The shortest and longest length, inclusive.
 * @param symbols How many letters copy, reverse and sort draw from.
 * @returns The prompts, answers and lengths, in order.
 */
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

/**
 * The drawn examples as padded token rows, next-token targets and answer weights.
 *
 * @param drawn The prompts, answers and lengths, as `examples` returns them.
 * @param width The row length $L$; at least the longest example plus 3 (the start, separator and end marks).
 * @returns The examples as `SequenceExamples`.
 */
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
 * `SEQUENCE_VOCABULARY`, with train and test sets drawn from `child(s, 'train')` and `child(s, 'test')`. Each example's
 * length is uniform on its range. Throws `DomainError` for an unknown task, lengths out of order, induction lengths
 * below 3 or test lengths above 8, or `symbols` outside 2 to 8.
 *
 * @param s The random stream.
 * @param options The task, the numbers of examples, the training and test lengths and the letters; see
 *   `SequenceTaskOptions`.
 * @returns The train and test examples, their shared row width, the vocabulary and the truth.
 *
 * @example Reversing strings
 * const data = sequenceTasks(stream(0), { task: 'reverse', n: 100, testN: 20 })
 * print('tokens:', data.train.tokens.shape, ' width:', data.width)
 * print('first prompts:', data.train.prompts.slice(0, 3), ' answers:', data.train.answers.slice(0, 3))
 * print('first row:', decodeSequence(toArray(data.train.tokens)[0]))
 * // The weights pick the answer tokens and the end mark.
 * const weighted = toArray(data.train.weights)[0].reduce((a, v) => a + v, 0)
 * print('weighted positions in row 0:', weighted, 'for the answer', data.train.answers[0], 'and the end mark')
 *
 * @example Longer strings in the test set
 * const data = sequenceTasks(stream(0), { task: 'addition', n: 100, testN: 20, maxLength: 3, testLength: 5 })
 * print('train lengths:', Math.min(...data.train.lengths), 'to', Math.max(...data.train.lengths))
 * print('test lengths:', Math.min(...data.test.lengths), 'to', Math.max(...data.test.lengths))
 * print('a test example:', data.test.prompts[0], '=', data.test.answers[0])
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
/** An operation of `modularArithmetic`, one of `MODULAR_OPERATIONS`. */
export type ModularOperation = (typeof MODULAR_OPERATIONS)[number]

/**
 * Whether a number is prime, by trial division.
 *
 * @param p The number.
 * @returns True when $p$ is a prime.
 */
const isPrime = (p: number) => {
  if (p < 2) return false
  for (let k = 2; k * k <= p; k++) if (p % k === 0) return false
  return true
}

/**
 * $b^{-1} \bmod p$ for prime $p$, by Fermat's little theorem ($b^{p-2}$), with square-and-multiply.
 *
 * @param b The residue to invert; not a multiple of $p$.
 * @param p The modulus, a prime small enough that $p^2$ is exact in floating point.
 * @returns The inverse, in $0, \dots, p - 1$.
 */
function inverse(b: number, p: number): number {
  let result = 1
  let base = b % p
  for (let e = p - 2; e > 0; e >>= 1) {
    if (e & 1) result = (result * base) % p
    base = (base * base) % p
  }
  return result
}

/**
 * $a \circ b \bmod p$, in $0, \dots, p - 1$; division multiplies by the inverse of $b$, so it needs a prime $p$ and
 * $b \ne 0$ (for $b = 0$ it returns 0).
 *
 * @param op The operation: `+`, `-`, `*` or `/`.
 * @param a The first residue, in $0, \dots, p - 1$.
 * @param b The second residue, in $0, \dots, p - 1$.
 * @param p The modulus.
 * @returns The residue $a \circ b \bmod p$.
 *
 * @example Division mod 7
 * print('3 / 5 mod 7 =', modularValue('/', 3, 5, 7))
 * print('check, 2 * 5 mod 7 =', modularValue('*', 2, 5, 7))
 * print('2 - 5 mod 7 =', modularValue('-', 2, 5, 7))
 */
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
 * risk 0), plus the real Fourier basis of $\integers_p$: the functions 1, $\cos(2\pi k a/p)$ and $\sin(2\pi k a/p)$
 * for $k = 1, \dots, \lfloor p/2 \rfloor$.
 */
export interface ModularTruth extends TruthContract {
  /** Always `'classification'`. */
  readonly task: 'classification'
  /** The operation. */
  readonly op: ModularOperation
  /** The modulus $p$. */
  readonly p: Size
  /** $a \circ b \bmod p$ for pairs `x` ($n \times 2$, columns $a$ and $b$): int32, $n$ values. */
  decide(x: Tensor): Tensor
  /** The frequencies $k = 0, \dots, \lfloor p/2 \rfloor$ of the basis. */
  readonly frequencies: readonly number[]
  /**
   * The orthonormal real Fourier basis, $p \times p$, one row per residue $a$: column 0 constant, then the cosine and
   * sine of each frequency in turn (for even $p$, the frequency $p/2$ has its cosine only).
   */
  readonly fourierBasis: Tensor
  /**
   * The share of a table's power at each frequency $k = 0, \dots, \lfloor p/2 \rfloor$, for a $p \times d$ table (one
   * row per residue, e.g. an embedding), summed over its columns: $\lvert \hat t_k \rvert^2$ of the one-sided DFT
   * along the residues, normalised to sum to one. Bins $k \ge 1$ are not doubled for $-k$. Throws `ShapeError` unless
   * the table has $p$ rows.
   */
  spectrum(table: Tensor): Float64Array
}

/**
 * The truth of $a \circ b \bmod p$: the operation as its decision, a predictive with all its mass on the answer, the
 * real Fourier basis of $\integers_p$ and the spectrum of a table against it.
 *
 * @param op The operation.
 * @param p The modulus.
 * @returns The truth.
 *
 * @example The answer, the basis, and the spectrum of a pure frequency
 * const t = modularTruth('+', 7)
 * print('3 + 5 mod 7 =', t.decide(tensor([[3, 5]])))
 * const b = toArray(t.fourierBasis)
 * const dot = (i, j) => b.reduce((a, row) => a + row[i] * row[j], 0)
 * print('columns 1 and 1, 1 and 2:', dot(1, 1), dot(1, 2))
 * // An embedding of the residues by cos and sin of frequency 2 has all its power there.
 * const angle = (a) => (2 * Math.PI * 2 * a) / 7
 * const table = tensor(Array.from({ length: 7 }, (_, a) => [Math.cos(angle(a)), Math.sin(angle(a))]))
 * print('frequencies:', t.frequencies, ' spectrum:', t.spectrum(table))
 */
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

/**
 * Pairs of residues: `x` ($n \times 2$, $a$ and $b$ as float64), labels `y` (int32, $n$ values), and `rows`, the
 * index of each pair in the full table (int32).
 */
export type ModularPart = Dataset & { readonly y: Tensor; readonly rows: Tensor }

/** The full table and its random train and test parts, with the truth beside them. */
export interface ModularArithmeticData {
  /** The operation. */
  readonly op: ModularOperation
  /** The modulus $p$. */
  readonly p: Size
  /** Every pair, in order of $a$ then $b$. */
  readonly table: ModularPart
  /** The training pairs, in table order. */
  readonly train: ModularPart
  /** The test pairs, the rest of the table, in table order. */
  readonly test: ModularPart
  /** The truth of the operation. */
  readonly truth: ModularTruth
}

/** Options of `modularArithmetic`. */
export interface ModularArithmeticOptions {
  /** The modulus (default 31; prime for division). */
  p?: Size
  /** Default `+`. */
  op?: ModularOperation
  /** The share of the table used for training, in $(0, 1)$ (default 0.5). */
  fraction?: number
}

/**
 * The table of $a \circ b \bmod p$ over every pair $(a, b) \in \integers_p^2$ ($b \ne 0$ for division), split at random
 * into a training share `fraction` of its rows (rounded, and at least one row in each part) and the test rest, drawn
 * from `child(s, 'split')`. Features are the pair $(a, b)$, labels $a \circ b$; the truth (`ModularTruth`) is beside
 * the parts rather than in their metadata, since dataset modifiers do not apply to a finite table. Throws `DomainError`
 * unless $p$ is an integer of at least 2 (a prime for division), the operation is known and the fraction is in
 * $(0, 1)$.
 *
 * @param s The random stream; the split is drawn from its child `split`.
 * @param options The modulus, the operation and the training share; see `ModularArithmeticOptions`.
 * @returns The full table, its train and test parts, and the truth.
 *
 * @example Addition mod 7
 * const data = modularArithmetic(stream(0), { p: 7 })
 * print('table:', data.table.x.shape, ' train:', data.train.x.shape, ' test:', data.test.x.shape)
 * print('first training pairs:', toArray(data.train.x).slice(0, 3), ' labels:', toArray(data.train.y).slice(0, 3))
 * const x = toArray(data.table.x)
 * print('labels are (a + b) mod 7:', toArray(data.table.y).every((v, i) => v === (x[i][0] + x[i][1]) % 7))
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
