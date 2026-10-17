/**
 * Synthetic tasks for comparing sequence layers on recall and state tracking, beside the string tasks of
 * `sequenceTasks`: selective copying and parity. Multi-query associative recall (`multiQueryAssociativeRecall`) is
 * generated in `aifn-methods/neural/sequence`, beside the probe that calls it, and re-exported with these.
 *
 * - `selectiveCopy`: a few content tokens at random positions among noise tokens; the answer is the content tokens in
 *   order (Gu and Dao, 2023, "Mamba", §3.1 and Figure 2). Token 0 is the noise and $1, \dots, V$ the content.
 * - `parityStrings`: random bit strings and the parity of their number of ones, the state-tracking task a linear
 *   recurrence with positive decays cannot solve (Grazzi et al., 2024, §5).
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { child, choice, integers, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** One selective-copying example: the input (0 for noise), the content tokens in order, and their positions. */
export type SelectiveCopyExample = {
  /** The sequence: content tokens at `positions`, 0 elsewhere. */
  input: number[]
  /** The content tokens in the order they appear, the answer. */
  target: number[]
  /** The positions of the content tokens, increasing. */
  positions: number[]
}

/**
 * One selective-copying example: `tokens` content tokens from $1, \dots, V$ at random distinct positions of a
 * sequence of `length`, noise (0) elsewhere. The answer is the content tokens in the order they appear.
 *
 * @param s The stream; only its children `positions` and `content` are drawn from.
 * @param options The sequence `length`, the number of content `tokens` and the content vocabulary size `vocab` ($V$).
 * @returns The input, the target and the content positions.
 *
 * @example Four tokens to copy out of sixteen
 * const c = selectiveCopy(stream(2), { length: 16, tokens: 4, vocab: 8 })
 * print('input:', c.input)
 * print('target:', c.target)
 */
export function selectiveCopy(
  s: Stream,
  options: { length: number; tokens: number; vocab: number },
): SelectiveCopyExample {
  const { length, tokens, vocab } = options
  const positions = toFlat(choice(child(s, 'positions'), length, tokens, { replace: false })).sort((a, b) => a - b)
  const content = toFlat(integers(child(s, 'content'), vocab, { shape: [tokens] }) as Tensor).map((v) => v + 1)
  const input = new Array<number>(length).fill(0)
  positions.forEach((p, i) => (input[p] = content[i]))
  return { input, target: content, positions }
}

/**
 * `n` random bit strings of `length` and the parity (0 even, 1 odd) of each string's number of ones. The bits are one
 * block of draws from `s`, row by row.
 *
 * @param s The stream the bits are drawn from.
 * @param options The string `length` and the number of strings `n`.
 * @returns The `bits`, one array per string, and the `parity` of each.
 *
 * @example Four strings of five bits
 * const p = parityStrings(stream(3), { length: 5, n: 4 })
 * print('bits:', p.bits)
 * print('parity:', p.parity)
 */
export function parityStrings(
  s: Stream,
  options: { length: number; n: number },
): { bits: number[][]; parity: number[] } {
  const { length, n } = options
  const flat = toFlat(integers(s, 2, { shape: [n, length] }) as Tensor)
  const bits = Array.from({ length: n }, (_, i) => flat.slice(i * length, (i + 1) * length))
  return { bits, parity: bits.map((b) => b.reduce((a, x) => a + x, 0) % 2) }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'selectiveCopy',
    name: 'Selective copying',
    summary: 'Content tokens scattered among noise; the answer is the content in order, which needs a selective state.',
    role: 'simulation',
    random: true,
    notes: ['long-range-and-recall-benchmarks'],
    cite: ['gu2023mamba'],
  },
  selectiveCopy,
)
fn(
  {
    key: 'parityStrings',
    name: 'Parity strings',
    summary: 'Random bit strings labelled by the parity of their ones, a state-tracking task.',
    role: 'simulation',
    random: true,
    notes: ['expressiveness-of-linear-recurrences'],
    cite: ['grazzi2024'],
  },
  parityStrings,
)
