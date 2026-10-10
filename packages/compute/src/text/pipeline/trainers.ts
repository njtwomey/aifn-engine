/**
 * Training a tokeniser's model on a corpus through its own normaliser and pre-tokeniser: the corpus is cut into
 * pre-tokens exactly as encoding will cut it, counted, and handed to the step-through trainers of
 * `aifn-compute/text/subword` (BPE merges, WordPiece merges, unigram pruning rounds); word-level and character
 * vocabularies are counted directly.
 * The vocabulary size counts everything the model will hold: special tokens, byte-fallback tokens and the alphabet.
 */

import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import {
  byteAlphabet,
  bpeSteps,
  unigramLmSteps,
  wordPieceSteps,
  type BpeState,
  type UnigramLmState,
  type WordPieceState,
} from 'aifn-compute/text/subword'
import { bpeStage, characterStage, unigramStage, wordLevelStage, wordPieceStage, type TokeniserModel } from './models'
import { preTokenCounts, tokeniser, type Tokeniser, type UntrainedTokeniser } from './tokeniser'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * What to train and its options. Unless given, the specials are `[UNK]` for WordPiece, none for BPE without byte
 * fallback and `<unk>` otherwise; the unknown token is `[UNK]` (WordPiece) or `<unk>` when it is among them, or none.
 */
export type Trainer =
  | {
      /** Byte-pair encoding. */
      readonly type: 'bpe'
      /** The target size of the whole vocabulary: specials, byte-fallback tokens, alphabet and merges. */
      readonly vocabularySize: number
      /** The special tokens, given the first ids. */
      readonly specials?: readonly string[]
      /** The unknown token, or null for none. */
      readonly unknown?: string | null
      /** Add the 256 `<0xNN>` tokens and fall back to them (SentencePiece, LLaMA). */
      readonly byteFallback?: boolean
      /** Every byte symbol in the alphabet, seen or not (byte-level BPE, as GPT-2). */
      readonly byteAlphabet?: boolean
      /** The end-of-word symbol appended to each pre-token (default none). */
      readonly endOfWord?: string
      /** Stop when the most frequent pair occurs fewer times than this (default 2). */
      readonly minCount?: number
      /** The BPE-dropout probability of the trained model (default 0). */
      readonly dropout?: number
    }
  | {
      /** WordPiece. */
      readonly type: 'wordPiece'
      /** The target size of the vocabulary, specials included. */
      readonly vocabularySize: number
      /** The special tokens, listed first. */
      readonly specials?: readonly string[]
      /** The continuation prefix (default `##`). */
      readonly prefix?: string
      /** The merge score (default `ratio`). */
      readonly criterion?: 'ratio' | 'likelihood'
    }
  | {
      /** The unigram language model. */
      readonly type: 'unigram'
      /**
       * The target size of the whole vocabulary: specials, byte-fallback tokens and pieces (exceeded when the single
       * characters alone are more).
       */
      readonly vocabularySize: number
      /** The special tokens, given the first ids. */
      readonly specials?: readonly string[]
      /** The unknown token, or null for none. */
      readonly unknown?: string | null
      /** Add the 256 `<0xNN>` tokens and fall back to them. */
      readonly byteFallback?: boolean
      /** The longest seed piece, in characters (default 16). */
      readonly maxPieceLength?: number
      /** The share of pieces each pruning round keeps (default 0.75). */
      readonly shrink?: number
    }
  | {
      /** A word-level vocabulary. */
      readonly type: 'wordLevel'
      /** The most tokens kept, specials included (default no limit). */
      readonly vocabularySize?: number
      /** The fewest occurrences a pre-token needs (default 1). */
      readonly minCount?: number
      /** The special tokens, given the first ids. */
      readonly specials?: readonly string[]
      /** The unknown token, or null for none. */
      readonly unknown?: string | null
    }
  | {
      /** A character vocabulary. */
      readonly type: 'character'
      /** The fewest occurrences a character needs (default 1). */
      readonly minCount?: number
      /** The special tokens, given the first ids. */
      readonly specials?: readonly string[]
      /** The unknown token, or null for none. */
      readonly unknown?: string | null
      /** Add the 256 `<0xNN>` tokens and fall back to them. */
      readonly byteFallback?: boolean
    }

/** The training state of a subword trainer. */
export type TrainerState = BpeState | WordPieceState | UnigramLmState

/**
 * The special tokens of a trainer, its own or the default.
 *
 * @param tr The trainer.
 * @returns The special tokens.
 */
const specialsOf = (tr: Trainer) =>
  tr.specials ?? (tr.type === 'wordPiece' ? ['[UNK]'] : tr.type === 'bpe' && !tr.byteFallback ? [] : ['<unk>'])
/**
 * The unknown token of a trainer: for WordPiece, `[UNK]` if a special, else the first special; otherwise the given one,
 * or `<unk>` if a special.
 *
 * @param tr The trainer.
 * @returns The unknown token, or null for none.
 */
const unknownOf = (tr: Trainer): string | null => {
  if (tr.type === 'wordPiece') {
    const sp = specialsOf(tr)
    return sp.includes('[UNK]') ? '[UNK]' : (sp[0] ?? null)
  }
  if (tr.unknown !== undefined) return tr.unknown
  return specialsOf(tr).includes('<unk>') ? '<unk>' : null
}

/**
 * The step-through training of a subword model (BPE, WordPiece or unigram) on the pre-tokens of `texts`: one merge,
 * or one pruning round, per step. Pass a state to {@link trainedModel} to get the model at that step. Throws
 * `DomainError` for a word-level or character trainer, which have no steps.
 *
 * @param t The tokeniser, or its parts before training: its normaliser and pre-tokeniser cut the texts.
 * @param texts The training texts.
 * @param trainer What to train; the vocabulary size passed on leaves room for the specials and byte-fallback tokens.
 * @returns The algorithm, to step with `run`; it takes no input.
 *
 * @example Watch the first BPE merges, and take the model after two
 * const trainer = { type: 'bpe', vocabularySize: 20, specials: ['<unk>'] }
 * const alg = trainingSteps({ preTokeniser: whitespacePreTokeniser() }, ['low lower lowest', 'low low lower'], trainer)
 * for (const k of [1, 2, 3]) print(`step ${k}:`, run(alg, undefined, k).merge.merged)
 * const model = trainedModel(trainer, run(alg, undefined, 2))
 * print('lowest =', modelSegment(model, 'lowest').map((x) => x.token))
 */
export function trainingSteps(
  t: UntrainedTokeniser | Tokeniser,
  texts: readonly string[],
  trainer: Trainer,
): Algorithm<void, TrainerState> {
  const words = preTokenCounts(t, texts)
  const specials = specialsOf(trainer)
  switch (trainer.type) {
    case 'bpe': {
      const reserved = specials.length + (trainer.byteFallback ? 256 : 0)
      return bpeSteps(words, {
        unit: 'character',
        endOfWord: trainer.endOfWord ?? '',
        merges: 1e6,
        vocabularySize: Math.max(1, trainer.vocabularySize - reserved),
        minCount: trainer.minCount ?? 2,
        alphabet: trainer.byteAlphabet ? byteAlphabet() : [],
      }) as Algorithm<void, TrainerState>
    }
    case 'wordPiece':
      return wordPieceSteps(words, {
        merges: 1e6,
        vocabularySize: trainer.vocabularySize,
        specials,
        prefix: trainer.prefix ?? '##',
        criterion: trainer.criterion ?? 'ratio',
      }) as Algorithm<void, TrainerState>
    case 'unigram': {
      const reserved = specials.length + (trainer.byteFallback ? 256 : 0)
      return unigramLmSteps(words, {
        vocabularySize: Math.max(1, trainer.vocabularySize - reserved),
        boundary: '',
        maxPieceLength: trainer.maxPieceLength ?? 16,
        shrink: trainer.shrink ?? 0.75,
        seedSize: 4000,
      }) as Algorithm<void, TrainerState>
    }
    default:
      throw new DomainError('trainingSteps', `trainingSteps: ${trainer.type} has no step-through training`)
  }
}

/**
 * The model of a subword trainer at a training state. Throws `DomainError` for a word-level or character trainer.
 *
 * @param trainer The trainer the state came from.
 * @param state A state of the algorithm `trainingSteps` returned for it.
 * @returns The model stage, with the trainer's specials, unknown token and byte fallback.
 *
 * @example A WordPiece model after one merge and after five
 * const trainer = { type: 'wordPiece', vocabularySize: 100 }
 * const alg = trainingSteps({ preTokeniser: whitespacePreTokeniser() }, ['hug hug pug pun bun hugs'], trainer)
 * for (const k of [1, 5]) {
 *   const model = trainedModel(trainer, run(alg, undefined, k))
 *   print(`after ${k}:`, modelSegment(model, 'hugs').map((x) => x.token))
 * }
 */
export function trainedModel(trainer: Trainer, state: TrainerState): TokeniserModel {
  const specials = specialsOf(trainer)
  const unknown = unknownOf(trainer)
  switch (trainer.type) {
    case 'bpe':
      return bpeStage(state as BpeState, {
        specials,
        unknown,
        byteFallback: trainer.byteFallback ?? false,
        fuseUnknown: trainer.byteFallback ?? false,
        endOfWord: trainer.endOfWord ?? '',
        dropout: trainer.dropout ?? 0,
      })
    case 'wordPiece':
      return wordPieceStage(state as WordPieceState, { specials, unknown, prefix: trainer.prefix ?? '##' })
    case 'unigram':
      return unigramStage(state as UnigramLmState, { specials, unknown, byteFallback: trainer.byteFallback ?? false })
    default:
      throw new DomainError('trainedModel', `trainedModel: ${trainer.type} has no training state`)
  }
}

/**
 * The entries counted at least `minCount` times, most frequent first.
 *
 * @param words Counts, in order of first appearance.
 * @param minCount The fewest occurrences kept.
 * @returns The kept entries with their counts.
 */
function counted(words: Map<string, number>, minCount: number): [string, number][] {
  // Most frequent first; ties in order of first appearance.
  return [...words.entries()].filter(([, c]) => c >= minCount).sort((a, b) => b[1] - a[1])
}

/**
 * Train the model of a tokeniser on `texts` and return the tokeniser with it: subword models run their trainer to the
 * end; a word-level vocabulary keeps the most frequent pre-tokens; a character vocabulary every character seen at
 * least `minCount` times. The trainer's specials are added to the tokeniser's.
 *
 * @param t The tokeniser, or its parts before training: its normaliser and pre-tokeniser cut the texts, and its
 *   other stages are kept.
 * @param texts The training texts.
 * @param trainer What to train.
 * @returns The tokeniser with the trained model.
 *
 * @example BPE trained on two sentences
 * const parts = { preTokeniser: whitespacePreTokeniser(), decoder: fuseDecoder() }
 * const t = trainTokeniser(parts, ['low lower lowest', 'low low lower'], { type: 'bpe', vocabularySize: 12 })
 * print('vocabulary =', t.model.vocabulary.tokens)
 * print('lowest =', encodeText(t, 'lowest').tokens)
 *
 * @example A word-level vocabulary of the three most frequent words
 * const parts = { preTokeniser: whitespacePreTokeniser() }
 * const t = trainTokeniser(parts, ['the cat sat on the mat', 'the cat'], { type: 'wordLevel', vocabularySize: 4 })
 * print('vocabulary =', t.model.vocabulary.tokens)
 * print(encodeText(t, 'the cat ran').tokens)
 *
 * @example A unigram model behind Metaspace
 * const parts = { preTokeniser: metaspacePreTokeniser(), decoder: metaspaceDecoder() }
 * const t = trainTokeniser(parts, ['hug pug pun bun hugs hug hug pun'], { type: 'unigram', vocabularySize: 14 })
 * const e = encodeText(t, 'hugs bun')
 * print(e.tokens, decodeIds(t, e.ids))
 */
export function trainTokeniser(
  t: UntrainedTokeniser | Tokeniser,
  texts: readonly string[],
  trainer: Trainer,
): Tokeniser {
  const specials = specialsOf(trainer)
  const unknown = unknownOf(trainer)
  let model: TokeniserModel
  if (trainer.type === 'wordLevel') {
    const keep = counted(preTokenCounts(t, texts), trainer.minCount ?? 1)
      .slice(0, Math.max(0, (trainer.vocabularySize ?? Infinity) - specials.length))
      .map(([w]) => w)
    model = wordLevelStage([...specials, ...keep.filter((w) => !specials.includes(w))], { specials, unknown })
  } else if (trainer.type === 'character') {
    const chars = new Map<string, number>()
    for (const [w, c] of preTokenCounts(t, texts)) for (const ch of w) chars.set(ch, (chars.get(ch) ?? 0) + c)
    const keep = counted(chars, trainer.minCount ?? 1).map(([ch]) => ch)
    model = characterStage(keep, { specials, unknown, byteFallback: trainer.byteFallback ?? false })
  } else {
    const steps = trainingSteps(t, texts, trainer)
    model = trainedModel(trainer, run(steps, undefined, 1e6))
  }
  return tokeniser({ ...t, model, specials: [...new Set([...(t.specials ?? []), ...specials])] })
}
