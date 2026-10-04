/**
 * Training a tokeniser's model on a corpus through its own normaliser and pre-tokeniser: the corpus is cut into
 * pre-tokens exactly as encoding will cut it, counted, and handed to the step-through trainers of `aifn-compute/text/subword`
 * (BPE merges, WordPiece merges, unigram pruning rounds); word-level and character vocabularies are counted directly.
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

/** What to train and its options. */
export type Trainer =
  | {
      readonly type: 'bpe'
      readonly vocabularySize: number
      readonly specials?: readonly string[]
      readonly unknown?: string | null
      /** Add the 256 `<0xNN>` tokens and fall back to them (SentencePiece, LLaMA). */
      readonly byteFallback?: boolean
      /** Every byte symbol in the alphabet, seen or not (byte-level BPE, as GPT-2). */
      readonly byteAlphabet?: boolean
      readonly endOfWord?: string
      readonly minCount?: number
      readonly dropout?: number
    }
  | {
      readonly type: 'wordPiece'
      readonly vocabularySize: number
      readonly specials?: readonly string[]
      readonly prefix?: string
      readonly criterion?: 'ratio' | 'likelihood'
    }
  | {
      readonly type: 'unigram'
      readonly vocabularySize: number
      readonly specials?: readonly string[]
      readonly unknown?: string | null
      readonly byteFallback?: boolean
      readonly maxPieceLength?: number
      readonly shrink?: number
    }
  | {
      readonly type: 'wordLevel'
      readonly vocabularySize?: number
      readonly minCount?: number
      readonly specials?: readonly string[]
      readonly unknown?: string | null
    }
  | {
      readonly type: 'character'
      readonly minCount?: number
      readonly specials?: readonly string[]
      readonly unknown?: string | null
      readonly byteFallback?: boolean
    }

/** The training state of a subword trainer. */
export type TrainerState = BpeState | WordPieceState | UnigramLmState

const specialsOf = (tr: Trainer) =>
  tr.specials ?? (tr.type === 'wordPiece' ? ['[UNK]'] : tr.type === 'bpe' && !tr.byteFallback ? [] : ['<unk>'])
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
 * or one pruning round, per step. Pass a state to {@link trainedModel} to get the model at that step.
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

/** The model of a subword trainer at a training state. */
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

function counted(words: Map<string, number>, minCount: number): [string, number][] {
  // Most frequent first; ties in order of first appearance.
  return [...words.entries()].filter(([, c]) => c >= minCount).sort((a, b) => b[1] - a[1])
}

/**
 * Train the model of a tokeniser on `texts` and return the tokeniser with it: subword models run their trainer to the
 * end; a word-level vocabulary keeps the most frequent pre-tokens; a character vocabulary every character seen at
 * least `minCount` times.
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
