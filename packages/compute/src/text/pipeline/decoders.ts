/**
 * Decoders, the last stage of a tokeniser pipeline: they turn the token strings of an encoding back into text, undoing
 * what the pre-tokeniser and model did (byte symbols back to UTF-8, "▁" back to spaces, "##" continuations joined,
 * `<0xNN>` byte tokens reassembled). Each maps a list of token strings to a list of strings, as Hugging Face's
 * `decode_chain`; the decoded text is their concatenation.
 */

import { byteAlphabet } from 'aifn-compute/text/subword'
import { byteOfToken } from './models'

/**
 * A decoder stage, tagged by `type`; its fields are the arguments of the factory of the same name (`metaspace`:
 * `replacement` and `prependScheme`; `wordPiece`: `prefix` and `cleanup`; `strip`: `content`, `left` and `right`;
 * `replace`: `pattern` and `content`; `endOfWord`: `suffix`; `sequence`: `decoders`).
 */
export type Decoder =
  | { readonly type: 'byteLevel' }
  | { readonly type: 'metaspace'; readonly replacement: string; readonly prependScheme: 'always' | 'first' | 'never' }
  | { readonly type: 'wordPiece'; readonly prefix: string; readonly cleanup: boolean }
  | { readonly type: 'byteFallback' }
  | { readonly type: 'fuse' }
  | { readonly type: 'strip'; readonly content: string; readonly left: number; readonly right: number }
  | { readonly type: 'replace'; readonly pattern: string; readonly content: string }
  | { readonly type: 'endOfWord'; readonly suffix: string }
  | { readonly type: 'sequence'; readonly decoders: readonly Decoder[] }

/**
 * Byte symbols back to bytes, decoded as UTF-8 (invalid sequences become U+FFFD). The tokens are joined first, so the
 * result is one string; a character that is not a byte symbol is kept as its own UTF-8 bytes.
 *
 * @returns The decoder.
 *
 * @example GPT-2 tokens back to text, and half a character
 * print(applyDecoder(byteLevelDecoder(), ['Hey', 'Ġfriend', ',', 'ĠcafÃ©', '!']))
 * print(applyDecoder(byteLevelDecoder(), ['caf', 'Ã']))
 */
export function byteLevelDecoder(): Decoder {
  return { type: 'byteLevel' }
}

/**
 * "▁" back to spaces, dropping the space the pre-tokeniser put before the first word.
 *
 * @param options `replacement` (default "▁"), and `prependScheme` (default `always`): unless `never`, a leading space
 *   of the first token is dropped.
 * @returns The decoder.
 *
 * @example SentencePiece pieces back to text
 * print(JSON.stringify(applyDecoder(metaspaceDecoder(), ['▁Hey', '▁fri', 'end', '!'])))
 * print(JSON.stringify(applyDecoder(metaspaceDecoder({ prependScheme: 'never' }), ['▁Hey', '▁friend'])))
 */
export function metaspaceDecoder(
  options: { replacement?: string; prependScheme?: 'always' | 'first' | 'never' } = {},
): Decoder {
  return {
    type: 'metaspace',
    replacement: options.replacement ?? '▁',
    prependScheme: options.prependScheme ?? 'always',
  }
}

/**
 * Join "##" continuations to the piece before and space the rest; `cleanup` removes spaces before punctuation.
 *
 * @param options `prefix` (default `##`), and `cleanup` (default true), Hugging Face's clean-up: no space before
 *   `.`, `?`, `!` and `,`, and contractions such as `n't` and `'s` rejoined.
 * @returns The decoder.
 *
 * @example With and without the clean-up
 * print(applyDecoder(wordPieceDecoder(), ['the', 'mat', '##s', ',', 'they', 'do', "n't", '.']))
 * print(applyDecoder(wordPieceDecoder({ cleanup: false }), ['the', 'mat', '##s', '.']))
 */
export function wordPieceDecoder(options: { prefix?: string; cleanup?: boolean } = {}): Decoder {
  return { type: 'wordPiece', prefix: options.prefix ?? '##', cleanup: options.cleanup ?? true }
}

/**
 * Runs of `<0xNN>` byte tokens back to the characters their bytes encode. A run that is not valid UTF-8 becomes one
 * U+FFFD per byte; other tokens pass through.
 *
 * @returns The decoder.
 *
 * @example A complete character, and a lone byte
 * print(applyDecoder(byteFallbackDecoder(), ['caf', '<0xC3>', '<0xA9>']))
 * print(applyDecoder(byteFallbackDecoder(), ['a', '<0xC3>', 'b']))
 */
export function byteFallbackDecoder(): Decoder {
  return { type: 'byteFallback' }
}

/**
 * Join every token into one string.
 *
 * @returns The decoder.
 *
 * @example Fused, against no decoder (joined by spaces)
 * print(applyDecoder(fuseDecoder(), ['lo', 'w', 'er']))
 * print(applyDecoder(null, ['lo', 'w', 'er']))
 */
export function fuseDecoder(): Decoder {
  return { type: 'fuse' }
}

/**
 * Remove up to `left` copies of `content` from the start of each token and `right` from its end.
 *
 * @param content The string to remove.
 * @param left The most copies removed from the start of each token.
 * @param right The most copies removed from the end of each token.
 * @returns The decoder.
 *
 * @example One leading space off each token, and two trailing underscores
 * print(JSON.stringify(applyDecoder(stripDecoder(' ', 1, 0), [' Hey', ' friend'])))
 * print(applyDecoder(stripDecoder('_', 0, 2), ['ab__', 'c___']))
 */
export function stripDecoder(content: string, left: number, right: number): Decoder {
  return { type: 'strip', content, left, right }
}

/**
 * Replace a pattern (regular-expression source) in each token.
 *
 * @param pattern The regular expression's source, compiled with the `gu` flags.
 * @param content The literal replacement.
 * @returns The decoder.
 *
 * @example The SentencePiece marker to spaces
 * print(JSON.stringify(applyDecoder(replaceDecoder('▁', ' '), ['▁Hey', '▁friend'])))
 */
export function replaceDecoder(pattern: string, content: string): Decoder {
  return { type: 'replace', pattern, content }
}

/**
 * The end-of-word symbol of character-level BPE ("</w>") becomes a space, and nothing after the last token.
 *
 * @param suffix The end-of-word symbol.
 * @returns The decoder.
 *
 * @example Sennrich-style tokens back to words
 * print(JSON.stringify(applyDecoder(endOfWordDecoder(), ['low', 'er</w>', 'low', 'est</w>'])))
 */
export function endOfWordDecoder(suffix = '</w>'): Decoder {
  return { type: 'endOfWord', suffix }
}

/**
 * Decoders applied in order, each to the token strings the one before returned.
 *
 * @param decoders The decoders, first applied first.
 * @returns The decoder.
 *
 * @example LLaMA's decoder: markers to spaces, bytes back, fused, the first space stripped
 * const d = decoderSequence(replaceDecoder('▁', ' '), byteFallbackDecoder(), fuseDecoder(), stripDecoder(' ', 1, 0))
 * print(JSON.stringify(applyDecoder(d, ['▁caf', '<0xC3>', '<0xA9>', '▁ok'])))
 */
export function decoderSequence(...decoders: Decoder[]): Decoder {
  return { type: 'sequence', decoders }
}

const utf8 = new TextDecoder('utf-8', { fatal: true })
const lossy = new TextDecoder('utf-8', { fatal: false })

let fromSymbol: Map<string, number> | null = null

/**
 * Apply one decoder to a list of token strings.
 *
 * @param d The decoder.
 * @param tokens The token strings; not modified.
 * @returns The decoded strings, whose concatenation is the text.
 */
function decodeOne(d: Decoder, tokens: readonly string[]): string[] {
  switch (d.type) {
    case 'byteLevel': {
      fromSymbol ??= new Map(byteAlphabet().map((s, b) => [s, b]))
      const bytes: number[] = []
      for (const c of tokens.join('')) {
        const b = fromSymbol.get(c)
        if (b !== undefined) bytes.push(b)
        else bytes.push(...new TextEncoder().encode(c))
      }
      return [lossy.decode(Uint8Array.from(bytes))]
    }
    case 'metaspace':
      return tokens.map((t, i) => {
        const s = t.split(d.replacement).join(' ')
        return i === 0 && d.prependScheme !== 'never' && s.startsWith(' ') ? s.slice(1) : s
      })
    case 'wordPiece':
      return tokens.map((t, i) => {
        let s = i > 0 ? (t.startsWith(d.prefix) ? t.slice(d.prefix.length) : ' ' + t) : t
        if (d.cleanup) s = cleanup(s)
        return s
      })
    case 'byteFallback': {
      const out: string[] = []
      let run: number[] = []
      const flush = () => {
        if (run.length === 0) return
        try {
          out.push(utf8.decode(Uint8Array.from(run)))
        } catch {
          for (let k = 0; k < run.length; k++) out.push('�')
        }
        run = []
      }
      for (const t of tokens) {
        const b = byteOfToken(t)
        if (b >= 0) run.push(b)
        else {
          flush()
          out.push(t)
        }
      }
      flush()
      return out
    }
    case 'fuse':
      return [tokens.join('')]
    case 'strip':
      return tokens.map((t) => {
        let s = t
        for (let k = 0; k < d.left && s.startsWith(d.content); k++) s = s.slice(d.content.length)
        for (let k = 0; k < d.right && s.endsWith(d.content); k++) s = s.slice(0, s.length - d.content.length)
        return s
      })
    case 'replace': {
      const re = new RegExp(d.pattern, 'gu')
      return tokens.map((t) => t.replace(re, () => d.content))
    }
    case 'endOfWord':
      return tokens.map((t, i) => t.split(d.suffix).join(i === tokens.length - 1 ? '' : ' '))
    case 'sequence':
      return d.decoders.reduce<string[]>((ts, e) => decodeOne(e, ts), [...tokens])
  }
}

// Hugging Face's WordPiece clean-up, applied to each decoded token.
/**
 * The clean-up of `wordPieceDecoder`: no space before `.`, `?`, `!` and `,`, and contractions rejoined.
 *
 * @param s A decoded token, with the space put before it.
 * @returns It with the spaces before punctuation and contractions removed.
 */
function cleanup(s: string): string {
  return s
    .replaceAll(' .', '.')
    .replaceAll(' ?', '?')
    .replaceAll(' !', '!')
    .replaceAll(' ,', ',')
    .replaceAll(" ' ", "'")
    .replaceAll(" n't", "n't")
    .replaceAll(" 'm", "'m")
    .replaceAll(' do not', " don't")
    .replaceAll(" 's", "'s")
    .replaceAll(" 've", "'ve")
    .replaceAll(" 're", "'re")
}

/**
 * Decode token strings to text with a decoder (none: the tokens joined by spaces, as Hugging Face does).
 *
 * @param d The decoder, or null.
 * @param tokens The token strings, in order.
 * @returns The text.
 *
 * @example WordPiece tokens back to text
 * print(applyDecoder(wordPieceDecoder(), ['un', '##want', '##ed', 'cat', '##s', '!']))
 */
export function applyDecoder(d: Decoder | null, tokens: readonly string[]): string {
  return d ? decodeOne(d, tokens).join('') : tokens.join(' ')
}
