/**
 * Byte-level symbols (Radford et al. 2019, the GPT-2 encoder): each of the 256 byte values is shown as one printable
 * character, so byte-level BPE can work on strings. Printable Latin-1 bytes stand for themselves; the others (control
 * characters, space, NBSP, soft hyphen) are shifted to U+0100 onwards, so a space is "Ġ" and a newline "Ċ".
 */

let toSymbol: string[] | null = null
let fromSymbol: Map<string, number> | null = null

/**
 * The two lookup tables, built on first use and kept: byte value to symbol, and symbol to byte value.
 *
 * @returns `toSymbol`, the 256 symbols indexed by byte value, and `fromSymbol`, its inverse.
 */
function tables(): { toSymbol: string[]; fromSymbol: Map<string, number> } {
  if (toSymbol && fromSymbol) return { toSymbol, fromSymbol }
  const printable = (b: number) => (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
  toSymbol = new Array<string>(256)
  let shifted = 0
  for (let b = 0; b < 256; b++) toSymbol[b] = String.fromCodePoint(printable(b) ? b : 256 + shifted++)
  fromSymbol = new Map(toSymbol.map((s, b) => [s, b]))
  return { toSymbol, fromSymbol }
}

/**
 * The 256 byte symbols, indexed by byte value: the base vocabulary of byte-level BPE.
 *
 * @returns The symbols, entry `b` standing for byte `b`. The array is shared: do not modify it.
 *
 * @example Printable bytes stand for themselves; space and newline are shifted
 * const A = byteAlphabet()
 * print('size =', A.length)
 * print('byte 65 =', A[65], ' byte 32 =', A[32], ' byte 10 =', A[10])
 */
export function byteAlphabet(): readonly string[] {
  return tables().toSymbol
}

const encoder = new TextEncoder()
const decoder = new TextDecoder('utf-8', { fatal: false })

/**
 * The UTF-8 bytes of a word as byte symbols, each with the [start, end) range of the character it belongs to (a
 * multi-byte character's bytes share its range).
 *
 * @param word The word to convert.
 * @returns One entry per UTF-8 byte, in order: its symbol and the range, in UTF-16 code units, of its character in
 *   `word`.
 *
 * @example The two bytes of "é" share its range
 * for (const s of byteSymbols('é!')) print(s.token, [s.start, s.end])
 */
export function byteSymbols(word: string): { token: string; start: number; end: number }[] {
  const { toSymbol: table } = tables()
  const out: { token: string; start: number; end: number }[] = []
  let at = 0
  for (const c of word) {
    for (const b of encoder.encode(c)) out.push({ token: table[b], start: at, end: at + c.length })
    at += c.length
  }
  return out
}

/**
 * Text from a string of byte symbols (invalid UTF-8, e.g. half a character, decodes to U+FFFD).
 *
 * @param symbols Byte symbols run together, as byte-level tokens are joined. A character that is not a byte symbol is
 *   skipped.
 * @returns The text the bytes encode in UTF-8.
 *
 * @example Round trip, and half a character
 * const symbols = byteSymbols(' é').map((s) => s.token).join('')
 * print('symbols =', symbols)
 * print('text =', JSON.stringify(textFromByteSymbols(symbols)))
 * print('first two bytes only =', textFromByteSymbols(symbols.slice(0, 2)))
 */
export function textFromByteSymbols(symbols: string): string {
  const { fromSymbol: table } = tables()
  const bytes: number[] = []
  for (const c of symbols) {
    const b = table.get(c)
    if (b !== undefined) bytes.push(b)
  }
  return decoder.decode(Uint8Array.from(bytes))
}
