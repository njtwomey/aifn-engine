/**
 * Byte-level symbols (Radford et al. 2019, the GPT-2 encoder): each of the 256 byte values is shown as one printable
 * character, so byte-level BPE can work on strings. Printable Latin-1 bytes stand for themselves; the others (control
 * characters, space, NBSP, soft hyphen) are shifted to U+0100 onwards, so a space is "Ġ" and a newline "Ċ".
 */

let toSymbol: string[] | null = null
let fromSymbol: Map<string, number> | null = null

function tables(): { toSymbol: string[]; fromSymbol: Map<string, number> } {
  if (toSymbol && fromSymbol) return { toSymbol, fromSymbol }
  const printable = (b: number) => (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
  toSymbol = new Array<string>(256)
  let shifted = 0
  for (let b = 0; b < 256; b++) toSymbol[b] = String.fromCodePoint(printable(b) ? b : 256 + shifted++)
  fromSymbol = new Map(toSymbol.map((s, b) => [s, b]))
  return { toSymbol, fromSymbol }
}

/** The 256 byte symbols, indexed by byte value. */
export function byteAlphabet(): readonly string[] {
  return tables().toSymbol
}

const encoder = new TextEncoder()
const decoder = new TextDecoder('utf-8', { fatal: false })

/**
 * The UTF-8 bytes of a word as byte symbols, each with the [start, end) range of the character it belongs to (a
 * multi-byte character's bytes share its range).
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

/** Text from a string of byte symbols (invalid UTF-8, e.g. half a character, decodes to U+FFFD). */
export function textFromByteSymbols(symbols: string): string {
  const { fromSymbol: table } = tables()
  const bytes: number[] = []
  for (const c of symbols) {
    const b = table.get(c)
    if (b !== undefined) bytes.push(b)
  }
  return decoder.decode(Uint8Array.from(bytes))
}
