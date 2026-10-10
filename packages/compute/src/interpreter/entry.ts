/**
 * The entry function's signature, read from the source without running it, so a page can draw controls for its
 * parameters. A small scanner finds `function make(…)` or `make = (…) =>` and splits the parameter list at its
 * top-level commas (brackets, strings and comments balanced). Only literal defaults become controls; anything else is
 * listed as fixed. Unreadable source gives no parameters, never an error.
 *
 * Types are inferred from the defaults (an integer literal is `int`, a decimal `real`, a boolean `bool`) and refined by
 * an optional JSDoc block above the function, which is ordinary JavaScript:
 *
 * ```js
 * /**
 *  * @param {int} n [10, 1000] number of points
 *  * @param {real:log} rate [0.001, 10] learning rate
 *  * @param {choice} shape ['sine', 'square'] curve shape
 *  *\/
 * function make(n = 200, rate = 0.1, shape = 'sine') { … }
 * ```
 */
import type { DimSpec, Space } from 'aifn-compute/foundation/contracts'

/** The control type of a parameter: a number, a switch, a choice, or `fixed` (no control). */
export type ParamType = 'int' | 'real' | 'bool' | 'choice' | 'fixed'

/** One parameter of the entry function, with what a control for it needs. */
export type EntryParam = {
  /** The parameter's name, or its source text when it is a rest parameter or a destructuring pattern. */
  readonly name: string
  /** The default's source text, if any. */
  readonly source?: string
  /** The default's value when it is a literal. */
  readonly value?: number | boolean | string | readonly number[]
  /** The control type. */
  readonly type: ParamType
  /** The smallest value of an `int` or `real` control. */
  readonly min?: number
  /** The largest value of an `int` or `real` control. */
  readonly max?: number
  /** `'log'` for a `real:log` parameter, whose control moves on a log scale. */
  readonly scale?: 'linear' | 'log'
  /** The options of a `choice` control. */
  readonly options?: readonly (string | number)[]
  /** The JSDoc description. */
  readonly doc?: string
}

/** The entry function's signature: whether it was found, its parameters in order, and a `Space` of the controllable. */
export type EntrySignature = {
  /** The entry function's name that was looked for. */
  readonly entry: string
  /** Whether the source defines it (and its parameter list closes). */
  readonly found: boolean
  /** Its parameters, in order, fixed ones included. */
  readonly params: readonly EntryParam[]
  /** One dimension per parameter with a control (`entrySpace` of `params`). */
  readonly space: Space
}

/** JSDoc types accepted after `@param`, for editors. */
export const PARAM_TYPES = ['int', 'real', 'real:log', 'bool', 'choice'] as const

const OPEN = '([{'
const CLOSE = ')]}'

/**
 * Index just past the end of a string or comment starting at `i`, or `i` when none starts there. A line comment ends
 * at its newline (which is not skipped); an unterminated string or comment runs to the end of the text.
 *
 * @param src The source text.
 * @param i The index to look at.
 * @returns The index of the first character after the string or comment, or `i`.
 */
function skip(src: string, i: number): number {
  const c = src[i]
  if (c === '/' && src[i + 1] === '/') {
    const end = src.indexOf('\n', i)
    return end < 0 ? src.length : end
  }
  if (c === '/' && src[i + 1] === '*') {
    const end = src.indexOf('*/', i + 2)
    return end < 0 ? src.length : end + 2
  }
  if (c === '"' || c === "'" || c === '`') {
    let j = i + 1
    while (j < src.length && src[j] !== c) j += src[j] === '\\' ? 2 : 1
    return Math.min(j + 1, src.length)
  }
  return i
}

/**
 * The text inside the bracket opening at `open` (balanced), or null when it never closes. Brackets of every kind are
 * counted together and those inside strings and comments are ignored.
 *
 * @param src The source text.
 * @param open The index of the opening bracket.
 * @returns The text between the bracket and its match, both excluded, or null.
 */
function inside(src: string, open: number): string | null {
  let depth = 0
  for (let i = open; i < src.length;) {
    const next = skip(src, i)
    if (next !== i) {
      i = next
      continue
    }
    if (OPEN.includes(src[i])) depth++
    else if (CLOSE.includes(src[i]) && --depth === 0) return src.slice(open + 1, i)
    i++
  }
  return null
}

/**
 * The text with its comments removed (strings kept): each comment becomes one space.
 *
 * @param text The source text.
 * @returns The text without comments.
 */
function stripComments(text: string): string {
  let out = ''
  for (let i = 0; i < text.length;) {
    const next = skip(text, i)
    if (next === i) out += text[i++]
    else {
      if (text[i] === '/') out += ' '
      else out += text.slice(i, next)
      i = next
    }
  }
  return out
}

/**
 * Split at top-level commas: those outside brackets, strings and comments.
 *
 * @param text A parameter list's text, without its parentheses.
 * @returns The parts, trimmed, with empty ones (a trailing comma) dropped.
 */
function splitTop(text: string): string[] {
  const parts: string[] = []
  let depth = 0
  let start = 0
  for (let i = 0; i < text.length;) {
    const next = skip(text, i)
    if (next !== i) {
      i = next
      continue
    }
    const c = text[i]
    if (OPEN.includes(c)) depth++
    else if (CLOSE.includes(c)) depth--
    else if (c === ',' && depth === 0) {
      parts.push(text.slice(start, i))
      start = i + 1
    }
    i++
  }
  parts.push(text.slice(start))
  return parts.map((p) => p.trim()).filter((p) => p !== '')
}

const NUMBER = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/

/**
 * A literal's value: a number, boolean, quoted string (single or double quotes) or JSON array of numbers; undefined
 * otherwise.
 *
 * @param text A default's source text.
 * @returns The value, or undefined when the text is not one of those literals.
 */
function literal(text: string): EntryParam['value'] {
  const t = text.trim()
  if (NUMBER.test(t)) return Number(t)
  if (t === 'true' || t === 'false') return t === 'true'
  const q = /^(['"])((?:[^\\]|\\.)*?)\1$/.exec(t)
  if (q) return q[2]
  if (t.startsWith('[')) {
    try {
      const v: unknown = JSON.parse(t)
      if (Array.isArray(v) && v.every((x) => typeof x === 'number')) return v as number[]
    } catch {
      return undefined
    }
  }
  return undefined
}

/**
 * A JSDoc list or range (`[0, 2]`, `['a', 'b']`) as values, or null.
 *
 * @param text The bracketed text; single quotes are read as double quotes and the rest as JSON.
 * @returns The numbers and strings of the list, or null when it is not a JSON array of them.
 */
function list(text: string): (string | number)[] | null {
  try {
    const v: unknown = JSON.parse(text.replace(/'/g, '"'))
    return Array.isArray(v) && v.every((x) => typeof x === 'number' || typeof x === 'string') ? v : null
  } catch {
    return null
  }
}

/**
 * What a JSDoc `@param` line says of a parameter: its `type` (between braces), its `range` or options (the bracketed
 * list after the name) and its `doc` (the rest of the line).
 */
type Doc = { type?: string; range?: (string | number)[]; doc?: string }

/**
 * `@param` lines of the JSDoc block that ends just before `at` (only whitespace, `export` or a `const`, `let` or `var`
 * between).
 *
 * @param src The program's source text.
 * @param at The index where the entry function's definition starts.
 * @returns What each `@param` line says, by parameter name; empty when there is no such block.
 */
function jsdoc(src: string, at: number): Map<string, Doc> {
  const out = new Map<string, Doc>()
  const before = src.slice(0, at).replace(/(?:export\s+)?(?:(?:const|let|var)\s*)?\s*$/, '')
  if (!before.endsWith('*/')) return out
  const start = before.lastIndexOf('/**')
  if (start < 0) return out
  const block = before.slice(start + 3, -2)
  for (const raw of block.split('\n')) {
    const line = raw.replace(/^\s*\*?\s?/, '')
    const m = /^@param\s+(?:\{([^}]*)\}\s*)?([\w$]+)\s*(\[[^\]]*\])?\s*(.*)$/.exec(line.trim())
    if (!m) continue
    const range = m[3] ? list(m[3]) : null
    out.set(m[2], { type: m[1]?.trim(), range: range ?? undefined, doc: m[4]?.trim() || undefined })
  }
  return out
}

/**
 * The default range of a number from its default: for an `int`, $[1, \max(10, 5d)]$ when $d > 0$ and $[-r, r]$
 * with $r = \max(10, 5\lvert d \rvert)$ otherwise; for a `real`, $[0, 2d]$, $[2d, -2d]$ or $[-1, 1]$ as $d$ is
 * positive, negative or zero.
 *
 * @param type The control type.
 * @param d The default value $d$.
 * @returns The range, as `[min, max]`.
 */
function inferredRange(type: 'int' | 'real', d: number): [number, number] {
  if (type === 'int') {
    if (d > 0) return [1, Math.max(10, 5 * d)]
    const r = Math.max(10, 5 * Math.abs(d))
    return [-r, r]
  }
  if (d > 0) return [0, 2 * d]
  if (d < 0) return [2 * d, -2 * d]
  return [-1, 1]
}

/**
 * One parameter from its source text and JSDoc. A declared type wins over the one inferred from the default; a
 * declared range is used when valid (for `real:log`, positive), else the range inferred from the default; and the
 * value is the default clamped to the range (rounded for an `int`), or the range's middle (geometric for `real:log`)
 * when there is no numeric default. A parameter that cannot have a control is `fixed`.
 *
 * @param text The parameter's source text: a name, with `= default` when it has one.
 * @param docs What the JSDoc block says of each parameter, by name.
 * @returns The parameter.
 */
function param(text: string, docs: Map<string, Doc>): EntryParam | null {
  const eq = (() => {
    // The first top-level `=` that is not part of `=>`, `==` or a comparison.
    let depth = 0
    for (let i = 0; i < text.length;) {
      const next = skip(text, i)
      if (next !== i) {
        i = next
        continue
      }
      const c = text[i]
      if (OPEN.includes(c)) depth++
      else if (CLOSE.includes(c)) depth--
      else if (c === '=' && depth === 0 && text[i + 1] !== '=' && text[i + 1] !== '>') return i
      i++
    }
    return -1
  })()
  const name = (eq < 0 ? text : text.slice(0, eq)).trim()
  if (!/^[A-Za-z_$][\w$]*$/.test(name)) {
    // A rest parameter or a destructuring pattern: listed, never a control.
    return { name, type: 'fixed' }
  }
  const source = eq < 0 ? undefined : text.slice(eq + 1).trim()
  const value = source === undefined ? undefined : literal(source)
  const d = docs.get(name)
  const doc = d?.doc
  const declared = d?.type?.toLowerCase()
  const range = d?.range
  const numbers = range && range.length === 2 && range.every((x) => typeof x === 'number') ? (range as number[]) : null
  const base = {
    name,
    ...(source === undefined ? {} : { source }),
    ...(value === undefined ? {} : { value }),
    ...(doc ? { doc } : {}),
  }

  if (declared === 'choice' && range && range.length > 0) {
    const options = range
    const fallback = options.includes(value as string | number) ? value : options[0]
    return { ...base, type: 'choice', options, value: fallback as string | number }
  }
  if (declared === 'bool' || (declared === undefined && typeof value === 'boolean'))
    return typeof value === 'boolean' || value === undefined
      ? { ...base, type: 'bool', value: value ?? false }
      : { ...base, type: 'fixed' }
  const numeric = typeof value === 'number' ? value : undefined
  const kind: 'int' | 'real' | null =
    declared === 'int'
      ? 'int'
      : declared === 'real' || declared === 'real:log'
        ? 'real'
        : declared === undefined && numeric !== undefined
          ? Number.isInteger(numeric) && source !== undefined && !/[.eE]/.test(source)
            ? 'int'
            : 'real'
          : null
  if (kind === null || (numeric === undefined && numbers === null)) return { ...base, type: 'fixed' }
  const log = declared === 'real:log'
  let [min, max] = numbers ?? inferredRange(kind, numeric ?? 0)
  if (kind === 'int') [min, max] = [Math.ceil(min), Math.floor(max)]
  if (!(max >= min) || (log && !(min > 0))) {
    if (numeric === undefined) return { ...base, type: 'fixed' }
    ;[min, max] = inferredRange(kind, numeric)
  }
  const raw = numeric ?? (log ? Math.sqrt(min * max) : (min + max) / 2)
  const v = Math.min(max, Math.max(min, kind === 'int' ? Math.round(raw) : raw))
  return { ...base, type: kind, value: v, min, max, ...(log ? { scale: 'log' as const } : {}) }
}

/**
 * A `Space` of the parameters that have controls: one dimension per `int`, `real`, `bool` or `choice` parameter, its
 * default the parameter's value; `fixed` ones are left out.
 *
 * @param params The parameters, as `entrySignature` reads them.
 * @returns The space, its dimensions keyed by parameter name.
 *
 * @example Controls for the parameters that have them
 * const { params } = entrySignature('function make(n = 20, noise = 0.5, f = Math.sin) {}')
 * print(entrySpace(params).dims)
 */
export function entrySpace(params: readonly EntryParam[]): Space {
  const dims: Record<string, DimSpec> = {}
  for (const p of params) {
    const doc = p.doc ? { doc: p.doc } : {}
    if (p.type === 'int') dims[p.name] = { type: 'int', min: p.min!, max: p.max!, default: p.value as number, ...doc }
    else if (p.type === 'real')
      dims[p.name] = {
        type: 'real',
        min: p.min!,
        max: p.max!,
        default: p.value as number,
        ...(p.scale ? { scale: p.scale } : {}),
        ...doc,
      }
    else if (p.type === 'bool') dims[p.name] = { type: 'bool', default: p.value as boolean, ...doc }
    else if (p.type === 'choice')
      dims[p.name] = { type: 'choice', options: p.options!, default: p.value as string | number, ...doc }
  }
  return { dims }
}

/**
 * A name with each `$` escaped, for use in a regular expression.
 *
 * @param s The name.
 * @returns The escaped name.
 */
const escape = (s: string) => s.replace(/[$]/g, '\\$')

/**
 * The signature of the entry function in `source`, read without running it; `found: false` when there is none to
 * read. It finds the first `function make(…)`, and failing that the first `make = (…) =>` or
 * `make = function (…)`.
 *
 * @param source The program's source text.
 * @param entry The entry function's name (default `make`); a name that is not an identifier is never found.
 * @returns The entry's name, whether it was found, its parameters and the `Space` of those with controls.
 *
 * @example Types inferred from the defaults
 * const sig = entrySignature('function make(n = 200, noise = 0.3, smooth = true, f = Math.sin) {}')
 * for (const p of sig.params) print(p.name, p.type, p.value, p.min, p.max)
 *
 * @example Types and ranges from a JSDoc block
 * const source = `
 * /** @param {real:log} rate [0.001, 10] learning rate
 *  * @param {choice} shape ['sine', 'square'] curve shape *\/
 * function make(rate = 0.1, shape = 'sine') {}`
 * for (const p of entrySignature(source).params) print(p.name, p.type, p.value, p.scale ?? p.options, '-', p.doc)
 */
export function entrySignature(source: string, entry = 'make'): EntrySignature {
  const none: EntrySignature = { entry, found: false, params: [], space: { dims: {} } }
  if (!/^[A-Za-z_$][\w$]*$/.test(entry)) return none
  const name = escape(entry)
  const patterns = [
    new RegExp(`\\bfunction\\s+${name}\\s*\\(`, 'g'),
    new RegExp(`(?:^|[^\\w$.])${name}\\s*=\\s*(?:async\\s+)?(?:function\\b[^(]*)?\\(`, 'g'),
  ]
  for (const re of patterns) {
    const m = re.exec(source)
    if (!m) continue
    const open = m.index + m[0].length - 1
    const text = inside(source, open)
    if (text === null) return none
    const start = m.index + (m[0].length - m[0].trimStart().length) + (/^[^\w$]/.test(m[0].trimStart()) ? 1 : 0)
    const docs = jsdoc(source, start)
    const params = splitTop(stripComments(text))
      .map((p) => param(p, docs))
      .filter((p): p is EntryParam => p !== null)
    return { entry, found: true, params, space: entrySpace(params) }
  }
  return none
}
