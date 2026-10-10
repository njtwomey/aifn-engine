/**
 * Running a program: plain JavaScript compiled with `new Function`, the prelude's names passed in as its parameters,
 * and checking one for syntax errors without running it.
 *
 * - **The result** is the program's own top-level `return` value; without one, the entry function (default `make`)
 *   is called, if the program defines it, and its value is the result.
 * - **Seeds.** Every random call draws from its own child of the run's root stream, keyed by its position in call
 *   order, so the same program and seed give the same data. The root is the run's `seed` option, keyed further by the
 *   program's `seed(s)` when it calls one (so a seed control outside the code still varies the data).
 * - **Errors** come back as values with the line and column where the engine reports them (runtime errors in V8 and
 *   Firefox; V8 gives no position for a syntax error inside `new Function`, so an editor marks those from its own
 *   parse).
 * - **Safety.** `new Function` runs with the page's globals in reach and with no step budget: an infinite loop never
 *   returns. Run untrusted or long programs in a worker that can be terminated (the lab does).
 */
import { child, type Stream } from 'aifn-compute/foundation/random'
import { CONSTANTS, corePrelude, rootStream } from './core-prelude'
import { lookup, type Context, type Prelude } from './prelude'

/** Options of `runProgram`. */
export type RunOptions = {
  /** The run's seed (default 0); a program's `seed(s)` keys it further. */
  seed?: number | string
  /** The names the program can call (default `corePrelude`). */
  prelude?: Prelude
  /** The function called when the program returns nothing (default `make`; null to never call one). */
  entry?: string | null
  /** Arguments of the entry call (default none, so its defaults apply). */
  args?: readonly unknown[]
}

/** An error of a program: its message, the error's name and, where the engine gives it, a 1-based line and column. */
export type RunError = {
  readonly message: string
  readonly name: string
  readonly line?: number
  readonly column?: number
}

/**
 * What a run gives: `ok`, and with it the result `value` (true) or the `error` (false); the `output` lines the program
 * printed, in order, up to the error if there was one; and `ms`, the wall-clock time taken in milliseconds.
 */
export type RunResult =
  | { readonly ok: true; readonly value: unknown; readonly output: readonly string[]; readonly ms: number }
  | { readonly ok: false; readonly error: RunError; readonly output: readonly string[]; readonly ms: number }

// The program runs inside a block (opened on the header's line, so line numbers are unchanged): its own `const stats`
// or `let print` then shadows a prelude name instead of colliding with the parameter of the same name.
const HEADER = '"use strict";{\n'
const FOOTER = '\n}'
const ARGS = '__aifnEntryArgs'
const IDENTIFIER = /^[A-Za-z_$][\w$]*$/

/**
 * The (line, column) of the innermost frame of compiled code in an error's stack, as the engine reports it.
 *
 * @param err The thrown value; anything that is not an `Error` has no stack.
 * @returns The 1-based line and column within the compiled function's source (header included), or null when the
 *   stack has no frame of compiled code.
 */
function framePosition(err: unknown): { line: number; column: number } | null {
  const stack = err instanceof Error ? (err.stack ?? '') : ''
  // V8: "(eval at …, <anonymous>:4:13)"; Firefox: "… > Function:4:13".
  const m = /(?:<anonymous>|Function):(\d+):(\d+)/.exec(stack)
  return m ? { line: Number(m[1]), column: Number(m[2]) } : null
}

/** How many lines the engine puts before the program's first line (measured once, as engines differ). */
const LINE_OFFSET: number | null = (() => {
  try {
    new Function('a', `${HEADER}throw new Error('probe')${FOOTER}`)(0)
  } catch (err) {
    const at = framePosition(err)
    if (at) return at.line - 1
  }
  return null
})()

/**
 * An error as a value, its position mapped to the program's own lines. Syntax errors, and positions outside the
 * program (in the appended entry call, or in the prelude's own code), get no line or column.
 *
 * @param err The thrown value; a non-`Error` becomes an `Error`-named message of its string form.
 * @param lines The number of lines of the program, to reject positions past its end.
 * @returns The message, the error's name and, when the position falls on the program, its line and column.
 */
function toRunError(err: unknown, lines: number): RunError {
  if (!(err instanceof Error)) return { message: String(err), name: 'Error' }
  const at = err instanceof SyntaxError ? null : framePosition(err)
  const line = at && LINE_OFFSET !== null ? at.line - LINE_OFFSET : undefined
  const base = { message: err.message, name: err.name }
  if (line === undefined || line < 1 || line > lines) return base
  return { ...base, line, column: at!.column }
}

/**
 * The names a program sees: each namespace as a frozen object, the top-level entries, the constants, `Math` and
 * `console`. Entries and namespaces whose names are not identifiers are left out.
 *
 * @param prelude The prelude whose entries become the names.
 * @param ctx The run's context, bound into every entry's implementation.
 * @returns The names and their values, in the order they become the compiled function's parameters.
 */
function scope(prelude: Prelude, ctx: Context): Map<string, unknown> {
  const names = new Map<string, unknown>()
  const spaces = new Map<string, Record<string, unknown>>(prelude.namespaces.map((n) => [n.name, {}]))
  for (const e of prelude.entries) {
    if (!IDENTIFIER.test(e.name)) continue
    const f = (...args: unknown[]) => e.impl(ctx, ...args)
    if (e.namespace === null) names.set(e.name, f)
    else spaces.get(e.namespace)![e.name] = f
  }
  for (const [k, v] of spaces) if (IDENTIFIER.test(k)) names.set(k, Object.freeze(v))
  for (const [k, v] of Object.entries(CONSTANTS)) if (!names.has(k)) names.set(k, v)
  // `Math.sin` is `math.sin` (so it maps over arrays), and `Math.random` is seeded.
  const math: Record<string, unknown> = {}
  const own = spaces.get('math') ?? {}
  for (const k of Object.getOwnPropertyNames(Math)) {
    const native = (Math as unknown as Record<string, unknown>)[k]
    math[k] = typeof native === 'function' && k in own ? own[k] : native
  }
  const uniform = lookup(prelude, 'random.uniform')
  if (uniform) math.random = () => uniform.impl(ctx)
  names.set('Math', Object.freeze(math))
  const print = names.get('print')
  if (print) names.set('console', Object.freeze({ log: print, info: print, warn: print, error: print }))
  return names
}

/**
 * A program's syntax error, or null when it compiles (it is not run). An editor calls it as the user types; V8 gives
 * no position for a syntax error inside `new Function`, so the error has no line there.
 *
 * @param source The program's source text.
 * @param prelude The names the program may use (default `corePrelude`); they are compiled as parameters, so a program
 *   that redeclares one still compiles.
 * @returns The syntax error, or null.
 *
 * @example A program that compiles and one that does not
 * print('fine:', checkProgram('const x = math.sin(1)\nreturn x'))
 * print('broken:', checkProgram('return 1 +'))
 */
export function checkProgram(source: string, prelude: Prelude = corePrelude): RunError | null {
  try {
    const names = [...scope(prelude, { draw: () => rootStream(0), seed: () => {}, log: () => {} }).keys()]
    new Function(...names, ARGS, `${HEADER}${source}${FOOTER}`)
    return null
  } catch (err) {
    return toRunError(err, source.split('\n').length)
  }
}

/**
 * Run a program and return its result (see the file comment): the top-level `return` value, or the entry function's
 * value when it returns nothing. Errors, the program's own included, are returned in the result, never thrown.
 *
 * @param source The program's source text.
 * @param options The seed, the prelude, the entry function and its arguments (see `RunOptions`).
 * @returns The result or the error, the printed output and the time taken.
 *
 * @example A two-line program and its value
 * const r = runProgram('const x = array.linspace(0, 1, 5)\nreturn x.map((v) => v * v)')
 * print('ok:', r.ok, 'value:', r.value)
 *
 * @example The entry function, its defaults and its arguments
 * const program = 'seed(7)\nfunction make(n = 3) { return random.normal(n) }'
 * print('make():', runProgram(program).value)
 * print('make(2):', runProgram(program, { args: [2] }).value)
 * print('same seed, same data:', runProgram(program).value)
 *
 * @example Printed output, and an error with its line
 * const r = runProgram('print("before")\nconst c = array.column([[1, 2], [3, 4]], 5)\nprint("after")')
 * print('output:', r.output)
 * print('error:', r.error.name, 'on line', r.error.line, '-', r.error.message)
 */
export function runProgram(source: string, options: RunOptions = {}): RunResult {
  const { seed = 0, prelude = corePrelude, entry = 'make', args = [] } = options
  const t0 = performance.now()
  const output: string[] = []
  let root: Stream = rootStream(seed)
  let calls = 0
  const ctx: Context = {
    draw: () => child(root, calls++),
    seed: (s) => {
      root = rootStream(seed, s)
      calls = 0
    },
    log: (line) => {
      output.push(line)
    },
  }
  const lines = source.split('\n').length
  const ms = () => performance.now() - t0
  try {
    if (entry !== null && !IDENTIFIER.test(entry)) throw new TypeError(`runProgram: '${entry}' is not a function name`)
    const names = scope(prelude, ctx)
    const tail = entry === null ? '' : `\n;return typeof ${entry} === 'function' ? ${entry}(...${ARGS}) : undefined`
    const program = new Function(...names.keys(), ARGS, `${HEADER}${source}${tail}${FOOTER}`)
    const value: unknown = program(...names.values(), args)
    return { ok: true, value, output, ms: ms() }
  } catch (err) {
    return { ok: false, error: toRunError(err, lines), output, ms: ms() }
  }
}
