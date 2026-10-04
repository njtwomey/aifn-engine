/**
 * Running a documentation example: the code is compiled with `new Function`, with the exports of the module it
 * documents and the common surface of `aifn-compute` (tensors, `grad`, streams, runners) in scope, as if imported by
 * name. It reads like a notebook cell: every statement that is an expression on a line of its own shows its value,
 * and `print` adds lines of output.
 *
 * Examples run on the page's own thread, so an edit that loops forever hangs the tab; they are meant to be small.
 */
import * as base from 'aifn-compute'
import { isTensor, shapeOf, toArray, type Tensor } from 'aifn-compute/foundation/tensor'
import type { DocPackage } from './data'

const MODULES = import.meta.glob([
  '../../../packages/compute/src/**/index.ts',
  '../../../packages/methods/src/**/index.ts',
  '!**/_*/**',
]) as Record<string, () => Promise<Record<string, unknown>>>

const loadModule = (pkg: DocPackage, path: string) => MODULES[`../../../packages/${pkg}/src/${path}/index.ts`]()

export type Scope = { names: Record<string, unknown>; own: ReadonlySet<string> }

/** What an example of a node can name: the node's exports over the common surface. */
export async function scopeOf(pkg: DocPackage, path: string): Promise<Scope> {
  const own = await loadModule(pkg, path)
  return { names: { ...base, ...own }, own: new Set(Object.keys(own)) }
}

const IDENTIFIER = /^[A-Za-z_$][\w$]*$/

/** The import lines the example stands for: the names it uses, from the node and from the common surface. */
export function importsOf(code: string, scope: Scope, specifier: string): string[] {
  // Names in comments, in strings and after a dot are not references to the scope.
  const bare = code
    .replace(/\/\/.*$/gm, '')
    .replace(/'[^'\n]*'|"[^"\n]*"|`[^`]*`/g, '')
    .replace(/\.[A-Za-z_$][\w$]*/g, '')
  const used = new Set(bare.match(/[A-Za-z_$][\w$]*/g) ?? [])
  const names = Object.keys(scope.names).filter((n) => used.has(n))
  const own = names.filter((n) => scope.own.has(n)).sort()
  const common = names.filter((n) => !scope.own.has(n)).sort()
  return [
    ...(common.length ? [`import { ${common.join(', ')} } from 'aifn-compute'`] : []),
    ...(own.length ? [`import { ${own.join(', ')} } from '${specifier}'`] : []),
  ]
}

const round = (v: number) => (Number.isInteger(v) || !Number.isFinite(v) ? String(v) : String(Number(v.toPrecision(6))))

function nested(v: unknown): string {
  if (typeof v === 'number') return round(v)
  if (Array.isArray(v)) return `[${v.map(nested).join(', ')}]`
  return String(v)
}

/** A value as a short text: numbers to six significant digits, tensors with their shape, objects by field. */
export function format(v: unknown, depth = 0): string {
  if (typeof v === 'number') return round(v)
  if (typeof v === 'string') return depth ? JSON.stringify(v) : v
  if (v === null || v === undefined || typeof v === 'boolean' || typeof v === 'bigint') return String(v)
  if (typeof v === 'function') return `[function ${v.name || 'anonymous'}]`
  if (isTensor(v)) {
    const t = v as Tensor
    const shape = shapeOf(t)
    if (shape.length === 0) return round(toArray(t) as number)
    const size = shape.reduce((a, b) => a * b, 1)
    return size > 200 ? `Tensor [${shape.join(' × ')}]` : nested(toArray(t))
  }
  if (ArrayBuffer.isView(v)) return format(Array.from(v as unknown as ArrayLike<number>), depth)
  if (Array.isArray(v)) {
    if (v.length > 50)
      return `[${v
        .slice(0, 50)
        .map((x) => format(x, depth + 1))
        .join(', ')}, … ${v.length} items]`
    return `[${v.map((x) => format(x, depth + 1)).join(', ')}]`
  }
  if (depth >= 3) return '{…}'
  const fields = Object.entries(v as object).filter(([, x]) => typeof x !== 'function')
  if (!fields.length) return Object.prototype.toString.call(v)
  const pad = '  '.repeat(depth + 1)
  return `{\n${fields.map(([k, x]) => `${pad}${k}: ${format(x, depth + 1)}`).join(',\n')}\n${'  '.repeat(depth)}}`
}

export type ExampleResult = {
  ok: boolean
  value: string
  output: string[]
  error: string
  ms: number
}

/** Is the line a whole expression (not a declaration, a block, a comment or part of a longer statement)? */
function isExpression(line: string): boolean {
  if (/^\s|^(const|let|var|return|for|while|if|function|class|throw|try|print)\b|^\/\/|^[})\]]|^$/.test(line))
    return false
  try {
    new Function(`return (${line})`)
    return true
  } catch {
    return false
  }
}

/** The code with each expression line turned into a call that shows its value. */
function shown(code: string): string {
  return code
    .split('\n')
    .map((line) => {
      const expression = line.replace(/\s*\/\/.*$/, '').replace(/;$/, '')
      return isExpression(expression) ? `__show(${JSON.stringify(expression)}, (${expression}))` : line
    })
    .join('\n')
}

/** Run an example in a node's scope. */
export function runExample(code: string, scope: Scope): ExampleResult {
  const output: string[] = []
  const show = (expression: string, value: unknown) => {
    if (value === undefined) return
    const text = format(value)
    output.push(text.includes('\n') ? `${expression} →\n${text}` : `${expression} → ${text}`)
  }
  const print = (...args: unknown[]) => void output.push(args.map((a) => format(a)).join(' '))
  const names = Object.keys(scope.names).filter((n) => IDENTIFIER.test(n) && n !== 'print')
  const source = shown(code)
  const t0 = performance.now()
  const done = (ok: boolean, value: string, error: string): ExampleResult => ({
    ok,
    value,
    output,
    error,
    ms: performance.now() - t0,
  })
  try {
    const fn = new Function(...names, 'print', '__show', `"use strict";\n${source}`)
    const value = fn(...names.map((n) => scope.names[n]), print, show)
    return done(true, value === undefined ? '' : format(value), '')
  } catch (e) {
    return done(false, '', e instanceof Error ? `${e.name}: ${e.message}` : String(e))
  }
}
