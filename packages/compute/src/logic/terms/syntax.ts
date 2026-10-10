/**
 * Reading Prolog text: a tokenizer and an operator-precedence parser for a practical subset of ISO Prolog.
 *
 * Supported: atoms (`mary`, `'New York'`, symbolic atoms such as `=<`), integers and decimals, variables (`X`, `_Y`,
 * and `_`, which is fresh at each occurrence), compound terms, lists `[a, b | T]`, the standard operators (`:-`, `;`,
 * `->`, `,`, `\+`, `=`, `\=`, `==`, `\==`, `is`, the arithmetic comparisons, `+ - * / // mod rem ** ^`, prefix `-`),
 * `%` and `/* … *\/` comments. A program is a sequence of clauses, each ending in a full stop; `?- Goal.` is a query.
 * Each clause or query numbers its variables 0, 1, … in order of first occurrence. Errors are `PrologSyntaxError`s,
 * which carry a line and column.
 */
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  atom,
  compound,
  INFIX,
  listTerm,
  numeral,
  PREFIX,
  termToString,
  variable,
  type PrintOptions,
  type Term,
} from './terms'

/**
 * A syntax error, with the 1-based line and column of the token it was found at; its message ends with both.
 *
 * @example Where a program fails to parse
 * try {
 *   parseProgram('p(a).\nq(b :- c.')
 * } catch (e) {
 *   print(e.name, e.message)
 *   print('line', e.line, 'column', e.column)
 * }
 */
export class PrologSyntaxError extends DomainError {
  /** The 1-based line of the offending token. */
  readonly line: number
  /** The 1-based column of the offending token. */
  readonly column: number
  constructor(message: string, line: number, column: number) {
    super('parseProgram', `${message} (line ${line}, column ${column})`)
    this.name = 'PrologSyntaxError'
    this.line = line
    this.column = column
  }
}

/**
 * What a token is: a `name` (an atom or functor, quoted or not, including symbol runs, `!` and `;`), a `var`, a
 * `number`, `punct` (brackets, `,` and `|`), the `end` full stop of a clause, or `eof` after the last token.
 */
type TokenKind = 'name' | 'var' | 'number' | 'punct' | 'end' | 'eof'
/** A token of Prolog text, with where it starts. */
interface Token {
  /** What the token is. */
  kind: TokenKind
  /** Its text; for a quoted atom, the name with the quotes and escapes resolved. */
  text: string
  /** The value of a `number` token. */
  value?: number
  /** True when a `(` follows immediately: a functor, not an operator applied to a bracketed term. */
  functional?: boolean
  /** True when the token was quoted: never an operator. */
  quoted?: boolean
  /** The 1-based line it starts on. */
  line: number
  /** The 1-based column it starts at. */
  column: number
  /** The 0-based offset of its first character in the text. */
  offset: number
}

/** A symbol character: runs of them make atoms such as `=<` and `:-`. */
const SYMBOL = /[+\-*/\\^<>=~:.?@#&$]/
/** A character that continues a name or variable. */
const ALNUM = /[A-Za-z0-9_]/

/**
 * Split Prolog text into tokens. White space and comments are skipped; a full stop followed by white space, `%` or
 * the end of the text is an `end` token; single- and double-quoted text is a quoted atom (`''` and backslash escapes
 * inside). Throws `PrologSyntaxError` for an unterminated comment or quoted atom and for a character that starts no
 * token.
 *
 * @param src The Prolog text.
 * @returns The tokens in order, ending with an `eof` token.
 */
function tokenize(src: string): Token[] {
  const out: Token[] = []
  let i = 0
  let line = 1
  let lineStart = 0
  const pos = (at: number) => ({ line, column: at - lineStart + 1, offset: at })
  const advanceLines = (from: number, to: number) => {
    for (let k = from; k < to; k++)
      if (src[k] === '\n') {
        line++
        lineStart = k + 1
      }
  }
  while (i < src.length) {
    const c = src[i]
    if (c === '\n' || c === ' ' || c === '\t' || c === '\r') {
      advanceLines(i, i + 1)
      i++
      continue
    }
    if (c === '%') {
      while (i < src.length && src[i] !== '\n') i++
      continue
    }
    if (c === '/' && src[i + 1] === '*') {
      const end = src.indexOf('*/', i + 2)
      if (end < 0) throw new PrologSyntaxError('unterminated comment', pos(i).line, pos(i).column)
      advanceLines(i, end + 2)
      i = end + 2
      continue
    }
    const start = pos(i)
    const push = (kind: TokenKind, text: string, extra: Partial<Token> = {}) => {
      out.push({ kind, text, ...start, ...extra })
    }
    if (/[0-9]/.test(c)) {
      let j = i
      while (j < src.length && /[0-9]/.test(src[j])) j++
      if (src[j] === '.' && /[0-9]/.test(src[j + 1] ?? '')) {
        j++
        while (j < src.length && /[0-9]/.test(src[j])) j++
      }
      if ((src[j] === 'e' || src[j] === 'E') && /[0-9+-]/.test(src[j + 1] ?? '') && src.slice(i, j).includes('.')) {
        j += 2
        while (j < src.length && /[0-9]/.test(src[j])) j++
      }
      push('number', src.slice(i, j), { value: Number(src.slice(i, j)) })
      i = j
    } else if (/[A-Z_]/.test(c)) {
      let j = i
      while (j < src.length && ALNUM.test(src[j])) j++
      push('var', src.slice(i, j))
      i = j
    } else if (/[a-z]/.test(c)) {
      let j = i
      while (j < src.length && ALNUM.test(src[j])) j++
      push('name', src.slice(i, j), { functional: src[j] === '(' })
      i = j
    } else if (c === "'" || c === '"') {
      let j = i + 1
      let text = ''
      for (;;) {
        if (j >= src.length) throw new PrologSyntaxError('unterminated quoted atom', start.line, start.column)
        const d = src[j]
        if (d === c) {
          if (src[j + 1] === c) {
            text += c
            j += 2
            continue
          }
          break
        }
        if (d === '\\') {
          const e = src[j + 1]
          text += e === 'n' ? '\n' : e === 't' ? '\t' : e
          j += 2
          continue
        }
        text += d
        j++
      }
      advanceLines(i, j)
      push('name', text, { functional: src[j + 1] === '(', quoted: true })
      i = j + 1
    } else if (c === '.' && (i + 1 >= src.length || /\s|%/.test(src[i + 1]))) {
      push('end', '.')
      i++
    } else if (c === '!' || c === ';' || c === ',' || c === '|') {
      push(c === '!' || c === ';' ? 'name' : 'punct', c, { functional: src[i + 1] === '(' })
      i++
    } else if ('()[]{}'.includes(c)) {
      push('punct', c)
      i++
    } else if (SYMBOL.test(c)) {
      let j = i
      while (j < src.length && SYMBOL.test(src[j])) {
        // A full stop ends the clause even straight after a symbol (`X = Y.`): stop before a '.' + space.
        if (src[j] === '.' && (j + 1 >= src.length || /\s|%/.test(src[j + 1])) && j > i) break
        j++
      }
      push('name', src.slice(i, j), { functional: src[j] === '(' })
      i = j
    } else throw new PrologSyntaxError(`unexpected character '${c}'`, start.line, start.column)
  }
  out.push({ kind: 'eof', text: '', ...pos(src.length) })
  return out
}

/** Variables of one clause by name; `_` is fresh at each occurrence. */
class Scope {
  /** The name of each variable, indexed by id. */
  readonly names: string[] = []
  /** The id of each named variable (not `_`). */
  private readonly ids = new Map<string, number>()
  get(name: string): Term {
    if (name === '_') {
      this.names.push('_')
      return variable('_', this.names.length - 1)
    }
    let id = this.ids.get(name)
    if (id === undefined) {
      id = this.names.length
      this.names.push(name)
      this.ids.set(name, id)
    }
    return variable(name, id)
  }
}

/**
 * Whether a token can start a term: a number, a variable, an opening bracket, or a name that is not an infix-only
 * operator. After a prefix operator, a token that cannot means the operator is an atom.
 *
 * @param t The token after the prefix operator.
 * @returns True when `t` can start the operator's argument.
 */
const startsTerm = (t: Token): boolean =>
  t.kind === 'number' ||
  t.kind === 'var' ||
  (t.kind === 'punct' && (t.text === '(' || t.text === '[' || t.text === '{')) ||
  (t.kind === 'name' && !(t.text in INFIX && !(t.text in PREFIX) && !t.functional))

/**
 * An operator-precedence parser over the tokens of one text: `parse(max)` reads a term of priority at most `max`,
 * numbering the variables of the current clause in its `Scope`, which `resetScope` replaces between clauses.
 */
class Parser {
  private i = 0
  private readonly tokens: Token[]
  private scope: Scope
  constructor(tokens: Token[], scope: Scope) {
    this.tokens = tokens
    this.scope = scope
  }

  get peek(): Token {
    return this.tokens[this.i]
  }
  next(): Token {
    return this.tokens[this.i++]
  }
  fail(message: string, t: Token = this.peek): never {
    throw new PrologSyntaxError(message, t.line, t.column)
  }
  expect(kind: TokenKind, text?: string): Token {
    const t = this.peek
    if (t.kind !== kind || (text !== undefined && t.text !== text))
      this.fail(`expected ${text ?? kind}, found ${t.kind === 'eof' ? 'end of text' : `'${t.text}'`}`)
    return this.next()
  }
  resetScope(): Scope {
    const s = this.scope
    this.scope = new Scope()
    return s
  }

  /** A term of priority at most `max`; returns the term and its priority. */
  parse(max: number): [Term, number] {
    let [left, leftPriority] = this.primary(max)
    for (;;) {
      const t = this.peek
      const name = t.kind === 'name' && !t.quoted ? t.text : t.kind === 'punct' && t.text === ',' ? ',' : null
      if (name === null) break
      const op = INFIX[name]
      if (!op || op.priority > max) break
      const leftMax = op.type === 'yfx' ? op.priority : op.priority - 1
      if (leftPriority > leftMax) break
      this.next()
      const [right] = this.parse(op.type === 'xfy' ? op.priority : op.priority - 1)
      left = compound(name, [left, right])
      leftPriority = op.priority
    }
    return [left, leftPriority]
  }

  private args(): Term[] {
    this.expect('punct', '(')
    const out = [this.parse(999)[0]]
    while (this.peek.kind === 'punct' && this.peek.text === ',') {
      this.next()
      out.push(this.parse(999)[0])
    }
    this.expect('punct', ')')
    return out
  }

  private primary(max: number): [Term, number] {
    const t = this.next()
    switch (t.kind) {
      case 'number':
        return [numeral(t.value!), 0]
      case 'var':
        return [this.scope.get(t.text), 0]
      case 'punct': {
        if (t.text === '(') {
          const [inner] = this.parse(1200)
          this.expect('punct', ')')
          return [inner, 0]
        }
        if (t.text === '[') {
          if (this.peek.kind === 'punct' && this.peek.text === ']') {
            this.next()
            return [atom('[]'), 0]
          }
          const items = [this.parse(999)[0]]
          while (this.peek.kind === 'punct' && this.peek.text === ',') {
            this.next()
            items.push(this.parse(999)[0])
          }
          let tail: Term = atom('[]')
          if (this.peek.kind === 'punct' && this.peek.text === '|') {
            this.next()
            tail = this.parse(999)[0]
          }
          this.expect('punct', ']')
          return [listTerm(items, tail), 0]
        }
        return this.fail(`unexpected '${t.text}'`, t)
      }
      case 'name': {
        if (t.functional) return [compound(t.text, this.args()), 0]
        const pre = t.quoted ? undefined : PREFIX[t.text]
        if (pre && startsTerm(this.peek)) {
          // `- 1` and `-1` read as the number −1, as in standard Prolog.
          if (t.text === '-' && this.peek.kind === 'number') {
            const n = this.next()
            return [numeral(-n.value!), 0]
          }
          let priority = pre.priority
          let argMax = pre.type === 'fy' ? priority : priority - 1
          if (priority > max) {
            priority = 999
            argMax = 999
          }
          const [arg] = this.parse(argMax)
          return [compound(t.text, [arg]), priority]
        }
        return [atom(t.text), 0]
      }
      case 'end':
        return this.fail('unexpected full stop', t)
      default:
        return this.fail('unexpected end of text', t)
    }
  }
}

/** A clause `head :- body` with its variables numbered 0, 1, … in order of first occurrence. */
export interface Clause {
  /** The head: an atom or compound. */
  readonly head: Term
  /** The body as a list of goals (the top-level conjunction flattened); empty for a fact. */
  readonly body: readonly Term[]
  /** Variable names by id. */
  readonly variableNames: readonly string[]
  /** Where the clause starts in the source (1-based), or 0 for clauses made in code. */
  readonly line: number
}

/** A query `?- goals.` with its variables numbered by first occurrence. */
export interface Query {
  /** The goals (the top-level conjunction flattened), left to right. */
  readonly goals: readonly Term[]
  /** Variable names by id. */
  readonly variableNames: readonly string[]
  /** Where the query starts in the source (1-based); 1 for `parseQuery`. */
  readonly line: number
}

/**
 * The goals of a conjunction `a, b, c` (left to right), however it is bracketed.
 *
 * @param term A goal, usually a conjunction; a term that is not `,` with two arguments is a single goal.
 * @returns The goals that are not themselves conjunctions, in order.
 *
 * @example Flatten a conjunction
 * print(conjuncts(parseTerm('a, (b, c), d')).map((g) => termToString(g)).join(' | '))
 */
export function conjuncts(term: Term): Term[] {
  const out: Term[] = []
  const go = (t: Term) => {
    if (t.kind === 'compound' && t.functor === ',' && t.args.length === 2) {
      go(t.args[0])
      go(t.args[1])
    } else out.push(t)
  }
  go(term)
  return out
}

/**
 * The conjunction $g_1, \dots, g_n$ of goals, nested to the right (`true` when empty).
 *
 * @param goals The goals $g_1, \dots, g_n$, in order.
 * @returns The conjunction, the goal itself when there is one, or the atom `true` when there are none.
 *
 * @example Join goals into one term
 * print(termToString(conjunction([atom('a'), atom('b'), atom('c')])))
 * print(termToString(conjunction([])))
 */
export function conjunction(goals: readonly Term[]): Term {
  if (goals.length === 0) return atom('true')
  return goals.slice(0, -1).reduceRight((acc, g) => compound(',', [g, acc]), goals[goals.length - 1])
}

/**
 * True when a term can be a goal or a clause head: an atom or a compound.
 *
 * @param t The term.
 * @returns Whether `t` is callable.
 */
const callable = (t: Term) => t.kind === 'atom' || t.kind === 'compound'

/**
 * A program's clauses and queries, read from Prolog text. Throws `PrologSyntaxError` with a line and column, also for
 * a directive (`:- Goal.`) and for a clause head that is not callable or is a conjunction or disjunction.
 *
 * @param source The Prolog text: clauses, each ending in a full stop, and queries written `?- Goal.`.
 * @returns `clauses` and `queries`, each in source order, with its own variables numbered from 0.
 *
 * @example Read a program with a query
 * const source = 'parent(tom, bob). grandparent(X, Z) :- parent(X, Y), parent(Y, Z). ?- grandparent(tom, W).'
 * const { clauses, queries } = parseProgram(source)
 * print(clauses.map((c) => clauseToString(c)).join('\n'))
 * print('variables of clause 2:', clauses[1].variableNames.join(', '))
 * print('query:', queries[0].goals.map((g) => termToString(g)).join(', '))
 */
export function parseProgram(source: string): { clauses: Clause[]; queries: Query[] } {
  const tokens = tokenize(source)
  const parser = new Parser(tokens, new Scope())
  const clauses: Clause[] = []
  const queries: Query[] = []
  while (parser.peek.kind !== 'eof') {
    const first = parser.peek
    const [term] = parser.parse(1200)
    parser.expect('end')
    const scope = parser.resetScope()
    if (term.kind === 'compound' && term.functor === '?-' && term.args.length === 1) {
      queries.push({ goals: conjuncts(term.args[0]), variableNames: scope.names, line: first.line })
      continue
    }
    if (term.kind === 'compound' && term.functor === ':-' && term.args.length === 1)
      parser.fail('directives (:- Goal) are not supported', first)
    const isRule = term.kind === 'compound' && term.functor === ':-' && term.args.length === 2
    const head = isRule ? term.args[0] : term
    if (!callable(head)) parser.fail(`a clause head must be an atom or compound, not ${head.kind}`, first)
    if (head.kind === 'compound' && (head.functor === ',' || head.functor === ';'))
      parser.fail('a clause head cannot be a conjunction or disjunction', first)
    const body = isRule ? conjuncts(term.args[1]) : []
    clauses.push({ head, body, variableNames: scope.names, line: first.line })
  }
  return { clauses, queries }
}

/**
 * A query read from text such as `grandparent(X, Y)` or `?- member(X, [a, b]).` (the `?-` and stop optional). Throws
 * `PrologSyntaxError` for empty text, a syntax error, or more than one term.
 *
 * @param source The query text: one term, a conjunction of goals.
 * @returns The query, its goals flattened and its variables numbered by first occurrence.
 *
 * @example The goals and variables of a query
 * const q = parseQuery('?- parent(X, Y), parent(Y, Z).')
 * print('goals:', q.goals.map((g) => termToString(g)).join(' | '))
 * print('variables:', q.variableNames.join(', '))
 */
export function parseQuery(source: string): Query {
  let text = source.trim()
  if (text.startsWith('?-')) text = text.slice(2)
  if (!/\.\s*$/.test(text)) text += ' .'
  const tokens = tokenize(text)
  const parser = new Parser(tokens, new Scope())
  if (parser.peek.kind === 'eof') parser.fail('empty query')
  const [term] = parser.parse(1200)
  parser.expect('end')
  if (parser.peek.kind !== 'eof') parser.fail('a query is one term ending in a full stop')
  const scope = parser.resetScope()
  return { goals: conjuncts(term), variableNames: scope.names, line: 1 }
}

/**
 * One term read from text (no full stop needed; a leading `?-` is dropped); its variables numbered by first occurrence.
 * Throws `PrologSyntaxError` as `parseQuery` does.
 *
 * @param source The text of one term.
 * @returns The term.
 *
 * @example Operators and lists
 * const t = parseTerm('1 + 2 * 3')
 * print(termToString(t), 'has functor', t.functor)
 * print(termToString(parseTerm('[H|T]')), termToString(parseTerm("'hello world'")))
 */
export function parseTerm(source: string): Term {
  return conjunction(parseQuery(source).goals)
}

/**
 * A clause as Prolog text: `head.` for a fact, or `head :- goal, goal.` for a rule.
 *
 * @param clause The head and body goals: a `Clause`, or any object with those two fields.
 * @param options How variables print, as in `termToString`.
 * @returns The text, ending in a full stop.
 *
 * @example A rule and a fact
 * print(clauseToString(parseProgram('p(X):-q(X,Y),\\+r(Y).').clauses[0]))
 * print(clauseToString({ head: atom('sunny'), body: [] }))
 */
export function clauseToString(clause: { head: Term; body: readonly Term[] }, options: PrintOptions = {}): string {
  if (clause.body.length === 0) return `${termToString(clause.head, options)}.`
  return `${termToString(compound(':-', [clause.head, conjunction(clause.body)]), options)}.`
}
