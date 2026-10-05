/**
 * `CodeEditor`: a code editor (CodeMirror 6), JavaScript for programs run by `aifn-compute/interpreter` or Prolog for
 * `aifn-compute/logic` (`language="prolog"`: highlighting, brackets and error marks only).
 *
 * - Highlighting, indentation and bracket handling from `@codemirror/lang-javascript`; colours from Tailwind classes,
 *   so they follow the lab's light and dark themes.
 * - Completion from the prelude: the namespaces at the top level, each namespace's members after `ns.` (`Math.` is
 *   `math.`), each with its signature, doc and source module; then the program's own names.
 * - Hover docs for namespaces and their members; JSDoc `@param {…}` types complete too.
 * - Error marks: syntax errors from the editor's own parse, and the `errors` the page passes in (a run's error with
 *   its line and column).
 * - Mod-Enter calls `onRun`.
 *
 * The editor owns its document: `value` sets it when it differs (switching examples), `onChange` reports edits.
 */
import { useEffect, useRef } from 'react'
import {
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
  type Completion,
  type CompletionContext,
} from '@codemirror/autocomplete'
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import { javascript, localCompletionSource } from '@codemirror/lang-javascript'
import { bracketMatching, ensureSyntaxTree, indentOnInput, indentUnit, syntaxHighlighting } from '@codemirror/language'
import { lintGutter, setDiagnostics, type Diagnostic } from '@codemirror/lint'
import { EditorState, type Extension } from '@codemirror/state'
import {
  drawSelection,
  EditorView,
  highlightActiveLine,
  highlightActiveLineGutter,
  hoverTooltip,
  keymap,
  lineNumbers,
} from '@codemirror/view'
import { tagHighlighter, tags as t } from '@lezer/highlight'
import {
  members,
  PARAM_TYPES,
  qualified,
  signature,
  type Namespace,
  type Prelude,
  type PreludeEntry,
} from 'aifn-compute/interpreter'
import { cn } from '../../lib/utils'
import { prologLanguage } from './prolog-language'

/** An error to mark: a message and, when known, a 1-based line and column. */
export type CodeError = { message: string; line?: number; column?: number }

export type CodeEditorProps = {
  value: string
  onChange?: (value: string) => void
  /** Default `javascript`. */
  language?: 'javascript' | 'prolog'
  /** The names a JavaScript program can call: completion and hover docs come from it. */
  prelude?: Prelude
  /** Errors to mark (e.g. the last run's). */
  errors?: readonly CodeError[]
  /** Called on Mod-Enter. */
  onRun?: () => void
  /** Accessible name of the editor. */
  label?: string
  className?: string
}

// Token colours as Tailwind classes (generated because they appear here), light then dark.
const highlighter = tagHighlighter([
  {
    tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword],
    class: 'text-violet-700 dark:text-violet-300',
  },
  { tag: [t.number, t.bool, t.null], class: 'text-amber-700 dark:text-amber-300' },
  { tag: [t.string, t.special(t.string)], class: 'text-emerald-700 dark:text-emerald-300' },
  { tag: t.comment, class: 'text-muted-foreground italic' },
  { tag: [t.function(t.variableName), t.function(t.propertyName)], class: 'text-sky-700 dark:text-sky-300' },
  { tag: t.definition(t.variableName), class: 'text-foreground font-medium' },
  { tag: [t.operator, t.punctuation, t.bracket], class: 'text-muted-foreground' },
  // Prolog: variables, functors.
  { tag: t.typeName, class: 'text-sky-700 dark:text-sky-300' },
  { tag: t.propertyName, class: 'text-foreground font-medium' },
])

const theme = EditorView.theme({
  '&': { fontSize: '13px', backgroundColor: 'var(--card)', color: 'var(--foreground)' },
  '&.cm-focused': { outline: 'none' },
  '.cm-scroller': { fontFamily: 'var(--font-mono)', lineHeight: '1.55' },
  '.cm-content': { caretColor: 'var(--foreground)', padding: '8px 0' },
  '.cm-gutters': { backgroundColor: 'var(--card)', color: 'var(--muted-foreground)', border: 'none' },
  '.cm-activeLine': { backgroundColor: 'color-mix(in oklab, var(--muted) 60%, transparent)' },
  '.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--foreground)' },
  '.cm-cursor': { borderLeftColor: 'var(--foreground)' },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
    backgroundColor: 'color-mix(in oklab, var(--ring) 35%, transparent) !important',
  },
  '.cm-matchingBracket': { backgroundColor: 'color-mix(in oklab, var(--ring) 30%, transparent)', outline: 'none' },
  '.cm-tooltip': {
    backgroundColor: 'var(--popover)',
    color: 'var(--popover-foreground)',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    boxShadow: '0 4px 12px rgb(0 0 0 / 0.12)',
  },
  '.cm-tooltip-autocomplete > ul': { fontFamily: 'var(--font-mono)', fontSize: '12px' },
  '.cm-tooltip-autocomplete > ul > li[aria-selected]': {
    backgroundColor: 'var(--accent)',
    color: 'var(--accent-foreground)',
  },
  '.cm-completionDetail': { color: 'var(--muted-foreground)', fontStyle: 'normal', marginLeft: '0.75em' },
  '.cm-completionInfo': { maxWidth: '340px', fontFamily: 'var(--font-sans)', fontSize: '12px', padding: '6px 8px' },
  '.cm-diagnostic': { fontFamily: 'var(--font-sans)', fontSize: '12px' },
  '.cm-lintRange-error': {
    backgroundImage: 'none',
    textDecoration: 'underline wavy var(--destructive)',
    textUnderlineOffset: '3px',
  },
  '.cm-gutter-lint': { width: '1em' },
})

/** A small element with text and inline styles. */
function el(text: string, style: Partial<CSSStyleDeclaration> = {}): HTMLElement {
  const d = document.createElement('div')
  d.textContent = text
  Object.assign(d.style, style)
  return d
}

const MONO = { fontFamily: 'var(--font-mono)', fontSize: '12px' }
const MUTED = { marginTop: '4px', color: 'var(--muted-foreground)' }

/** A doc card for an entry: signature, doc, return value and source module. */
function docNode(e: PreludeEntry): HTMLElement {
  const box = el('', { maxWidth: '360px' })
  box.append(el(signature(e), MONO), el(e.doc, { marginTop: '4px' }))
  if (e.returns) box.append(el(`returns ${e.returns}`, MUTED))
  box.append(el(`from ${e.source}`, { ...MUTED, ...MONO, fontSize: '11px' }))
  return box
}

/** A doc card for a namespace: its doc, source and member count. */
function namespaceNode(n: Namespace, count: number): HTMLElement {
  const box = el('', { maxWidth: '360px' })
  box.append(el(`${n.name} · ${count} functions`, MONO), el(n.doc, { marginTop: '4px' }))
  box.append(el(`from ${n.source}`, { ...MUTED, ...MONO, fontSize: '11px' }))
  return box
}

/** The namespace `Math` stands for. */
const ALIASES: Record<string, string> = { Math: 'math' }

/**
 * Completions from the prelude: after `ns.` the namespace's members (with signature, doc and source); elsewhere the
 * namespaces themselves and the top-level names. Not after any other object's dot (`x.map` is left to the program).
 */
function preludeCompletions(prelude: Prelude) {
  const option = (e: PreludeEntry): Completion => ({
    label: e.name,
    type: e.namespace === null ? 'keyword' : 'function',
    detail: signature(e).slice(qualified(e).length),
    info: () => docNode(e),
  })
  const byNamespace = new Map<string, Completion[]>(
    prelude.namespaces.map((n) => [n.name, members(prelude, n.name).map(option)]),
  )
  const top: Completion[] = [
    ...prelude.namespaces.map((n) => ({
      label: n.name,
      type: 'namespace',
      detail: n.doc,
      info: () => namespaceNode(n, byNamespace.get(n.name)!.length),
      boost: 2,
    })),
    ...members(prelude, null).map((e) => ({ ...option(e), boost: 1 })),
  ]
  const types: Completion[] = PARAM_TYPES.map((t) => ({ label: t, type: 'type', detail: 'parameter type' }))
  return (ctx: CompletionContext) => {
    // A JSDoc `@param {…}` type: int, real, real:log, bool, choice.
    const type = ctx.matchBefore(/@param\s*\{[\w:]*/)
    if (type) return { from: type.from + type.text.indexOf('{') + 1, options: types, validFor: /^[\w:]*$/ }
    const member = ctx.matchBefore(/[\w$]+\.[\w$]*/)
    if (member) {
      const dot = member.text.indexOf('.')
      const ns = member.text.slice(0, dot)
      const options = byNamespace.get(ALIASES[ns] ?? ns)
      if (!options) return null
      // A word before the namespace's name (`foo.math.`) is another object's property.
      if (member.from > 0 && ctx.state.sliceDoc(member.from - 1, member.from) === '.') return null
      return { from: member.from + dot + 1, options, validFor: /^[\w$]*$/ }
    }
    const word = ctx.matchBefore(/[\w$]*/)
    if (!word || (word.from === word.to && !ctx.explicit)) return null
    if (word.from > 0 && ctx.state.sliceDoc(word.from - 1, word.from) === '.') return null
    return { from: word.from, options: top, validFor: /^[\w$]*$/ }
  }
}

/** Hover docs: a namespace member (`random.normal`, `Math.sin`), a namespace, or a top-level name. */
function preludeHover(prelude: Prelude) {
  const entries = new Map(prelude.entries.map((e) => [qualified(e), e]))
  const spaces = new Map(prelude.namespaces.map((n) => [n.name, n]))
  return hoverTooltip((view, pos) => {
    const line = view.state.doc.lineAt(pos)
    const text = line.text
    // The word under the pointer, and the name before it when it follows a dot (`random.normal`).
    let a = pos - line.from
    let b = a
    while (a > 0 && /[\w$]/.test(text[a - 1])) a--
    while (b < text.length && /[\w$]/.test(text[b])) b++
    if (a === b) return null
    const word = text.slice(a, b)
    const owner = text[a - 1] === '.' ? /([\w$]+)\.$/.exec(text.slice(0, a)) : null
    const shadowed = owner !== null && owner.index > 0 && text[owner.index - 1] === '.'
    const tip = (node: HTMLElement) => ({
      pos: line.from + a,
      end: line.from + b,
      above: true,
      create: () => ({ dom: node }),
    })
    if (owner) {
      const e = shadowed ? undefined : entries.get(`${ALIASES[owner[1]] ?? owner[1]}.${word}`)
      return e ? tip(docNode(e)) : null
    }
    const space = spaces.get(ALIASES[word] ?? word)
    if (space) return tip(namespaceNode(space, members(prelude, space.name).length))
    const e = entries.get(word)
    return e && e.namespace === null ? tip(docNode(e)) : null
  })
}

/** Syntax errors from the editor's own parse, plus the given errors placed at their line and column. */
function diagnosticsOf(state: EditorState, errors: readonly CodeError[]): Diagnostic[] {
  const out: Diagnostic[] = []
  const tree = ensureSyntaxTree(state, state.doc.length, 50)
  let syntax = 0
  tree?.iterate({
    enter: (node) => {
      if (!node.type.isError || syntax > 2) return
      syntax++
      const from = Math.min(node.from, state.doc.length)
      out.push({
        from,
        to: Math.min(Math.max(node.to, from + 1), state.doc.length),
        severity: 'error',
        message: 'Syntax error',
      })
    },
  })
  for (const e of errors) {
    if (e.line === undefined || e.line < 1 || e.line > state.doc.lines) {
      if (syntax === 0) out.push({ from: 0, to: Math.min(1, state.doc.length), severity: 'error', message: e.message })
      continue
    }
    const line = state.doc.line(e.line)
    const from = Math.min(line.from + Math.max(0, (e.column ?? 1) - 1), line.to)
    // Underline the word at the column (or the rest of the line).
    const rest = line.text.slice(from - line.from)
    const word = /^[\w$.]+/.exec(rest)?.[0].length ?? rest.length
    out.push({ from, to: Math.max(from + 1, Math.min(line.to, from + word)), severity: 'error', message: e.message })
  }
  return out
}

/** The editor (see the module comment). */
export function CodeEditor({
  value,
  onChange,
  prelude,
  language = 'javascript',
  errors = [],
  onRun,
  label = 'Code',
  className,
}: CodeEditorProps) {
  const host = useRef<HTMLDivElement>(null)
  const view = useRef<EditorView | null>(null)
  const live = useRef({ onChange, onRun, errors })
  live.current = { onChange, onRun, errors }

  useEffect(() => {
    if (!host.current) return
    const extensions: Extension[] = [
      lineNumbers(),
      highlightActiveLineGutter(),
      highlightActiveLine(),
      drawSelection(),
      history(),
      indentUnit.of('  '),
      indentOnInput(),
      bracketMatching(),
      closeBrackets(),
      ...(language === 'prolog'
        ? [prologLanguage]
        : [
            javascript(),
            autocompletion({
              override: [...(prelude ? [preludeCompletions(prelude)] : []), localCompletionSource],
              icons: false,
            }),
            ...(prelude ? [preludeHover(prelude)] : []),
          ]),
      syntaxHighlighting(highlighter),
      lintGutter(),
      keymap.of([
        { key: 'Mod-Enter', run: () => (live.current.onRun?.(), true) },
        ...closeBracketsKeymap,
        ...completionKeymap,
        ...historyKeymap,
        ...defaultKeymap,
        indentWithTab,
      ]),
      EditorView.updateListener.of((u) => {
        if (!u.docChanged) return
        live.current.onChange?.(u.state.doc.toString())
        u.view.dispatch(setDiagnostics(u.state, diagnosticsOf(u.state, live.current.errors)))
      }),
      EditorView.contentAttributes.of({ 'aria-label': label }),
      theme,
    ]
    const v = new EditorView({ state: EditorState.create({ doc: value, extensions }), parent: host.current })
    view.current = v
    v.dispatch(setDiagnostics(v.state, diagnosticsOf(v.state, live.current.errors)))
    return () => {
      v.destroy()
      view.current = null
    }
    // The editor is built once per prelude; `value` and `errors` are pushed in by the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prelude, label, language])

  useEffect(() => {
    const v = view.current
    if (!v || v.state.doc.toString() === value) return
    v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: value } })
  }, [value])

  useEffect(() => {
    const v = view.current
    if (v) v.dispatch(setDiagnostics(v.state, diagnosticsOf(v.state, errors)))
  }, [errors])

  return (
    <div
      ref={host}
      className={cn('overflow-hidden rounded-md border bg-card [&_.cm-editor]:max-h-[28rem]', className)}
      data-code-editor
    />
  )
}
