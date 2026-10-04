/**
 * `CodeBlock`: read-only source, highlighted, with a copy button. For showing code (a snippet beside its live example),
 * not editing it: use `CodeEditor` for that.
 *
 * Highlighting parses with the Lezer grammar behind `@codemirror/lang-javascript` (TSX by default; TypeScript, JSX and
 * JavaScript are subsets) and `@lezer/highlight`'s `highlightCode`, with token colours as Tailwind classes, so it
 * follows the light and dark themes. No editor is created: it renders on the server and costs one parse per change.
 *
 *   <CodeBlock code={source} />
 */
import { useMemo, useState } from 'react'
import { Check, Copy } from 'lucide-react'
import { tsxLanguage } from '@codemirror/lang-javascript'
import { highlightCode, tagHighlighter, tags as t } from '@lezer/highlight'
import { Button } from '@render/ui/button'
import { cn } from '@render/lib/utils'

export type CodeBlockProps = {
  code: string
  /** A caption over the code, e.g. its file name. */
  title?: string
  /** Line numbers in the gutter (default false). */
  lineNumbers?: boolean
  className?: string
}

// Token colours as Tailwind classes (generated because they appear here), light then dark.
const highlighter = tagHighlighter([
  {
    tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword, t.self],
    class: 'text-violet-700 dark:text-violet-300',
  },
  { tag: [t.number, t.bool, t.null, t.atom], class: 'text-amber-700 dark:text-amber-300' },
  { tag: [t.string, t.special(t.string), t.regexp, t.escape], class: 'text-emerald-700 dark:text-emerald-300' },
  { tag: [t.comment, t.lineComment, t.blockComment], class: 'text-muted-foreground italic' },
  { tag: [t.function(t.variableName), t.function(t.propertyName)], class: 'text-sky-700 dark:text-sky-300' },
  { tag: [t.typeName, t.className, t.namespace], class: 'text-teal-700 dark:text-teal-300' },
  { tag: [t.tagName, t.angleBracket], class: 'text-rose-700 dark:text-rose-300' },
  { tag: t.attributeName, class: 'text-amber-800 dark:text-amber-200' },
  { tag: t.definition(t.variableName), class: 'text-foreground font-medium' },
  { tag: [t.operator, t.punctuation, t.bracket, t.separator], class: 'text-muted-foreground' },
])

type Token = { text: string; cls: string }

function highlight(code: string): Token[][] {
  const lines: Token[][] = [[]]
  highlightCode(
    code,
    tsxLanguage.parser.parse(code),
    highlighter,
    (text, cls) => lines[lines.length - 1].push({ text, cls }),
    () => lines.push([]),
  )
  return lines
}

export function CodeBlock({ code, title, lineNumbers = false, className }: CodeBlockProps) {
  const lines = useMemo(() => highlight(code), [code])
  const [copied, setCopied] = useState(false)
  const copy = () => {
    void navigator.clipboard?.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    })
  }
  return (
    <div className={cn('group relative overflow-hidden rounded-lg border bg-card text-card-foreground', className)}>
      {title && (
        <div className="border-b px-3 py-1.5 font-mono text-xs text-muted-foreground" data-code-title="">
          {title}
        </div>
      )}
      <Button
        variant="ghost"
        size="icon-sm"
        className="absolute top-1 right-1 z-10 bg-card/80 backdrop-blur"
        aria-label={copied ? 'Copied' : 'Copy code'}
        title={copied ? 'Copied' : 'Copy code'}
        onClick={copy}
      >
        {copied ? <Check /> : <Copy />}
      </Button>
      <pre className="overflow-x-auto py-2.5 font-mono text-[12.5px] leading-[1.55]">
        <code className="grid min-w-max">
          {lines.map((line, i) => (
            <span key={i} className="flex px-3">
              {lineNumbers && (
                <span className="mr-4 inline-block w-6 shrink-0 text-right text-muted-foreground/60 select-none">
                  {i + 1}
                </span>
              )}
              <span className="pr-8">
                {line.length === 0
                  ? ' '
                  : line.map((tok, j) =>
                      tok.cls ? (
                        <span key={j} className={tok.cls}>
                          {tok.text}
                        </span>
                      ) : (
                        tok.text
                      ),
                    )}
              </span>
            </span>
          ))}
        </code>
      </pre>
    </div>
  )
}
