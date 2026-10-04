import { Fragment, type ReactNode } from 'react'
import { CodeBlock, Tex } from 'aifn-render'

/** Inline marks of a doc comment: `code`, $maths$ (TeX, set by KaTeX) and **strong**. */
function inline(text: string): ReactNode {
  return text.split(/(`[^`]+`|\$[^$\n]+\$|\*\*[^*]+\*\*)/).map((part, i) =>
    part.startsWith('`') ? (
      <code key={i} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground">
        {part.slice(1, -1)}
      </code>
    ) : part.startsWith('$') ? (
      <Tex key={i}>{part.slice(1, -1)}</Tex>
    ) : part.startsWith('**') ? (
      <strong key={i} className="font-medium text-foreground">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  )
}

type Block = { kind: 'p' | 'li' | 'code'; text: string }

/** A doc comment as blocks: paragraphs, list items (with their continuation lines) and fenced code. */
function blocks(text: string): Block[] {
  const out: Block[] = []
  let fence: string[] | null = null
  let open: Block | null = null
  for (const line of text.split('\n')) {
    if (/^\s*```/.test(line)) {
      if (fence) out.push({ kind: 'code', text: fence.join('\n') })
      fence = fence ? null : []
      open = null
    } else if (fence) fence.push(line)
    else if (line.trim() === '') open = null
    else if (/^\s*- /.test(line)) out.push((open = { kind: 'li', text: line.replace(/^\s*- /, '') }))
    else if (open && (open.kind === 'p' || /^\s+/.test(line))) open.text += ` ${line.trim()}`
    else out.push((open = { kind: 'p', text: line.trim() }))
  }
  return out
}

/** One line of a doc comment with its inline marks set. */
export function Inline({ text }: { text: string }) {
  return <>{inline(text)}</>
}

/** The prose of a doc comment: the few Markdown forms the source uses (paragraphs, lists, code, maths, strong). */
export function Markdown({ text, className }: { text: string; className?: string }) {
  const all = blocks(text)
  const out: ReactNode[] = []
  for (let i = 0; i < all.length; i++) {
    const b = all[i]
    if (b.kind === 'code') out.push(<CodeBlock key={i} code={b.text} />)
    else if (b.kind === 'p') out.push(<p key={i}>{inline(b.text)}</p>)
    else {
      const items: Block[] = []
      while (all[i]?.kind === 'li') items.push(all[i++])
      i--
      out.push(
        <ul key={i} className="list-disc space-y-1.5 pl-5">
          {items.map((it, j) => (
            <li key={j}>{inline(it.text)}</li>
          ))}
        </ul>,
      )
    }
  }
  return <div className={className}>{out}</div>
}
