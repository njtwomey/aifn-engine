import { ChevronLeft, ChevronRight } from 'lucide-react'
import { CodeBlock, FigurePage } from 'aifn-render'
import { SECTIONS } from '@examples/recipe'
import { ENTRIES, type Entry } from './registry'
import { hrefOf, onLink } from './paths'
import { Prose } from './Prose'

/** One recipe: the question, a sentence or two, the live example and the example's own (trimmed) source. */
export function RecipePage({ entry }: { entry: Entry }) {
  const section = SECTIONS.find((s) => s.id === entry.section)!
  const at = ENTRIES.indexOf(entry)
  const prev = ENTRIES[at - 1]
  const next = ENTRIES[at + 1]
  const { Example } = entry
  return (
    <FigurePage scope={entry.path} key={entry.path}>
      <header className="space-y-1.5">
        <div className="flex items-center justify-between gap-4">
          <a
            href={`${hrefOf('render')}#${section.id}`}
            onClick={onLink('render')}
            className="shrink-0 text-xs font-medium tracking-wide text-muted-foreground uppercase hover:text-foreground"
          >
            {section.title}
          </a>
          {/* The neighbours, as at the foot of the page, so stepping through recipes needs no scrolling. */}
          <nav aria-label="Neighbouring recipes" className="flex min-w-0 items-center gap-4 text-xs">
            {prev && <Neighbour entry={prev} dir="prev" />}
            {next && <Neighbour entry={next} dir="next" />}
          </nav>
        </div>
        <h1 className="text-xl font-semibold tracking-tight">{entry.question}</h1>
        <Prose text={entry.explain} className="text-base text-muted-foreground" />
      </header>
      <Example />
      <CodeBlock code={entry.snippet} title={entry.file} />
      <footer className="flex justify-between gap-4 border-t pt-4 text-sm">
        {prev ? <Neighbour entry={prev} dir="prev" /> : <span />}
        {next ? <Neighbour entry={next} dir="next" /> : <span />}
      </footer>
    </FigurePage>
  )
}

function Neighbour({ entry, dir }: { entry: Entry; dir: 'prev' | 'next' }) {
  return (
    <a
      href={hrefOf(entry.path)}
      onClick={onLink(entry.path)}
      title={dir === 'prev' ? `Previous: ${entry.title}` : `Next: ${entry.title}`}
      className="flex min-w-0 items-center gap-1 text-muted-foreground hover:text-foreground"
    >
      {dir === 'prev' && <ChevronLeft className="size-4 shrink-0" />}
      <span className="truncate">{entry.title}</span>
      {dir === 'next' && <ChevronRight className="size-4 shrink-0" />}
    </a>
  )
}
