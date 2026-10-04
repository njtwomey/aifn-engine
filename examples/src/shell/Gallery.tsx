import { ImageOff } from 'lucide-react'
import { useState } from 'react'
import { useTheme } from 'aifn-render'
import { SECTIONS } from '@examples/recipe'
import { entriesOf, ENTRIES, type Entry } from './registry'
import { onLink, thumbnailOf } from './paths'

/** The front page: a thumbnail per recipe, grouped by section. */
export function Gallery() {
  return (
    <div className="flex flex-col gap-10">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">aifn-render examples</h1>
        <p className="max-w-prose text-sm text-muted-foreground">
          A gallery of recipes for the rendering system: {ENTRIES.length} small, live examples. Each page answers one
          question with the chart, the few lines of code that make it, and a sentence or two on why.
        </p>
      </header>
      {SECTIONS.map((s) => {
        const items = entriesOf(s.id)
        if (!items.length) return null
        return (
          <section key={s.id} id={s.id} className="space-y-3">
            <div>
              <h2 className="text-lg font-semibold tracking-tight">{s.title}</h2>
              <p className="text-sm text-muted-foreground">{s.blurb}</p>
            </div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-3">
              {items.map((e) => (
                <Card key={e.path} entry={e} />
              ))}
            </div>
          </section>
        )
      })}
    </div>
  )
}

function Card({ entry }: { entry: Entry }) {
  const { resolved } = useTheme()
  const [missing, setMissing] = useState(false)
  return (
    <a
      href={`/${entry.path}`}
      onClick={onLink(entry.path)}
      className="group overflow-hidden rounded-lg border bg-card transition-colors hover:border-foreground/30"
    >
      <div className="flex aspect-[4/3] items-center justify-center overflow-hidden border-b bg-background">
        {missing ? (
          <ImageOff className="size-6 text-muted-foreground/50" aria-hidden />
        ) : (
          <img
            src={thumbnailOf(entry.path, resolved)}
            alt=""
            loading="lazy"
            onError={() => setMissing(true)}
            className="size-full object-contain p-1.5 transition-transform group-hover:scale-[1.03]"
          />
        )}
      </div>
      <div className="px-3 py-2">
        <div className="text-sm font-medium">{entry.title}</div>
        <div className="line-clamp-2 text-xs text-muted-foreground">{entry.question}</div>
      </div>
    </a>
  )
}
