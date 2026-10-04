import { ImageOff } from 'lucide-react'
import { useState } from 'react'
import { useTheme } from 'aifn-render'
import { SECTIONS } from '@examples/recipe'
import { entriesOf, ENTRIES, type Entry } from './registry'
import { hrefOf, onLink, thumbnailOf } from './paths'

const REPOSITORY = 'https://github.com/njtwomey/aifn-engine'

/** The engine's packages, as the front page introduces them. */
const PACKAGES = [
  {
    name: 'aifn-compute',
    role: 'Numerics',
    text: 'Tensors, automatic differentiation, linear algebra, probability, optimisation, inference, signals and neural layers, written in TypeScript with no dependencies. It runs in the browser, in a Web Worker and in Node, and its results are tested against NumPy, SciPy, scikit-learn and PyTorch.',
  },
  {
    name: 'aifn-methods',
    role: 'Models and data',
    text: 'Named methods built on the numerics: regression, trees, Gaussian processes, clustering, topic and sequence models, diffusion, reinforcement-learning agents and environments, and the datasets to run them on. Algorithms expose their steps, so a figure can play a fit forwards and backwards.',
  },
  {
    name: 'aifn-render',
    role: 'Figures',
    text: 'React components for interactive figures: plots made of layers on Apache ECharts, SVG diagrams, sliders and other controls, live equations, players for step-by-step traces, and state kept in the URL. Heavy computation moves to a worker so the page stays responsive.',
  },
] as const

const BUILT_ON = ['React', 'Apache ECharts', 'Tailwind CSS', 'shadcn/ui on Base UI', 'KaTeX', 'CodeMirror', 'Lucide']

/** The front page: what the engine is, then a thumbnail per recipe, grouped by section. */
export function Gallery() {
  return (
    <div className="flex flex-col gap-10">
      <header className="space-y-5">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">AIFN Engine</h1>
          <p className="max-w-prose text-base text-muted-foreground">
            Tooling for interactive, browser-based exposition of machine learning and the mathematics under it. A reader
            moves a slider or drags a point, and the model is refitted and redrawn in the page: the numerics, the
            methods and the figures are all here, and nothing is computed on a server.
          </p>
        </div>
        <div className="grid gap-3 md:grid-cols-3">
          {PACKAGES.map((p) => (
            <div key={p.name} className="space-y-1.5 rounded-lg border bg-card p-4">
              <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{p.role}</div>
              <div className="font-mono text-sm font-medium">{p.name}</div>
              <p className="text-sm text-muted-foreground">{p.text}</p>
            </div>
          ))}
        </div>
        <p className="text-sm text-muted-foreground">
          Built on {BUILT_ON.join(', ')}. Source on{' '}
          <a href={REPOSITORY} className="text-foreground underline underline-offset-4 hover:no-underline">
            GitHub
          </a>
          .
        </p>
      </header>
      <div className="space-y-2">
        <h2 className="text-xl font-semibold tracking-tight">Gallery</h2>
        <p className="max-w-prose text-sm text-muted-foreground">
          {ENTRIES.length} small, live recipes for the rendering system. Each page answers one question with the chart,
          the few lines of code that make it, and a sentence or two on why.
        </p>
      </div>
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
      href={hrefOf(entry.path)}
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
