import { PACKAGES as DOCS, TREE } from '../docs/data'
import { hrefOf, onLink } from './paths'
import { ENTRIES } from './registry'

const REPOSITORY = 'https://github.com/njtwomey/aifn-engine'

/** The engine's packages, as the front page introduces them. */
const PACKAGES = [
  {
    name: 'aifn-compute',
    path: 'compute',
    role: 'Numerics',
    text: 'Tensors, automatic differentiation, linear algebra, probability, optimisation, inference, signals and neural layers, written in TypeScript with no dependencies. It runs in the browser, in a Web Worker and in Node, and its results are tested against NumPy, SciPy, scikit-learn and PyTorch.',
  },
  {
    name: 'aifn-methods',
    path: 'methods',
    role: 'Models and data',
    text: 'Named methods built on the numerics: regression, trees, Gaussian processes, clustering, topic and sequence models, diffusion, reinforcement-learning agents and environments, and the datasets to run them on. Algorithms expose their steps, so a figure can play a fit forwards and backwards.',
  },
  {
    name: 'aifn-render',
    path: 'render',
    role: 'Figures',
    text: 'React components for interactive figures: plots made of layers on Apache ECharts, SVG diagrams, sliders and other controls, live equations, players for step-by-step traces, and state kept in the URL. Heavy computation moves to a worker so the page stays responsive.',
  },
] as const

const BUILT_ON = ['React', 'Apache ECharts', 'Tailwind CSS', 'shadcn/ui on Base UI', 'KaTeX', 'CodeMirror', 'Lucide']

/** What each card leads to. */
const LEADS: Record<(typeof PACKAGES)[number]['path'], string> = {
  compute: `${TREE.compute.length} families: reference and runnable examples`,
  methods: `${TREE.methods.length} areas: reference`,
  render: `${ENTRIES.length} live recipes in the gallery`,
}

/** The front page: what the engine is, and a card into each package's pages. */
export function Home() {
  return (
    <div className="flex flex-col gap-8">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">AIFN Engine</h1>
        <p className="text-base text-muted-foreground">
          Tooling for interactive, browser-based exposition of machine learning and the mathematics under it. A reader
          moves a slider or drags a point, and the model is refitted and redrawn in the page: the numerics, the methods
          and the figures are all here, and nothing is computed on a server.
        </p>
      </header>
      <div className="grid gap-3 md:grid-cols-3">
        {PACKAGES.map((p) => (
          <a
            key={p.name}
            href={hrefOf(p.path)}
            onClick={onLink(p.path)}
            className="flex flex-col gap-1.5 rounded-lg border bg-card p-4 transition-colors hover:border-foreground/30"
          >
            <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{p.role}</div>
            <div className="font-mono text-sm font-medium">{p.name}</div>
            <p className="flex-1 text-sm text-muted-foreground">{p.text}</p>
            <div className="pt-1 text-sm font-medium">
              {DOCS[p.path as 'compute' | 'methods']?.title ?? 'Render'} →{' '}
              <span className="font-normal text-muted-foreground">{LEADS[p.path]}</span>
            </div>
          </a>
        ))}
      </div>
      <p className="text-sm text-muted-foreground">
        Built on {BUILT_ON.join(', ')}. Source on{' '}
        <a href={REPOSITORY} className="text-foreground underline underline-offset-4 hover:no-underline">
          GitHub
        </a>
        .
      </p>
    </div>
  )
}
