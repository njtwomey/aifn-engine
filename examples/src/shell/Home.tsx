import { ArrowRight, BrainCircuit, LayoutDashboard, Sigma } from 'lucide-react'
import type { ReactNode } from 'react'
import { Button, useTheme } from 'aifn-render'
import { SECTIONS } from '@examples/recipe'
import { nodePath, TREE, type DocTreeNode } from '../docs/data'
import { Carousel, type Slide } from './Carousel'
import { FourierEpicycles } from './FourierEpicycles'
import { HeroFigure } from './HeroFigure'
import { MIXTURE_PLAY_MS, MixtureSteps } from './MixtureSteps'
import { NeighbourRegions } from './NeighbourRegions'
import { NetworkTraining } from './NetworkTraining'
import { OptimiserRace } from './OptimiserRace'
import { hrefOf, onLink, thumbnailOf } from './paths'
import { ENTRIES } from './registry'

const REPOSITORY = 'https://github.com/njtwomey/aifn-engine'

type Chip = { label: string; path: string }

/** The engine's packages, as the landing page introduces them: what each is, how it is imported, and what it holds. */
const PACKAGES: {
  name: string
  path: string
  role: string
  Icon: typeof Sigma
  text: string
  /** The import line shown on the card (built in JSX: the import check reads quoted module paths in strings). */
  usage: { names: string; module: string }
  chips: Chip[]
  cta: string
}[] = [
  {
    name: 'aifn-compute',
    path: 'compute',
    role: 'Numerics',
    Icon: Sigma,
    text: 'Tensors, autodiff, linear algebra, probability, optimisation and neural layers. No dependencies; tested against NumPy, SciPy and PyTorch.',
    usage: { names: 'cholesky', module: 'aifn-compute/numerics/linalg' },
    chips: TREE.compute.map((n: DocTreeNode) => ({ label: n.name, path: nodePath(n) })),
    cta: 'Compute reference',
  },
  {
    name: 'aifn-methods',
    path: 'methods',
    role: 'Models and data',
    Icon: BrainCircuit,
    text: 'Regression, Gaussian processes, clustering, diffusion, RL agents and the datasets to run them on, with algorithms you can step through.',
    usage: { names: 'gpPosterior', module: 'aifn-methods/learning/gaussian-processes' },
    chips: TREE.methods.map((n: DocTreeNode) => ({ label: n.name, path: nodePath(n) })),
    cta: 'Methods reference',
  },
  {
    name: 'aifn-render',
    path: 'render',
    role: 'Figures',
    Icon: LayoutDashboard,
    text: 'React figures: plots, diagrams, controls, live equations and players, with heavy work moved to a worker.',
    usage: { names: 'Figure, Plot, Curve', module: 'aifn-render' },
    chips: SECTIONS.map((s) => ({ label: s.title, path: `render#${s.id}` })),
    cta: 'Render Gallery',
  },
]

/** Gallery recipes shown as the landing page's mosaic: one strong picture from many sections. */
const SHOWCASE = [
  'layout/subplots-grid',
  'datasets/ecg',
  'diagrams/factor-graph',
  'fields/contours',
  'animation/step-through-trace',
  'statistical/density',
  'datasets/iris',
  'fields/heatmap',
  'fields/class-regions',
  'diagrams/tree-view',
  'fields/pixels',
  'datasets/fonts',
]

const BUILT_ON = ['React', 'Apache ECharts', 'Tailwind CSS', 'shadcn/ui on Base UI', 'KaTeX', 'CodeMirror', 'Lucide']

const leaves = (nodes: readonly DocTreeNode[]): DocTreeNode[] =>
  nodes.flatMap((n) => (n.children.length ? leaves(n.children) : [n]))
const modules = [...leaves(TREE.compute), ...leaves(TREE.methods)]
const functions = modules.reduce((a, n) => a + n.files.reduce((b, f) => b + f.key.length + f.supporting.length, 0), 0)
const examples = modules.reduce((a, n) => a + n.examples, 0)

const STATS = [
  { value: ENTRIES.length, label: 'live recipes' },
  { value: functions, label: 'documented functions' },
  { value: examples, label: 'runnable examples' },
  { value: 0, label: 'servers involved' },
]

/** An in-app link (`path#hash`): followed in place on a plain click. */
const go = (path: string) => {
  const [p, hash = ''] = path.split('#')
  return { href: hash ? `${hrefOf(p)}#${hash}` : hrefOf(p), onClick: onLink(p, hash) }
}

/** The landing page (under the shell's top bar): a live figure, the engine in numbers, more live figures, the packages and the gallery. */
export function Home() {
  return (
    <>
      <div className="mx-auto flex w-full max-w-[1200px] flex-col gap-20 px-4 pt-10 pb-12 md:px-8">
        <Hero />
        <Stats />
        <Showcase />
        <Packages />
        <Mosaic />
        <footer className="border-t pt-6 text-sm text-muted-foreground">
          Built on {BUILT_ON.join(', ')}. Source on{' '}
          <a href={REPOSITORY} className="text-foreground underline underline-offset-4 hover:no-underline">
            GitHub
          </a>
          .
        </footer>
      </div>
    </>
  )
}

function Hero() {
  return (
    <section className="relative grid items-center gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
      <div
        aria-hidden
        className="pointer-events-none absolute -inset-x-16 -top-24 -z-10 h-[520px] bg-[radial-gradient(ellipse_at_top_left,color-mix(in_oklab,#2a78d6_22%,transparent),transparent_60%),radial-gradient(ellipse_at_80%_20%,color-mix(in_oklab,#eb6834_16%,transparent),transparent_55%)]"
      />
      <div className="space-y-6">
        <div className="inline-flex items-center gap-2 rounded-full border bg-background/60 px-3 py-1 text-xs text-muted-foreground backdrop-blur">
          <span className="size-1.5 animate-pulse rounded-full bg-success" />
          Everything on this site runs in your browser
        </div>
        <h1 className="text-4xl leading-[1.05] font-semibold tracking-tight md:text-6xl">
          Machine learning you can{' '}
          <span className="bg-gradient-to-r from-[#2a78d6] via-[#1baf7a] to-[#eb6834] bg-clip-text text-transparent">
            grab and move
          </span>
          .
        </h1>
        <p className="max-w-prose text-base text-muted-foreground md:text-lg">
          AIFN Engine is the numerics, the models and the figures behind interactive explanations of machine learning.
          Drag a point and the model refits; nothing is computed on a server.
        </p>
        <div className="flex flex-wrap gap-2">
          <Button size="lg" render={<a {...go('render')} />}>
            Browse the gallery <ArrowRight />
          </Button>
          <Button size="lg" variant="outline" render={<a {...go('compute')} />}>
            Read the docs
          </Button>
        </div>
      </div>
      <Framed>
        <HeroFigure />
      </Framed>
    </section>
  )
}

function Framed({ children }: { children: ReactNode }) {
  return <div className="rounded-xl border bg-card p-2 shadow-lg shadow-black/5 dark:shadow-black/40">{children}</div>
}

function Stats() {
  return (
    <section className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border md:grid-cols-4">
      {STATS.map((s) => (
        <div key={s.label} className="bg-card px-5 py-5">
          <div className="font-mono text-3xl font-semibold tracking-tight tabular-nums md:text-4xl">
            {s.value.toLocaleString()}
          </div>
          <div className="text-sm text-muted-foreground">{s.label}</div>
        </div>
      ))}
    </section>
  )
}

function SectionHead({ kicker, title, children }: { kicker: string; title: string; children: ReactNode }) {
  return (
    <div className="max-w-2xl space-y-2">
      <div className="text-xs font-medium tracking-widest text-muted-foreground uppercase">{kicker}</div>
      <h2 className="text-3xl font-semibold tracking-tight">{title}</h2>
      <p className="text-muted-foreground">{children}</p>
    </div>
  )
}

/** The carousel's slides: figures made for this page, each moving on its own while it is shown. */
const SLIDES: Slide[] = [
  {
    label: 'Optimisers',
    blurb: 'Seven first-order methods and L-BFGS raced over a landscape with four minima, from a start that wanders.',
    Figure: OptimiserRace,
  },
  {
    label: 'Neural network',
    blurb: 'A small network trained live on two spirals; click to add points and watch it refit.',
    Figure: NetworkTraining,
    duration: 12000,
  },
  {
    label: 'Classifier',
    blurb: 'A k-nearest-neighbour classifier whose regions are redrawn as the classes drift. Drag them yourself.',
    Figure: NeighbourRegions,
  },
  {
    label: 'Mixture',
    blurb: 'A Gaussian mixture fitted by EM from one corner to convergence, again and again. Scrub or step it.',
    Figure: MixtureSteps,
    duration: MIXTURE_PLAY_MS,
  },
  {
    label: 'Fourier',
    blurb: 'A Fourier series drawn by a chain of rotating circles. Pick a shape and how many circles.',
    Figure: FourierEpicycles,
    duration: 10000,
  },
]

function Showcase() {
  return (
    <section className="space-y-6">
      <SectionHead kicker="Live, in the page" title="Every figure is a running model">
        Not videos and not screenshots: each of these is recomputed on every move of the pointer, by the same code the
        reference pages document.
      </SectionHead>
      <Carousel slides={SLIDES} />
    </section>
  )
}

function Packages() {
  return (
    <section className="space-y-6">
      <SectionHead kicker="The stack" title="Three packages, one engine">
        Numerics at the bottom, named methods on top of them, and the figures that draw both. Each is documented from
        its own source, with examples you can edit and run.
      </SectionHead>
      <div className="grid gap-4 md:grid-cols-3">
        {PACKAGES.map(({ Icon, ...p }) => (
          <div key={p.name} className="flex flex-col gap-4 rounded-xl border bg-card p-6">
            <div className="flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-lg bg-gradient-to-br from-[#2a78d6]/20 to-[#1baf7a]/20">
                <Icon className="size-5" />
              </div>
              <div>
                <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{p.role}</div>
                <div className="font-mono text-sm font-semibold">{p.name}</div>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">{p.text}</p>
            <code className="block rounded-md bg-muted px-3 py-2 font-mono text-xs break-words">
              import {'{'} {p.usage.names} {'}'} from &apos;{p.usage.module}&apos;
            </code>
            <div className="flex flex-1 flex-wrap content-start gap-1.5">
              {p.chips.map((c) => (
                <a
                  key={c.path}
                  {...go(c.path)}
                  className="rounded-full border px-2.5 py-0.5 text-xs text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground"
                >
                  {c.label}
                </a>
              ))}
            </div>
            <a {...go(p.path)} className="group flex items-center gap-1 pt-1 text-sm font-medium">
              {p.cta}
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
            </a>
          </div>
        ))}
      </div>
    </section>
  )
}

function Mosaic() {
  const { resolved } = useTheme()
  const showcase = SHOWCASE.map((p) => ENTRIES.find((e) => e.path === `render/${p}`)).filter((e) => e !== undefined)
  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <SectionHead kicker="Render Gallery" title={`${ENTRIES.length} recipes, one question each`}>
          Every picture is a live figure and the few lines that make it.
        </SectionHead>
        <Button variant="outline" render={<a {...go('render')} />}>
          See them all <ArrowRight />
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {showcase.map((e) => (
          <a
            key={e.path}
            {...go(e.path)}
            className="group relative overflow-hidden rounded-lg border bg-background transition hover:-translate-y-0.5 hover:border-foreground/30 hover:shadow-md"
          >
            <img
              src={thumbnailOf(e, resolved)}
              alt=""
              loading="lazy"
              className="aspect-[4/3] w-full object-contain p-1.5 transition-transform duration-300 group-hover:scale-105"
            />
            <div className="absolute inset-x-0 bottom-0 translate-y-full bg-gradient-to-t from-background via-background/90 to-transparent px-3 pt-6 pb-2 text-sm font-medium transition-transform group-hover:translate-y-0">
              {e.title}
            </div>
          </a>
        ))}
      </div>
    </section>
  )
}
