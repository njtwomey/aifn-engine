/**
 * A recipe: one file under `recipes/<section>/<NN-slug>.tsx` that answers one question ("How do I draw a line
 * chart?"). It exports `recipe` (this metadata) and, as its default export, the live example. The gallery, the sidebar
 * and the page are generated from these files; the code shown under the example is the file's own source
 * (`shell/snippet.ts` trims it).
 */
export type Recipe = {
  /** The gallery card's name, e.g. 'Line chart'. */
  title: string
  /** The question the page answers, e.g. 'How do I draw a line chart?'. */
  question: string
  /** One or two sentences under the question; `code` in backticks is set as code. */
  explain: string
}

/** The gallery's sections, in order, like the sections of the Matplotlib gallery. A recipe's folder names its section. */
export const SECTIONS = [
  {
    id: 'lines',
    title: 'Lines, bars and markers',
    blurb: 'Curves, scatter plots, bars, areas and series built by code.',
  },
  { id: 'statistical', title: 'Statistical', blurb: 'Histograms, densities, mass functions, supports and rugs.' },
  {
    id: 'datasets',
    title: 'Datasets',
    blurb:
      'The real datasets of aifn-methods, each drawn the way it is read: ECG, Iris, Old Faithful, Anscombe and more.',
  },
  { id: 'fields', title: 'Images and fields', blurb: 'Heatmaps, contours, vector fields, pixels and filled shapes.' },
  {
    id: 'axes',
    title: 'Axes and scales',
    blurb: 'Fitting, holding and fixing ranges; log, categorical and equal axes.',
  },
  {
    id: 'layout',
    title: 'Figures, subplots and layout',
    blurb: 'The figure frame, grids of plots, dashboards and panels.',
  },
  {
    id: 'text',
    title: 'Text, labels and maths',
    blurb: 'Annotations, TeX, live equations, readouts and status lines.',
  },
  {
    id: 'colour',
    title: 'Colour',
    blurb: 'Palette slots, ink and muted marks, sequential and diverging scales, themes.',
  },
  {
    id: 'controls',
    title: 'Controls',
    blurb: 'Sliders, typed numbers, choices, switches, groups, variants, URL state, players and buttons.',
  },
  {
    id: 'interaction',
    title: 'Interaction',
    blurb: 'Handles, probes, pins, brushes, clicks, live marks and computed values.',
  },
  { id: 'animation', title: 'Animation', blurb: 'Players, step-through traces, live algorithms and frame loops.' },
  {
    id: 'gym',
    title: 'Gym and grids',
    blurb: 'Grids for mazes, gridworlds and search; environment playback and the gym trainer.',
  },
  {
    id: 'interpreter',
    title: 'Interpreter',
    blurb: 'Type a program and see its value as a plot, controls, a table or an image.',
  },
  { id: 'diagrams', title: 'Diagrams and trees', blurb: 'Hand-placed diagrams, graphical models and tree views.' },
] as const

export type SectionId = (typeof SECTIONS)[number]['id']
