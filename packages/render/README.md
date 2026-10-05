# aifn-render

React components for interactive figures: plots made of layers on Apache ECharts, SVG diagrams, controls, live
equations, players for step-by-step traces, and state kept in the URL. Built on `aifn-compute` and `aifn-methods`.
Recipes for everything it draws are in the [gallery](https://njtwomey.github.io/aifn-engine/render).

## Installing

The packages are released as tarballs on this repository's GitHub releases (see the root README for the URLs).
`aifn-render` depends on the `aifn-compute` and `aifn-methods` of the same release, and expects the app to provide
`react` and `react-dom` (19 or later): there must be one copy of React on the page.

```tsx
import { Curve, Figure, Plot, Providers, useAxis } from 'aifn-render'

function Line() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="A line">
      <Plot x={x} y={y}>
        <Curve name="line" x={[0, 1, 2]} y={[0, 1, 4]} />
      </Plot>
    </Figure>
  )
}

// Once, around the app (or around each island of figures): the theme and the figures' shared context.
;<Providers>
  <Line />
</Providers>
```

## Styles

The components are written with Tailwind classes against a set of design tokens. Take one of two routes.

**Without Tailwind**: import the compiled stylesheet, which holds the tokens and every class the components use. It
has no CSS reset, so it leaves the rest of the page alone.

```ts
import 'aifn-render/styles.css'
import 'katex/dist/katex.min.css' // for maths
```

**With Tailwind v4**: import the theme and let Tailwind scan the package, so your own classes and the components'
share one build.

```css
@import 'tailwindcss';
@import 'aifn-render/theme.css';
@import 'aifn-render/base.css'; /* optional: page-level defaults (border colour, page background and font) */
@import 'katex/dist/katex.min.css';
@source '../node_modules/aifn-render';
```

Dark mode is the `dark` class on an ancestor; `Providers` puts it on `<html>` and remembers the reader's choice
(`<Providers theme="light">` pins it). To restyle, redefine the variables under `:root` and `.dark` after the import.
Data colours are not CSS: they come from `aifn-render/design`.

## The compute worker

Figures that compute in the background (`useComputed` with `mode: 'worker'`, `useProgram`, `GymTrainer`) start a Web
Worker from this package's `state/compute.worker.js` with `new Worker(new URL(…, import.meta.url))`. Vite, webpack 5
and other bundlers that understand that form bundle the worker for you; nothing needs configuring. An app that cannot
ship a separate worker file (a single-file HTML export) should keep to figures that compute on the page's own thread.

## In this repository

The source is under `src/`, exported as TypeScript through the workspace. `make packages` builds the package
(`scripts/package.ts`), and `make packages-smoke` installs it into an empty project and uses it.
