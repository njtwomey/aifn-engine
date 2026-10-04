# aifn-render examples

A gallery of recipes for `aifn-render`, in the manner of the Matplotlib gallery: each page answers one question ("How
do I draw a line chart?") with the live chart, the few lines that make it, and a sentence or two. It is about
rendering and UX only; data are inline arrays or a few lines of plain maths, and `aifn-compute` appears only where a render
feature takes an `aifn-compute` object (a distribution for `Density`, a `Tree` for `TreeView`, an `Algorithm` for a trace).

| Command               | What it does                                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `make examples`       | Dev server on http://localhost:5192/ (gallery at `/`, recipes at `/<section>/<slug>`)          |
| `make examples-check` | Renders the gallery and every recipe on the server, reports throws, checks the import boundary |

## Adding a recipe

Add one file, `src/recipes/<section>/<NN-slug>.tsx` (sections are listed in `src/recipe.ts`; `NN` orders the files
within a section and is not part of the URL). It exports `recipe` (title, question, one or two sentences) and the live
example as its default export:

```tsx
import { Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Line chart',
  question: 'How do I draw a line chart?',
  explain: 'A `Plot` takes two axis models; each `Curve` is one line.',
}

export default function LineChart() {
  …
}
```

The gallery, the sidebar and the page are generated from the files. The code shown under the example is the file's own
source, imported with `?raw` (`src/shell/snippet.ts`): imports are kept (except `@examples/recipe`), the `recipe` block
is dropped, and when the file has `// region` … `// endregion` blocks (a note may follow the word; in JSX, `{/* region */}`) only those are shown, so boilerplate stays in the
file but out of the snippet. A line ending in `// hide` is dropped. Keep examples to 10–40 lines; `make examples-check`
warns above 45.

Code is highlighted by `CodeBlock` from `aifn-render` (read-only, copyable; Lezer's TSX grammar, no editor).
