import { Curve, Figure, fromSpace, Plot, Points, useAxis, useFigureState, variants } from 'aifn-render'
import { useMemo } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { clamp, defaults } from 'aifn-compute/foundation/space'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { datasetRegistry, generate, type Dataset } from 'aifn-methods/data'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Synthetic datasets',
  question: 'What do the synthetic datasets look like?',
  explain:
    'Every seeded generator of `aifn-methods/data` is registered in `datasetRegistry` with its knobs as a `Space`. `variants` over the registry, with `fromSpace` for each generator’s knobs, gives one picker for all of them: two-feature data are drawn by class (past the palette’s eight colours, they repeat), one feature against the target, a series over its index, and three-dimensional manifolds from the side.',
}

// region
// Two-feature classes and clusters first, then manifolds, one-feature regressions and series; data a scatter of two
// features cannot show (more features, or several features against a target) are left out.
const ORDER = ['classification', 'clustering', 'manifold', 'regression', 'sequence']
const shown = (task: string, columns: number) =>
  columns === 1 ? task === 'regression' || task === 'sequence' : task === 'regression' ? false : columns <= 3
const entries = Object.entries(datasetRegistry)
  .filter(([, e]) => {
    if (e.info.output !== 'dataset' || e.info.module === 'data/real' || !ORDER.includes(e.info.task)) return false
    return shown(e.info.task, (generate(e, stream(1), defaults(e.info.knobs)) as Dataset).x.shape[1])
  })
  .sort(([, a], [, b]) => ORDER.indexOf(a.info.task) - ORDER.indexOf(b.info.task))
const cases = Object.fromEntries(entries.map(([k, e]) => [k, { label: e.info.name, params: fromSpace(e.info.knobs) }]))
// endregion

/** What to draw: the data as points (by class, when labelled) or a curve, with labels for the axes. */
type View = { x: number[]; y: number[]; axes: string[]; group?: number[]; curve?: boolean }

function view(d: Dataset, task: string): View {
  const rows = toRows(d.x)
  const y = d.y ? Array.from(toFlat(d.y)) : null
  const labelled = y !== null && (task === 'classification' || task === 'clustering') && y.every(Number.isInteger)
  // A signal dataset holds its times in x and its values in y; other series hold their values in x.
  if ('signal' in d) return { curve: true, x: rows.map((r) => r[0]), y: y!, axes: ['time', 'x'] }
  if (task === 'sequence')
    return { curve: true, x: rows.map((_, i) => i), y: rows.map((r) => r[0]), axes: ['index', 'x'] }
  if (rows[0].length === 1) return { x: rows.map((r) => r[0]), y: y ?? [], axes: ['x', 'y'] }
  const last = rows[0].length - 1 // the side view of a 3-D manifold: first against last feature
  return {
    x: rows.map((r) => r[0]),
    y: rows.map((r) => r[last]),
    group: labelled ? y! : undefined,
    axes: ['x₁', `x${last === 1 ? '₂' : '₃'}`],
  }
}

export default function Synthetic() {
  // region
  const s = useFigureState({ data: variants(cases, { label: 'dataset', choiceLabel: 'generator', initial: 'moons' }) })
  const { key, values } = s.data
  const entry = datasetRegistry[key]
  // `clamp` drops the knobs that are off (blobs' `prevalence` applies to two centres only).
  // A combination the generator rejects (an annulus whose inner radius passes its outer) is reported, not drawn.
  const { v, error } = useMemo((): { v: View; error?: string } => {
    try {
      return { v: view(generate(entry, stream(1), clamp(entry.info.knobs, values)) as Dataset, entry.info.task) }
    } catch (e) {
      return { v: { x: [], y: [], axes: ['x₁', 'x₂'] }, error: (e as Error).message }
    }
  }, [entry, values])
  // endregion
  const x = useAxis({ label: v.axes[0], key })
  const y = useAxis({ label: v.axes[1], key })
  return (
    <Figure
      title={entry.info.name}
      purpose={entry.info.summary ?? ''}
      state={s}
      defaultSize="L"
      caption={error && <span className="text-destructive">{error}</span>}
    >
      <Plot x={x} y={y}>
        {v.curve ? (
          <Curve name={key} x={v.x} y={v.y} />
        ) : (
          <Points name={key} x={v.x} y={v.y} group={v.group?.map((k) => k % 8)} />
        )}
      </Plot>
    </Figure>
  )
}
