import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { call, Figure, Histogram, int, Plot, StatusText, useAxis, useComputed, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Work in a worker',
  question: 'How do I move a computation off the page, into a Web Worker?',
  explain:
    'With `mode: "worker"`, `useComputed` takes a task: `call(address, ...args)` names an `aifn-compute` export and plain arguments, so it can cross to the worker. `initial` is the value until the first answer; `then` maps the result on the page.',
}

export default function Worker() {
  const s = useFigureState({ n: int(20000, { ge: 100, le: 1_000_000, label: 'draws' }) })
  const draws = useComputed(
    () => call('foundation/random/normals', call('foundation/random/stream', 'examples-worker'), s.n),
    [s.n],
    { mode: 'worker', initial: null as number[] | null, then: (t) => Array.from(toFlat(t as Tensor)) },
  )
  const x = useAxis({ label: 'z', range: [-4, 4] })
  const y = useAxis({ label: 'density', range: [0, 0.5] })
  return (
    <Figure
      title="Draws made in a worker"
      purpose="Normal draws computed off the main thread."
      state={s}
      readouts={
        <StatusText>{draws.value ? `${s.n} draws in ${draws.ms.toFixed(0)} ms (worker)` : 'computing…'}</StatusText>
      }
    >
      <Plot x={x} y={y}>
        {draws.value && (
          <Histogram
            name="draws"
            values={draws.value}
            bins={60}
            range={[-4, 4]}
            normalize="density"
            stale={draws.stale}
          />
        )}
      </Plot>
    </Figure>
  )
}
