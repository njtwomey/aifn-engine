import {
  Figure,
  pinField,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
  usePinned,
  type PlotPointer,
} from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Hover and pin',
  question: 'How do I preview an item on hover and pin it on click?',
  explain:
    '`usePinned` over a `pinField` gives hover-and-pin: hover previews, a click pins (kept in the URL), a second click or Escape unpins. `focus` is the hovered item, else the pinned one. `onPointer` reports moves and clicks in data coordinates.',
}

const r = rng(13)
const px = Array.from({ length: 40 }, () => r.normal())
const py = Array.from({ length: 40 }, () => r.normal())
const nearest = ([a, b]: [number, number]) => {
  const d = px.map((v, i) => (v - a) ** 2 + (py[i] - b) ** 2)
  return d.indexOf(Math.min(...d))
}

export default function PinOnClick() {
  // region
  const s = useFigureState({ pin: pinField(px.length - 1) })
  const pin = usePinned(s.pin, (v) => s.set('pin', v))
  const onPointer = (e: PlotPointer) =>
    e.type === 'leave'
      ? pin.hover(null)
      : e.type === 'move'
        ? pin.hover(nearest(e.point))
        : pin.toggle(nearest(e.point))
  const f = pin.focus
  // endregion
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', equal: x })
  return (
    <Figure
      title="Hover and pin"
      purpose="Hover a point to preview it; click to pin it."
      state={s}
      hoverReadout={false}
      readouts={
        <Readout
          label={pin.pinned === null ? 'hovered' : 'pinned'}
          value={f === null ? '—' : `#${f} (${px[f].toFixed(2)}, ${py[f].toFixed(2)})`}
        />
      }
    >
      {/* region */}
      <Plot x={x} y={y} onPointer={onPointer}>
        <Points name="points" x={px} y={py} />
        {f !== null && <Points name="focus" x={[px[f]]} y={[py[f]]} emphasis size={12} live />}
      </Plot>
      {/* endregion */}
    </Figure>
  )
}
