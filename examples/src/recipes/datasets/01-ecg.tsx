import { Curve, Figure, Plot, Plots, Points, Slider, useAxis } from 'aifn-render'
import { useState } from 'react'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { ECG_BEAT_CLASSES, ecgBeats, mitBihEcg } from 'aifn-methods/data/real/ecg'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'ECG (MIT-BIH)',
  question: 'What does the MIT-BIH ECG look like, beat by beat?',
  explain:
    '`mitBihEcg()` is five minutes of lead MLII of MIT-BIH record 208 at 360 Hz with the cardiologists’ beat labels; `ecgBeats()` cuts a window around every beat. Slide through the record: the premature ventricular contractions (PVCs) are the wide, early beats. Below, the average beat of each class.',
}

// region
const { signal, beats } = mitBihEcg()
const mv = Array.from(toFlat(signal.data))
const windows = ecgBeats({ before: 0.25, after: 0.4, centre: true })
// endregion
const rows = toFlat(windows.x)
const labels = Array.from(toFlat(windows.y!))
const width = windows.x.shape[1]
const lag = windows.meta.featureNames!.map((name) => parseFloat(name))
const shown = [0, 2, 3] // normal, ventricular ectopic, fusion
const mean = (c: number) =>
  lag.map((_, j) => {
    const of = labels.flatMap((l, r) => (l === c ? [rows[r * width + j]] : []))
    return of.reduce((a, b) => a + b, 0) / of.length
  })

// region
export default function Ecg() {
  const [start, setStart] = useState(60)
  const from = Math.round(start * 360)
  const t = Array.from({ length: 8 * 360 }, (_, i) => (from + i) / 360)
  const seen = beats.filter((b) => b.sample >= from && b.sample < from + t.length)
  const time = useAxis({ label: 'time (s)' })
  const amp = useAxis({ label: 'mV' })
  const ms = useAxis({ label: 'time from the beat (ms)' })
  const avg = useAxis({ label: 'mV' })
  return (
    <Figure
      title="MIT-BIH record 208"
      purpose="Eight seconds of ECG with the annotated beats, and the average beat of each class."
      defaultSize="L"
      controls={<Slider label="start (s)" value={start} min={0} max={292} step={1} onChange={setStart} />}
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={time} y={amp}>
          <Curve name="lead MLII" x={t} y={mv.slice(from, from + t.length)} emphasis thin />
          <Points
            x={seen.map((b) => b.time)}
            y={seen.map((b) => mv[b.sample])}
            group={seen.map((b) => b.label)}
            groupNames={[...ECG_BEAT_CLASSES]}
          />
        </Plot>
        <Plot x={ms} y={avg}>
          {shown.map((c) => (
            <Curve key={c} name={`${ECG_BEAT_CLASSES[c]} (mean)`} x={lag} y={mean(c)} slot={c} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
// endregion
