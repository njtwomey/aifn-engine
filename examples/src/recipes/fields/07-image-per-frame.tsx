import { Figure, Pixels, Player, Plot, Readout, useAxis, useComputed, usePlayhead } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Image computed per frame',
  question: 'How do I animate an image whose frames are too many or too big to precompute?',
  explain:
    'Compute each frame from its index with `useComputed`: it runs at most once per animation frame on the latest index and, when a frame is slow, returns the last one with `stale: true` rather than blocking the player. Each frame is a fresh flat typed array on a fixed `range`.',
}

const N = 200
const FRAMES = 300

// Two point sources of circular waves: the field at time t, as one Float32Array in [−2, 2]
function waves(t: number): Float32Array {
  const out = new Float32Array(N * N)
  for (let r = 0; r < N; r++)
    for (let c = 0; c < N; c++)
      out[r * N + c] =
        Math.sin(0.35 * Math.hypot(r - 100, c - 70) - t) + Math.sin(0.35 * Math.hypot(r - 100, c - 130) - t)
  return out
}

export default function ImagePerFrame() {
  const [frame, setFrame] = usePlayhead(FRAMES)
  const field = useComputed(() => waves(frame / 6), [frame])
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', inverse: true, equal: x })
  return (
    <Figure
      title="Interference of two sources"
      purpose="Each frame is computed from its index when it is shown."
      controls={<Player value={frame} onChange={setFrame} count={FRAMES} label="frame" />}
      readouts={<Readout label="compute" value={`${field.ms.toFixed(1)} ms per frame`} />}
    >
      <Plot x={x} y={y}>
        <Pixels width={N} height={N} values={field.value} range={[-2, 2]} scale="diverging" stale={field.stale} />
      </Plot>
    </Figure>
  )
}
