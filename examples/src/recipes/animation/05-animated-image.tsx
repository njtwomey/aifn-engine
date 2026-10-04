import { Figure, Pixels, Player, Plot, Readout, useAxis, usePlayhead } from 'aifn-render'
import { useEffect, useMemo, useRef } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Animated image',
  question: 'How do I animate an image, such as a field evolving over time, efficiently?',
  explain:
    'Precompute one flat `Float32Array` per frame (cheap here, so `useMemo`) and let a `Player` pick the frame; give `Pixels` a fixed `range` so the colours do not refit. A changed layer reaches the chart as a patch, and `Pixels` redraws its own canvas in place, so a frame costs one `putImageData`.',
}

const N = 96
const FRAMES = 240

// Heat diffusion on an insulated N × N plate, from two hot discs: one Float32Array per frame, values in [0, 1]
function diffuse(): Float32Array[] {
  let u = Float32Array.from({ length: N * N }, (_, i) => {
    const [r, c] = [Math.floor(i / N), i % N]
    return Math.hypot(r - 30, c - 30) < 9 || Math.hypot(r - 62, c - 64) < 14 ? 1 : 0
  })
  const frames = [u]
  for (let f = 1; f < FRAMES; f++) {
    for (let step = 0; step < 4; step++) {
      const v = new Float32Array(N * N)
      for (let i = 0; i < N * N; i++) {
        const [r, c] = [Math.floor(i / N), i % N]
        const near =
          (r > 0 ? u[i - N] : u[i]) +
          (r < N - 1 ? u[i + N] : u[i]) +
          (c > 0 ? u[i - 1] : u[i]) +
          (c < N - 1 ? u[i + 1] : u[i])
        v[i] = u[i] + 0.2 * (near - 4 * u[i])
      }
      u = v
    }
    frames.push(u)
  }
  return frames
}

// region
export default function AnimatedImage() {
  const frames = useMemo(diffuse, [])
  const [frame, setFrame] = usePlayhead(FRAMES)
  // The frame's cost: render, commit and the chart's patch (child effects run first); a ref, so measuring adds no render
  const cost = useRef(0)
  const start = performance.now()
  useEffect(() => void (cost.current = performance.now() - start))
  const x = useAxis({ label: 'column' })
  const y = useAxis({ label: 'row', inverse: true, equal: x })
  return (
    <Figure
      title="Heat spreading on a plate"
      purpose="Two hot discs diffuse across an insulated plate."
      controls={<Player value={frame} onChange={setFrame} count={FRAMES} label="frame" />}
      readouts={<Readout label="last frame" value={`${cost.current.toFixed(1)} ms`} />}
      caption="Press play: 60 frames a second. The readout is the previous frame's cost."
    >
      <Plot x={x} y={y}>
        <Pixels width={N} height={N} values={frames[frame]} range={[0, 1]} />
      </Plot>
    </Figure>
  )
}
// endregion
