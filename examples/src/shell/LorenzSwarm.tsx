import { normal, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { rungeKutta } from 'aifn-compute/dynamics/ode'
import { choice, Curve, Figure, Plot, Points, slider, StatusText, useAxis, useFigureState } from 'aifn-render'
import { useMemo, useRef, useState } from 'react'
import { useFrames, useOnScreen } from './live'

const N = 10
/** One colour per particle, evenly round the hue wheel. */
const RAINBOW = Array.from({ length: N }, (_, i) => `hsl(${(360 * i) / N}, 85%, 58%)`)
const SIGMA = 10
const BETA = 8 / 3
/** Fixed offsets of the swarm from its centre, scaled by the spread. */
const BALL = toRows(normal(stream('home/lorenz'), 0, 1, { shape: [N, 3] }))
/** Points of each particle's trail kept on screen. */
const TAIL = 400
/** Integration steps per frame, of size H. */
const PER_FRAME = 3
const H = 0.008

/** The Lorenz system for all N particles at once, as one vector of length 3N: x, y, z of particle i at 3i, 3i+1, 3i+2. */
const lorenz = (rho: number) => (_t: number, v: Tensor) => {
  const s = toFlat(v)
  const out = new Float64Array(s.length)
  for (let i = 0; i < s.length; i += 3) {
    const [x, y, z] = [s[i], s[i + 1], s[i + 2]]
    out[i] = SIGMA * (y - x)
    out[i + 1] = x * (rho - z) - y
    out[i + 2] = x * y - BETA * z
  }
  return out
}

/**
 * Ten particles started almost together in the Lorenz system, integrated by RK4 a few steps a frame and drawn
 * in the x–z plane with short trails. Above ρ ≈ 24.7 nearby particles separate exponentially (chaos) and spread over
 * the butterfly; below it they settle on a fixed point. A click restarts the swarm at that x and z.
 */
export function LorenzSwarm() {
  const s = useFigureState({
    rho: slider(10, 40, 28, { label: 'ρ' }),
    spread: choice(['0.001', '0.1', '1'], '0.001', { label: 'initial spread' }),
  })
  const [centre, setCentre] = useState<[number, number]>([1, 20])
  const x0 = useMemo(() => {
    const r = Number(s.spread)
    return BALL.flatMap(([u, v, w]) => [centre[0] + r * u, centre[0] + r * v, centre[1] + r * w])
  }, [centre, s.spread])
  const run = useMemo(() => live(rungeKutta(lorenz(s.rho), 'rk4', { stepSize: H }), { x0 }), [s.rho, x0])
  const [frames, setFrames] = useState<{ trails: number[][][]; t: number }>({ trails: [], t: 0 })
  const [seen, setSeen] = useState(run)
  if (seen !== run) {
    setSeen(run)
    setFrames({ trails: [], t: 0 })
  }
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  useFrames(shown, () => {
    let state = null
    for (let k = 0; k < PER_FRAME; k++) {
      const next = run.next()
      if (next.done) return
      state = next.value.state
    }
    if (!state) return
    const v = toFlat(state.x)
    const now = Array.from({ length: N }, (_, i) => [v[3 * i], v[3 * i + 2]])
    setFrames((f) => ({ trails: [...f.trails, now].slice(-TAIL), t: state.time }))
  })
  // Each particle's trail, its own colour.
  const trails = Array.from({ length: N }, (_, i) => ({
    xs: frames.trails.map((f) => f[i][0]),
    zs: frames.trails.map((f) => f[i][1]),
  }))
  const last = frames.trails.at(-1)
  const x = useAxis({ label: 'x', range: [-25, 25] })
  const y = useAxis({ label: 'z', range: [0, 55] })
  return (
    <div ref={box}>
      <Figure
        title="Chaos from a single point"
        purpose="Ten particles started a hair apart in the Lorenz system, integrated by RK4 in the page."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        readouts={<StatusText>{`t = ${frames.t.toFixed(1)}`}</StatusText>}
        caption="Click to restart the swarm there. Lower ρ below about 24.7 and the chaos stops: they settle on a point."
      >
        <Plot x={x} y={y} onPlotClick={([a, b]) => setCentre([a, b])}>
          {trails.map((t, i) => (
            <Curve key={i} x={t.xs} y={t.zs} color={RAINBOW[i]} silent live />
          ))}
          {last && last.map((p, i) => <Points key={i} x={[p[0]]} y={[p[1]]} color={RAINBOW[i]} size={8} live />)}
        </Plot>
      </Figure>
    </div>
  )
}
