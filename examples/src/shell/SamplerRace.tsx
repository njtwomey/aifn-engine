import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { hmc, randomWalkMetropolis, type ChainStart, type LogDensity } from 'aifn-compute/inference/stochastic'
import {
  choice,
  Contours,
  Curve,
  Figure,
  Plot,
  Points,
  Raster,
  slider,
  StatusText,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { useMemo, useRef, useState } from 'react'
import { grid } from '@examples/data'
import { useFrames, useNarrow, useOnScreen } from './live'

/** The doughnut's radius and the spread of the distance from the centre about it. */
const RADIUS = 2.8
const WIDTH = 0.4
/**
 * A doughnut: the distance from the origin normal about `RADIUS`, every angle alike. A walk has to go round a ring
 * whose middle is empty, which a random walk does slowly and HMC's momentum carries it along.
 */
const doughnut = (a: number, b: number) => -0.5 * ((Math.hypot(a, b) - RADIUS) / WIDTH) ** 2
const TARGET: LogDensity = {
  kind: 'log-density',
  name: 'doughnut',
  dim: 2,
  normalised: false,
  logDensity: (theta) => {
    const [a, b] = toFlat(theta as Tensor)
    return doughnut(a, b)
  },
  grad: (theta) => {
    const [a, b] = toFlat(theta)
    const r = Math.max(Math.hypot(a, b), 1e-9)
    const k = -(r - RADIUS) / (WIDTH * WIDTH * r)
    return [k * a, k * b]
  },
}

const BOX = 4.5
const gx = grid(-BOX, BOX, 240)
const density = gx.map((b) => gx.map((a) => Math.exp(doughnut(a, b))))
/**
 * The log density shifted and clipped to [−4, 4], for a diverging scale: red on the ring, pale where log π = −4
 * and blue in the middle and outside.
 */
const shade = gx.map((b) => gx.map((a) => Math.max(-4, Math.min(4, doughnut(a, b) + 4))))
/** Draws kept on the plot: the oldest fall away. */
const KEEP = 600

type Chain = { x: number[]; trajectory: number[][]; accepted: boolean }
type Sampler = Algorithm<ChainStart, Status & { x: Tensor; accepted: boolean; trajectory?: Tensor }>

/**
 * Hamiltonian Monte Carlo against random-walk Metropolis on a doughnut-shaped density, one step a frame. HMC's leapfrog
 * trajectory is drawn as it goes; the draws build up behind. A click moves the chain there; changing the method or
 * step size starts it again.
 */
export function SamplerRace() {
  const s = useFigureState({
    method: choice(['HMC', 'random walk'], 'HMC', { label: 'sampler' }),
    step: slider(0.05, 1.2, 0.25, { label: 'step size' }),
  })
  const [start, setStart] = useState<[number, number]>([-3.6, -3.4])
  const sampler = useMemo(
    () =>
      (s.method === 'HMC'
        ? hmc(TARGET, { stepSize: s.step, steps: 12 })
        : randomWalkMetropolis(TARGET, { scale: s.step })) as unknown as Sampler,
    [s.method, s.step],
  )
  // A fresh chain for every sampler and start: the generator is pulled one step a frame.
  const chain = useMemo(() => live(sampler, { x0: start }), [sampler, start])
  const [draws, setDraws] = useState<{
    xs: number[]
    ys: number[]
    now: Chain | null
    accepted: number
    steps: number
  }>({
    xs: [],
    ys: [],
    now: null,
    accepted: 0,
    steps: 0,
  })
  const [seen, setSeen] = useState(chain)
  if (seen !== chain) {
    setSeen(chain)
    setDraws({ xs: [], ys: [], now: null, accepted: 0, steps: 0 })
  }
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  const narrow = useNarrow()
  useFrames(
    shown,
    () => {
      const next = chain.next()
      if (next.done) return
      const st = next.value.state
      const x = Array.from(toFlat(st.x))
      const trajectory = st.trajectory ? toRows(st.trajectory) : []
      setDraws((d) => ({
        xs: [...d.xs, x[0]].slice(-KEEP),
        ys: [...d.ys, x[1]].slice(-KEEP),
        now: { x, trajectory, accepted: st.accepted },
        accepted: d.accepted + (st.accepted ? 1 : 0),
        steps: d.steps + 1,
      }))
    },
    12,
  )
  const x = useAxis({ label: 'x₁', range: [-BOX, BOX] })
  const y = useAxis({ label: 'x₂', range: [-BOX, BOX], equal: x })
  const now = draws.now
  return (
    <div ref={box}>
      <Figure
        title="Sample a doughnut"
        purpose="Hamiltonian Monte Carlo sweeps round the ring; a random walk shuffles. Both target the same density."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        aspect={0.9}
        controlsCollapsed
        readouts={
          <StatusText>
            {draws.steps > 1
              ? `${draws.steps} steps, ${((100 * draws.accepted) / draws.steps).toFixed(0)}% accepted.`
              : 'Starting…'}
          </StatusText>
        }
        caption="Click to restart the chain there. Too large a step and proposals fail; too small and it crawls."
      >
        <Plot x={x} y={y} onPlotClick={([a, b]) => setStart([a, b])}>
          <Raster
            x={gx}
            y={gx}
            z={shade}
            scale="diverging"
            range={[-4, 4]}
            fillOpacity={0.7}
            valueLabel="log π + 4"
            colorBar={!narrow}
          />
          <Contours x={gx} y={gx} z={density} levels={[0.02, 0.1, 0.3, 0.6, 0.9]} labels={false} />
          <Points name="draws" x={draws.xs} y={draws.ys} slot={2} size={7} live />
          {now && now.trajectory.length > 1 && (
            <Curve
              name="trajectory"
              x={now.trajectory.map((p) => p[0])}
              y={now.trajectory.map((p) => p[1])}
              emphasis={now.accepted}
              muted={!now.accepted}
              showPoints
              silent
              live
            />
          )}
          {now && <Points name="now" x={[now.x[0]]} y={[now.x[1]]} emphasis size={13} live />}
        </Plot>
      </Figure>
    </div>
  )
}
