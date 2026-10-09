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
import { useFrames, useOnScreen } from './live'

/** A banana: x₁ ~ N(0, 1.8²), and x₂ given x₁ normal about a parabola: a curved ridge that is hard to walk. */
const banana = (a: number, b: number) => -0.5 * (a / 1.8) ** 2 - 0.5 * ((b - 1.6 + 0.35 * a * a) / 0.55) ** 2
const TARGET: LogDensity = {
  kind: 'log-density',
  name: 'banana',
  dim: 2,
  normalised: false,
  logDensity: (theta) => {
    const [a, b] = toFlat(theta as Tensor)
    return banana(a, b)
  },
  grad: (theta) => {
    const [a, b] = toFlat(theta)
    const r = (b - 1.6 + 0.35 * a * a) / 0.55 ** 2
    return [-a / 1.8 ** 2 - r * 0.7 * a, -r]
  },
}

const gx = grid(-5, 5, 240)
const gy = grid(-5.5, 3, 240)
const density = gy.map((b) => gx.map((a) => Math.exp(banana(a, b))))
/**
 * The log density shifted and clipped to [−4, 4], for a diverging scale: red on the ridge, pale where log π = −4
 * and blue in the tails.
 */
const shade = gy.map((b) => gx.map((a) => Math.max(-4, Math.min(4, banana(a, b) + 4))))
/** Draws kept on the plot: the oldest fall away. */
const KEEP = 600

type Chain = { x: number[]; trajectory: number[][]; accepted: boolean }
type Sampler = Algorithm<ChainStart, Status & { x: Tensor; accepted: boolean; trajectory?: Tensor }>

/**
 * Hamiltonian Monte Carlo against random-walk Metropolis on a banana-shaped density, one step a frame. HMC's leapfrog
 * trajectory is drawn as it goes; the draws build up behind. A click moves the chain there; changing the method or
 * step size starts it again.
 */
export function SamplerRace() {
  const s = useFigureState({
    method: choice(['HMC', 'random walk'], 'HMC', { label: 'sampler' }),
    step: slider(0.05, 1.2, 0.25, { label: 'step size' }),
  })
  const [start, setStart] = useState<[number, number]>([-3.5, -3])
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
  const x = useAxis({ label: 'x₁', range: [-5, 5] })
  const y = useAxis({ label: 'x₂', range: [-5.5, 3] })
  const now = draws.now
  return (
    <div ref={box}>
      <Figure
        title="Sample a banana"
        purpose="Hamiltonian Monte Carlo glides along the ridge; a random walk shuffles. Both target the same density."
        state={s}
        hoverReadout={false}
        defaultSize="L"
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
          <Raster x={gx} y={gy} z={shade} scale="diverging" range={[-4, 4]} fillOpacity={0.7} valueLabel="log π + 4" />
          <Contours x={gx} y={gy} z={density} levels={[0.02, 0.1, 0.3, 0.6, 0.9]} labels={false} />
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
