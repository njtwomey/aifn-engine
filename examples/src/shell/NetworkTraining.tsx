import { grad } from 'aifn-compute/foundation/autodiff'
import { treeLeaves, type Params } from 'aifn-compute/foundation/pytree'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { add, mean, mul, neg, square, sum, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { softplus } from 'aifn-compute/numerics/special'
import { Mlp } from 'aifn-compute/nn'
import { adamRule, applyUpdates, type RuleState } from 'aifn-compute/optim/first-order'
import {
  Button,
  choice,
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
import { useRef, useState } from 'react'
import { grid } from '@examples/data'
import { useFrames, useOnScreen, useNarrow } from './live'

type Pt = { x: number; y: number; c: 0 | 1 }

/** Fixed uniform and normal draws that the datasets below place and jitter their points with. */
const U = toRows(uniform(stream('home/net/data/u'), -3.8, 3.8, { shape: [240, 2] }))
const Z = toRows(normal(stream('home/net/data/z'), 0, 1, { shape: [260, 2] }))

/**
 * Two interleaved spirals of nearly two turns each, 130 points an arm with Gaussian noise of sd 0.15: the classic hard
 * case, where a straight line gets half of them wrong.
 */
function spirals(): Pt[] {
  const out: Pt[] = []
  for (let i = 0; i < 130; i++) {
    const r = 0.35 + (3.3 * i) / 130
    const t = 3.2 * r
    for (const c of [0, 1] as const) {
      const a = t + c * Math.PI
      const [u, v] = Z[2 * i + c]
      out.push({ x: r * Math.cos(a) + 0.15 * u, y: r * Math.sin(a) + 0.15 * v, c })
    }
  }
  return out
}

/** A 4 × 4 checkerboard of squares of side 2: sixteen regions, so the network has to carve many corners. */
const checkerboard = (): Pt[] =>
  U.map(([x, y]) => ({ x, y, c: ((Math.floor(x / 2) + Math.floor(y / 2)) & 1) as 0 | 1 }))

/** Three noisy rings, the middle one the other class: the boundary is two nested closed curves. */
const rings = (): Pt[] =>
  Array.from({ length: 210 }, (_, i) => {
    const k = i % 3
    const r = [0.9, 2.1, 3.3][k] + 0.15 * Z[i][0]
    const a = (2 * Math.PI * i) / 70 + 0.3 * Z[i][1]
    return { x: r * Math.cos(a), y: r * Math.sin(a), c: (k === 1 ? 1 : 0) as 0 | 1 }
  })

/** Two interleaved half-moons with noise: one bend, easy for a network and impossible for a line. */
const moons = (): Pt[] =>
  Array.from({ length: 200 }, (_, i) => {
    const c = (i % 2) as 0 | 1
    const t = (Math.PI * (i >> 1)) / 99
    const [u, v] = c ? [1 - Math.cos(t), 0.5 - Math.sin(t)] : [Math.cos(t), Math.sin(t)]
    return { x: 2.2 * (u - 0.5) + 0.2 * Z[i][0], y: 2.2 * (v - 0.25) + 0.2 * Z[i][1], c }
  })

const DATASETS = { spirals, checkerboard, rings, moons }
const ORDER = Object.keys(DATASETS) as (keyof typeof DATASETS)[]

/** The network at each width the reader can pick: 2 → w → w → 1, tanh. */
const WIDTHS = ['8', '16', '32'] as const
const NETS = Object.fromEntries(WIDTHS.map((w) => [w, Mlp([2, +w, +w, 1], { activation: 'tanh' })])) as Record<
  (typeof WIDTHS)[number],
  ReturnType<typeof Mlp>
>
const rule = adamRule({ stepSize: 0.01 })
const gx = grid(-4, 4, 56)
const GRID = tensor(gx.flatMap((b) => gx.map((a) => [a, b])))
/** The drawn grid: four times finer than the network is evaluated on, filled by bicubic interpolation. */
const FINE = 4
const fx = grid(-4, 4, (gx.length - 1) * FINE + 1)

/** Catmull–Rom weights for the four samples around a point a fraction `t` of the way from the second to the third. */
const cubic = (t: number): [number, number, number, number] => {
  const t2 = t * t
  const t3 = t2 * t
  return [0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2), 0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2)]
}

/**
 * The network's output on the coarse grid as the colour field 2p − 1 = tanh(logit / 2), which rolls off smoothly
 * where the network is sure rather than clipping, interpolated bicubically onto the fine grid: smooth shading, no
 * kinks along the coarse cells.
 */
function surface(logits: ArrayLike<number>): number[][] {
  const n = gx.length
  const at = (i: number, j: number) => {
    const v = logits[Math.min(n - 1, Math.max(0, i)) * n + Math.min(n - 1, Math.max(0, j))]
    return Math.tanh(v / 2)
  }
  return fx.map((_, I) => {
    const i = Math.min(Math.floor(I / FINE), n - 2)
    const wy = cubic(I / FINE - i)
    return fx.map((_, J) => {
      const j = Math.min(Math.floor(J / FINE), n - 2)
      const wx = cubic(J / FINE - j)
      let v = 0
      for (let a = 0; a < 4; a++) {
        let row = 0
        for (let b = 0; b < 4; b++) row += wx[b] * at(i - 1 + a, j - 1 + b)
        v += wy[a] * row
      }
      return Math.max(-1, Math.min(1, v))
    })
  })
}
/**
 * Adam steps per frame (frames come 30 a second): slow enough that a fit of the spirals unfolds over a dozen seconds or
 * so, the surface finding the arms one fold at a time.
 */
const STEPS = 1
/** Points of the loss curve kept: past this, every other one is dropped, so a long run stays cheap to draw. */
const CURVE = 1000

type Net = ReturnType<typeof Mlp>

/** The logistic loss of the network's logits against labels ±1, mean over the points. */
function lossOf(net: Net, points: Pt[]) {
  const X = tensor(points.map((p) => [p.x, p.y]))
  const Y = tensor(
    points.map((p) => (p.c ? 1 : -1)),
    [points.length, 1],
  )
  return (params: Params) => mean(softplus(neg(mul(Y, net.apply(params as never, X) as Tensor))))
}

/** The sum of squares of every weight and bias: the L2 penalty that keeps the surface smooth. */
const squares = (params: Params) => treeLeaves<Tensor>(params).reduce((a, l) => add(a, sum(square(l.value))), 0)

type View = { step: number; loss: number; accuracy: number; z: number[][]; steps: number[]; losses: number[] }
const EMPTY: View = { step: 0, loss: NaN, accuracy: NaN, z: [], steps: [], losses: [] }

/**
 * A small network (2 → w → w → 1, tanh) trained by Adam in the page, with an L2 penalty on its weights so the surface
 * stays smooth: its decision surface redrawn as it learns, and its loss over the steps underneath. It trains while on
 * screen, a step a frame, and keeps going: once every point is right the loss still falls as the surface sharpens. A
 * click adds a point of the chosen class, which it then has to fit; new weights, a new dataset or width start over.
 */
export function NetworkTraining() {
  const s = useFigureState({
    data: choice(ORDER, 'spirals', { label: 'dataset' }),
    width: choice(WIDTHS, '32', { label: 'units per layer' }),
    l2: slider(0, 0.003, 0.0001, { step: 0.0001, label: 'L2 penalty' }),
    add: choice(['a', 'b'], 'b', { label: 'a click adds class' }),
  })
  const net = NETS[s.width]
  const data = s.data
  const [points, setPoints] = useState(() => DATASETS[data]())
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  const narrow = useNarrow()
  // The run's weights, optimiser state and seed; `run` is the run they belong to; a new one starts from new weights.
  const live = useRef({ run: -1, seed: 0, params: null as Params | null, state: null as RuleState | null })
  const [run, setRun] = useState(0)
  const [view, setView] = useState<View>(EMPTY)
  const tick = useRef(0)
  const restart = () => {
    setRun((n) => n + 1)
    setView(EMPTY)
  }
  // A new dataset or width starts a new run.
  const [seen, setSeen] = useState({ data, width: s.width })
  if (seen.data !== data || seen.width !== s.width) {
    if (seen.data !== data) setPoints(DATASETS[data]())
    setSeen({ data, width: s.width })
    restart()
  }
  useFrames(shown, () => {
    const r = live.current
    if (r.run !== run || !r.params) {
      const seed = r.seed + 1
      live.current = { run, seed, params: net.init(stream(`home/net/${seed}`)) as Params, state: null }
      return
    }
    const loss = lossOf(net, points)
    const penalised = (p: Params) => add(loss(p), mul(s.l2, squares(p)))
    const g = grad(penalised as never) as unknown as (p: Params) => Params
    r.state ??= rule.init(r.params)
    for (let k = 0; k < STEPS; k++) {
      const u = rule.update(g(r.params), r.state, r.params)
      r.state = u.state
      r.params = applyUpdates(r.params, u.updates)
    }
    const step = view.step + STEPS
    // The surface is redrawn every other frame: it costs more than the training.
    const redraw = (tick.current = (tick.current + 1) % 2) === 0 || view.z.length === 0
    const z = redraw ? surface(toFlat(net.apply(r.params as never, GRID) as Tensor)) : view.z
    const own = toFlat(net.apply(r.params as never, tensor(points.map((p) => [p.x, p.y]))) as Tensor)
    const accuracy = points.reduce((a, p, i) => a + ((own[i] > 0 ? 1 : 0) === p.c ? 1 : 0), 0) / points.length
    const value = Number(loss(r.params))
    setView((v) => {
      const thin = v.steps.length >= CURVE
      const keep = <T,>(a: T[]) => (thin ? a.filter((_, i) => i % 2 === 0) : a)
      return { step, loss: value, accuracy, z, steps: [...keep(v.steps), step], losses: [...keep(v.losses), value] }
    })
  })
  const x = useAxis({ label: 'x₁', range: [-4, 4] })
  const y = useAxis({ label: 'x₂', range: [-4, 4], equal: x })
  const steps = useAxis({ label: 'step', range: [0, Math.max(300, view.step)] })
  const losses = useAxis({ label: 'data loss', log: true, range: [0.001, 1] })
  return (
    <div ref={box}>
      <Figure
        title="Watch a network learn"
        purpose="A two-layer network trained by Adam, live in the page, on two interleaved spirals."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        aspect={0.95}
        controlsCollapsed
        readouts={
          <div className="flex flex-wrap items-center gap-2">
            <StatusText>
              {Number.isFinite(view.loss)
                ? `Step ${view.step}: data loss ${view.loss.toFixed(3)}, ${(100 * view.accuracy).toFixed(0)}% of points right.`
                : 'Starting…'}
            </StatusText>
            <Button size="sm" variant="outline" className="ml-auto" onClick={restart}>
              New weights
            </Button>
            <Button size="sm" variant="outline" onClick={() => setPoints(DATASETS[data]())}>
              Reset points
            </Button>
          </div>
        }
        caption="Click the plot to add a point of the chosen class. Fewer units and it cannot fold the plane enough; more L2 and the surface is smoother but less sure."
      >
        <Plot
          x={x}
          y={y}
          onPlotClick={([a, b]) => setPoints((ps) => [...ps, { x: a, y: b, c: s.add === 'a' ? 0 : 1 }])}
        >
          {view.z.length > 0 && (
            <Raster
              x={fx}
              y={fx}
              z={view.z}
              scale="diverging"
              range={[-1, 1]}
              fillOpacity={0.45}
              valueLabel="2p − 1"
              colorBar={!narrow}
              boundary
              live
            />
          )}
          <Points
            name="points"
            x={points.map((p) => p.x)}
            y={points.map((p) => p.y)}
            group={points.map((p) => p.c)}
            groupNames={['a', 'b']}
            live
          />
        </Plot>
        <Plot x={steps} y={losses} scale={0.3}>
          <Curve name="data loss" x={view.steps} y={view.losses.map((l) => Math.max(l, 0.001))} live />
        </Plot>
      </Figure>
    </div>
  )
}
