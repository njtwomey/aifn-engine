import { grad } from 'aifn-compute/foundation/autodiff'
import { treeLeaves, type Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { add, mean, mul, neg, square, sum, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { softplus } from 'aifn-compute/numerics/special'
import { Mlp } from 'aifn-compute/nn'
import { adamRule, applyUpdates, type RuleState } from 'aifn-compute/optim/first-order'
import { Button, choice, Figure, Plot, Points, Raster, slider, StatusText, useAxis, useFigureState } from 'aifn-render'
import { useRef, useState } from 'react'
import { grid } from '@examples/data'
import { useFrames, useOnScreen } from './live'

type Pt = { x: number; y: number; c: 0 | 1 }

/** Two interleaved spirals, the classic hard case for a small network: a straight line gets half of them wrong. */
function spirals(): Pt[] {
  const out: Pt[] = []
  for (let i = 0; i < 90; i++) {
    const r = 0.35 + (3.3 * i) / 90
    const t = 1.9 * r
    const jitter = 0.12 * Math.sin(i * 12.9898)
    for (const c of [0, 1] as const) {
      const a = t + c * Math.PI
      out.push({ x: (r + jitter) * Math.cos(a), y: (r - jitter) * Math.sin(a), c })
    }
  }
  return out
}

const net = Mlp([2, 24, 24, 1], { activation: 'tanh' })
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
/** Milliseconds of training per frame: the rest of the frame draws. */
const BUDGET = 9

/** The logistic loss of the network's logits against labels ±1, mean over the points. */
function lossOf(points: Pt[]) {
  const X = tensor(points.map((p) => [p.x, p.y]))
  const Y = tensor(
    points.map((p) => (p.c ? 1 : -1)),
    [points.length, 1],
  )
  return (params: Params) => mean(softplus(neg(mul(Y, net.apply(params as never, X) as Tensor))))
}

/** The sum of squares of every weight and bias: the L2 penalty that keeps the surface smooth. */
const squares = (params: Params) => treeLeaves<Tensor>(params).reduce((a, l) => add(a, sum(square(l.value))), 0)

/**
 * A small network (2 → 24 → 24 → 1, tanh) trained by Adam in the page on two spirals, with an L2 penalty on its weights
 * so the surface stays smooth, its decision surface redrawn as it learns. It trains while on screen; a click adds a point of the chosen class, which it then has to fit. Once it
 * has fitted everything it holds a moment, then starts again from new random weights.
 */
export function NetworkTraining() {
  const s = useFigureState({
    l2: slider(0, 0.02, 0.003, { label: 'L2 penalty' }),
    add: choice(['a', 'b'], 'b', { label: 'a click adds class' }),
  })
  const [points, setPoints] = useState(spirals)
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  const run = useRef({ seed: 0, params: net.init(stream('home/net/0')) as Params, state: null as RuleState | null })
  const [view, setView] = useState({ step: 0, loss: NaN, accuracy: NaN, z: [] as number[][] })
  const fitted = useRef(0)
  const tick = useRef(0)
  const restart = () => {
    const seed = run.current.seed + 1
    run.current = { seed, params: net.init(stream(`home/net/${seed}`)) as Params, state: null }
    fitted.current = 0
    setView((v) => ({ ...v, step: 0 }))
  }
  useFrames(shown, (_t, dt) => {
    const r = run.current
    const loss = lossOf(points)
    const penalised = (p: Params) => add(loss(p), mul(s.l2, squares(p)))
    const g = grad(penalised as never) as unknown as (p: Params) => Params
    r.state ??= rule.init(r.params)
    let steps = 0
    const t0 = performance.now()
    while (performance.now() - t0 < BUDGET) {
      const u = rule.update(g(r.params), r.state, r.params)
      r.state = u.state
      r.params = applyUpdates(r.params, u.updates)
      steps++
    }
    // The surface is redrawn every other frame: it costs about as much as the training.
    const redraw = (tick.current = (tick.current + 1) % 2) === 0 || view.z.length === 0
    const z = redraw ? surface(toFlat(net.apply(r.params as never, GRID) as Tensor)) : view.z
    const own = toFlat(net.apply(r.params as never, tensor(points.map((p) => [p.x, p.y]))) as Tensor)
    const accuracy = points.reduce((a, p, i) => a + ((own[i] > 0 ? 1 : 0) === p.c ? 1 : 0), 0) / points.length
    fitted.current = accuracy === 1 ? fitted.current + dt : 0
    if (fitted.current > 3) restart()
    else setView((v) => ({ step: v.step + steps, loss: Number(loss(r.params)), accuracy, z }))
  })
  const x = useAxis({ label: 'x₁', range: [-4, 4] })
  const y = useAxis({ label: 'x₂', range: [-4, 4], equal: x })
  return (
    <div ref={box}>
      <Figure
        title="Watch a network learn"
        purpose="A two-layer network trained by Adam, live in the page, on two interleaved spirals."
        state={s}
        hoverReadout={false}
        defaultSize="L"
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
            <Button size="sm" variant="outline" onClick={() => setPoints(spirals())}>
              Reset points
            </Button>
          </div>
        }
        caption="Click the plot to add a point of the chosen class. Raise the L2 penalty for a smoother, less certain fit."
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
      </Figure>
    </div>
  )
}
