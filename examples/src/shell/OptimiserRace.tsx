import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { adagrad, adam, gradientDescent, momentum, nesterov, rmsprop } from 'aifn-compute/optim/first-order'
import { lbfgs } from 'aifn-compute/optim/second-order'
import { Contours, Curve, Figure, Handle, Plot, Points, Raster, useAxis, type Vec2 } from 'aifn-render'
import { useRef, useState } from 'react'
import { grid } from '@examples/data'
import { useFrames, useOnScreen, useTouched, useNarrow } from './live'

/** Himmelblau's function: four minima of equal depth, so where an optimiser ends up depends on how it moves. */
const f = (a: number, b: number) => (a * a + b - 11) ** 2 + (a + b * b - 7) ** 2
const objective = (x: Tensor) => {
  const [a, b] = toFlat(x)
  const u = a * a + b - 11
  const v = a + b * b - 7
  return { value: u * u + v * v, grad: [4 * a * u + 2 * v, 2 * u + 4 * b * v] }
}

const xs = grid(-5, 5, 110)
const z = xs.map((b) => xs.map((a) => Math.log10(1 + f(a, b))))
/**
 * Every first-order method of `aifn-compute/optim/first-order`, with step sizes that keep each stable here, and L-BFGS,
 * which builds a curvature estimate from its last few gradients and stops once converged.
 */
const METHODS = [
  { name: 'gradient descent', alg: gradientDescent(objective, { stepSize: 0.004 }) },
  { name: 'line search', alg: gradientDescent(objective, { lineSearch: 'backtracking' }) },
  { name: 'momentum', alg: momentum(objective, { stepSize: 0.001, momentum: 0.9 }) },
  { name: 'Nesterov', alg: nesterov(objective, { stepSize: 0.001, momentum: 0.9 }) },
  { name: 'AdaGrad', alg: adagrad(objective, { stepSize: 0.4 }) },
  { name: 'RMSProp', alg: rmsprop(objective, { stepSize: 0.03 }) },
  { name: 'Adam', alg: adam(objective, { stepSize: 0.15 }) },
  { name: 'L-BFGS', alg: lbfgs(objective) },
]
const STEPS = 150

function pathOf<S extends Status & { x: Tensor }>(alg: Algorithm<{ x0: Vec2 }, S>, x0: Vec2) {
  const px: number[] = []
  const py: number[] = []
  for (const { state } of live(alg, { x0 })) {
    const [a, b] = toFlat(state.x)
    px.push(a)
    py.push(b)
    if (px.length > STEPS) break
  }
  return { px, py }
}

/** Where the start wanders on its own: a slow Lissajous figure through all four basins. */
const wander = (t: number): Vec2 => [3.4 * Math.sin(0.21 * t - 0.3), 3.2 * Math.sin(0.29 * t + 3.3)]

/**
 * Eight optimisers raced from one start over Himmelblau's four-minimum landscape. On screen, the start wanders on its
 * own and the paths follow; dragging it takes over, and the wandering glides back a few seconds after the reader lets go.
 */
export function OptimiserRace() {
  const [start, setStart] = useState<Vec2>(wander(0))
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  const narrow = useNarrow()
  const { touched, touch } = useTouched(5)
  useFrames(shown && !touched, (t, dt) => {
    const target = wander(t)
    const k = Math.min(1, 2.5 * dt)
    setStart((s) => [s[0] + k * (target[0] - s[0]), s[1] + k * (target[1] - s[1])])
  })
  const paths = METHODS.map((m) => pathOf(m.alg as Algorithm<{ x0: Vec2 }, Status & { x: Tensor }>, start))
  const x = useAxis({ label: 'x₁', range: [-5, 5] })
  const y = useAxis({ label: 'x₂', range: [-5, 5], equal: x })
  return (
    <div ref={box}>
      <Figure
        title="Race the optimisers"
        purpose="Seven first-order methods and L-BFGS from the same start, up to 150 steps each."
        hoverReadout={false}
        defaultSize="L"
        aspect={1}
        controlsCollapsed
        caption="The start wanders on its own; drag it to take over. Four minima, and the methods split between them."
      >
        <Plot x={x} y={y}>
          <Raster x={xs} y={xs} z={z} fillOpacity={0.55} valueLabel="log₁₀(1 + f)" colorBar={!narrow} />
          <Contours x={xs} y={xs} z={z} levels={[0.5, 1, 1.5, 2, 2.5]} />
          {paths.map((p, i) => (
            <Curve key={i} name={METHODS[i].name} x={p.px} y={p.py} slot={i} live />
          ))}
          <Points
            name="end"
            x={paths.map((p) => p.px.at(-1)!)}
            y={paths.map((p) => p.py.at(-1)!)}
            group={METHODS.map((_, i) => i)}
            groupNames={METHODS.map((m) => m.name)}
            live
          />
          <Handle
            kind="point"
            at={start}
            onDrag={(p) => {
              touch()
              setStart(p)
            }}
            label="start"
          />
        </Plot>
      </Figure>
    </div>
  )
}
