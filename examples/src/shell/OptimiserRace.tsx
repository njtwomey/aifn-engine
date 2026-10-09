import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { adagrad, adam, gradientDescent, momentum, nesterov, rmsprop } from 'aifn-compute/optim/first-order'
import { Contours, Curve, Figure, Handle, Plot, Points, Raster, useAxis, type Vec2 } from 'aifn-render'
import { useState } from 'react'
import { grid } from '@examples/data'

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
/** Every first-order method of `aifn-compute/optim/first-order`, with step sizes that keep each stable here. */
const METHODS = [
  { name: 'gradient descent', alg: gradientDescent(objective, { stepSize: 0.004 }) },
  { name: 'line search', alg: gradientDescent(objective, { lineSearch: 'backtracking' }) },
  { name: 'momentum', alg: momentum(objective, { stepSize: 0.001, momentum: 0.9 }) },
  { name: 'Nesterov', alg: nesterov(objective, { stepSize: 0.001, momentum: 0.9 }) },
  { name: 'AdaGrad', alg: adagrad(objective, { stepSize: 0.4 }) },
  { name: 'RMSProp', alg: rmsprop(objective, { stepSize: 0.03 }) },
  { name: 'Adam', alg: adam(objective, { stepSize: 0.15 }) },
]
const STEPS = 150

function pathOf(alg: (typeof METHODS)[number]['alg'], x0: Vec2) {
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

/** Seven optimisers raced from one draggable start over Himmelblau's four-minimum landscape. */
export function OptimiserRace() {
  const [start, setStart] = useState<Vec2>([-1, -0.5])
  const paths = METHODS.map((m) => pathOf(m.alg, start))
  const x = useAxis({ label: 'x₁', range: [-5, 5] })
  const y = useAxis({ label: 'x₂', range: [-5, 5], equal: x })
  return (
    <Figure
      title="Race the optimisers"
      purpose="Seven first-order methods from the same start, 150 steps each."
      hoverReadout={false}
      defaultSize="L"
      caption="Drag the start: the landscape has four minima, and the methods split between them."
    >
      <Plot x={x} y={y}>
        <Raster x={xs} y={xs} z={z} fillOpacity={0.55} valueLabel="log₁₀(1 + f)" />
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
        <Handle kind="point" at={start} onDrag={setStart} label="start" />
      </Plot>
    </Figure>
  )
}
