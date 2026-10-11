import { partials } from 'aifn-compute/dynamics/fields'
import { grad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { add, fromData, mean, mul, square, sub, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { Mlp } from 'aifn-compute/nn'
import { xavierUniform } from 'aifn-compute/nn/init'
import { fullBatchTraining } from 'aifn-compute/nn/training'
import { adamRule, applyUpdates, type RuleState } from 'aifn-compute/optim/first-order'
import { Button, choice, Curve, Figure, Plot, Points, slider, StatusText, useAxis, useFigureState } from 'aifn-render'
import { useMemo, useRef, useState } from 'react'
import { useFrames, useOnScreen } from './live'

/** The oscillator's natural frequency ω₀: about two and a half swings on [0, 1]. */
const OMEGA = 15
/** Observations, evenly spaced from t = 0 to the end of the data. */
const OBSERVED = 10
/** Collocation points, where the equation is enforced: evenly spaced over the whole of [0, 1]. */
const COLLOCATION = 40
/** Steps a frame for each network: Adam's are cheap, an L-BFGS step is a line search. */
const STEPS = { adam: 6, lbfgs: 3 }
/** Adam steps before L-BFGS takes over, when it does: enough to leave the flat start. */
const WARMUP = 100
const RATES = ['0.003', '0.01', '0.03'] as const
/** Points of the error curves kept. */
const CURVE = 600
const net = Mlp([1, 32, 32, 1], { activation: 'tanh', init: xavierUniform() })
const ts = Array.from({ length: 151 }, (_, i) => i / 150)
const column = (v: readonly number[]) => fromData(Float64Array.from(v), [v.length, 1])
const T = column(ts)
const tc = Array.from({ length: COLLOCATION }, (_, i) => i / (COLLOCATION - 1))
const C = column(tc)
/** The network on [0, 1], its input scaled to [−1, 1]. */
const u = (p: Params, t: Value) => net.apply(p as never, sub(mul(t, 2), 1))

/** The underdamped oscillator from x = 1 at rest: x(t) = e^{−ζω₀t}(cos ω_d t + (ζω₀/ω_d) sin ω_d t). */
const exact = (zeta: number) => {
  const wd = OMEGA * Math.sqrt(1 - zeta * zeta)
  return (t: number) => Math.exp(-zeta * OMEGA * t) * (Math.cos(wd * t) + ((zeta * OMEGA) / wd) * Math.sin(wd * t))
}

type Lbfgs = Iterator<{ state: { params: unknown }; stopped?: unknown }>
type Fit = { params: Params; state: RuleState; lbfgs: Lbfgs | null; done: boolean }
type View = {
  step: number
  phase: 'adam' | 'lbfgs'
  data: number[]
  pinn: number[]
  steps: number[]
  dataErr: number[]
  pinnErr: number[]
}
const EMPTY: View = { step: 0, phase: 'adam', data: [], pinn: [], steps: [], dataErr: [], pinnErr: [] }

/**
 * A physics-informed network against a plain one on a damped oscillator, x″ + 2ζω₀x′ + ω₀²x = 0. Both see the same few
 * observations near the start and start from the same weights; the physics-informed one is also penalised for breaking
 * the equation at points spread over the whole interval, its derivatives taken by autodiff. The plain network fits the
 * points and then guesses; the equation carries the other through swings it never saw.
 */
export function PinnSolve() {
  const s = useFigureState({
    zeta: slider(0.02, 0.4, 0.22, { step: 0.02, label: 'damping ζ' }),
    until: slider(0.15, 0.6, 0.24, { step: 0.03, label: 'data up to t' }),
    optimiser: choice(['Adam, then L-BFGS', 'Adam'], 'Adam, then L-BFGS', { label: 'optimiser' }),
    rate: choice(RATES, '0.03', { label: 'Adam learning rate' }),
  })
  const [run, setRun] = useState(0)
  const problem = useMemo(() => {
    const x = exact(s.zeta)
    const td = Array.from({ length: OBSERVED }, (_, i) => (s.until * i) / (OBSERVED - 1))
    const X = column(td)
    const Y = column(td.map(x))
    const truth = ts.map(x)
    const norm = Math.hypot(...truth)
    const misfit = (p: Params) => mean(square(sub(u(p, X), Y)))
    // The residual divided by ω₀², so that its terms are of the size of x.
    const residual = (p: Params) => {
      const [v, vt, vtt] = partials((t) => u(p, t), C, [[], [0], [0, 0]])
      return add(add(mul(1 / OMEGA ** 2, vtt), mul((2 * s.zeta) / OMEGA, vt)), v)
    }
    const physics = (p: Params) => add(misfit(p), mean(square(residual(p))))
    return {
      td,
      yd: td.map(x),
      truth,
      error: (curve: number[]) => Math.hypot(...curve.map((v, i) => v - truth[i])) / norm,
      rule: adamRule({ stepSize: +s.rate }),
      lbfgs: s.optimiser !== 'Adam',
      data: { loss: misfit, grad: grad(misfit as never) as unknown as (p: Params) => Params },
      pinn: { loss: physics, grad: grad(physics as never) as unknown as (p: Params) => Params },
    }
  }, [s.zeta, s.until, s.rate, s.optimiser])
  // Both networks' weights and Adam states, for the problem and run they belong to.
  const fits = useRef<{ problem: unknown; run: number; data: Fit; pinn: Fit } | null>(null)
  const [view, setView] = useState<View>(EMPTY)
  const [seen, setSeen] = useState({ problem, run })
  if (seen.problem !== problem || seen.run !== run) {
    setSeen({ problem, run })
    setView(EMPTY)
  }
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  useFrames(shown, () => {
    if (fits.current?.problem !== problem || fits.current.run !== run) {
      const params = net.init(stream(`home/pinn/${run}`)) as Params
      const start = (): Fit => ({ params, state: problem.rule.init(params), lbfgs: null, done: false })
      fits.current = { problem, run, data: start(), pinn: start() }
    }
    const f = fits.current!
    const phase = problem.lbfgs && view.step >= WARMUP ? 'lbfgs' : 'adam'
    const train = (fit: Fit, objective: { loss: (p: Params) => Value; grad: (p: Params) => Params }) => {
      for (let k = 0; k < STEPS[phase] && !fit.done; k++) {
        if (phase === 'adam') {
          const next = problem.rule.update(objective.grad(fit.params), fit.state, fit.params)
          fit.params = applyUpdates(fit.params, next.updates)
          fit.state = next.state
          continue
        }
        if (!fit.lbfgs) {
          fit.lbfgs = live(fullBatchTraining({ loss: objective.loss, data: null }), { params: fit.params }) as Lbfgs
          fit.lbfgs.next()
        }
        const next = fit.lbfgs.next()
        if (next.done) fit.done = true
        else {
          fit.params = next.value.state.params as Params
          if (next.value.stopped) fit.done = true
        }
      }
    }
    train(f.data, problem.data)
    train(f.pinn, problem.pinn)
    const data = Array.from(toFlat(u(f.data.params, T) as Tensor))
    const pinn = Array.from(toFlat(u(f.pinn.params, T) as Tensor))
    setView((v) => {
      // Past CURVE points every other one is dropped, so a long run stays cheap to draw.
      const keep = <T,>(a: T[]) => (v.steps.length >= CURVE ? a.filter((_, i) => i % 2 === 0) : a)
      return {
        step: v.step + STEPS[phase],
        phase,
        data,
        pinn,
        steps: [...keep(v.steps), v.step + STEPS[phase]],
        dataErr: [...keep(v.dataErr), problem.error(data)],
        pinnErr: [...keep(v.pinnErr), problem.error(pinn)],
      }
    })
  })
  const t = useAxis({ label: 't', range: [0, 1] })
  const x = useAxis({ label: 'x', range: [-1.3, 1.3] })
  const steps = useAxis({ label: 'step', range: [0, Math.max(1000, view.step)] })
  const err = useAxis({ label: 'relative error', log: true, range: [0.001, 10] })
  const last = (a: number[]) => (100 * (a.at(-1) ?? NaN)).toFixed(1)
  return (
    <div ref={box}>
      <Figure
        title="Physics fills the gap"
        purpose="Two networks see the same few points of a damped oscillator; one is also told the equation it obeys."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        aspect={0.95}
        controlsCollapsed
        readouts={
          <div className="flex flex-wrap items-center gap-2">
            <StatusText>
              {view.step
                ? `Step ${view.step} (${view.phase === 'adam' ? 'Adam' : 'L-BFGS'}): ${last(view.pinnErr)}% off with the equation, ${last(view.dataErr)}% without.`
                : 'Starting…'}
            </StatusText>
            <Button size="sm" variant="outline" className="ml-auto" onClick={() => setRun((n) => n + 1)}>
              New weights
            </Button>
          </div>
        }
        caption="Dashes: the true motion. Ticks along the bottom: where the equation is enforced. After a hundred Adam steps L-BFGS takes over and homes in; stop the data earlier and only the plain network suffers."
      >
        <Plot x={t} y={x}>
          <Curve name="true motion" x={ts} y={problem.truth} muted dashed />
          {view.data.length > 0 && <Curve name="data only" x={ts} y={view.data} slot={1} live />}
          {view.pinn.length > 0 && <Curve name="with the equation" x={ts} y={view.pinn} slot={0} emphasis live />}
          <Points name="observations" x={problem.td} y={problem.yd} slot={2} size={9} />
          <Points name="collocation" x={tc} y={tc.map(() => -1.22)} size={3} muted />
        </Plot>
        <Plot x={steps} y={err} scale={0.3}>
          <Curve name="data only" x={view.steps} y={view.dataErr.map((e) => Math.min(e, 10))} slot={1} live />
          <Curve
            name="with the equation"
            x={view.steps}
            y={view.pinnErr.map((e) => Math.max(e, 0.001))}
            emphasis
            live
          />
        </Plot>
      </Figure>
    </div>
  )
}
