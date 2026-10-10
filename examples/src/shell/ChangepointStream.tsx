import {
  bocpdForecast,
  bocpdInit,
  bocpdUpdate,
  mapChangepoints,
  normalGamma,
  runLengthRow,
  type BocpdState,
} from 'aifn-compute/inference/filtering'
import {
  Area,
  Button,
  Curve,
  Figure,
  int,
  Plot,
  Plots,
  Points,
  Raster,
  Segments,
  StatusText,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { useRef, useState } from 'react'
import { rng } from '@examples/data'
import { useFrames, useOnScreen, useNarrow } from './live'

/** Time steps on screen. */
const WINDOW = 160
/** Run lengths drawn (longer runs are tracked but not shown). */
const RUNS = 120
/** Observations per second while the stream plays. */
const RATE = 18
const MODEL = normalGamma({ mean: 0, kappa: 0.05, alpha: 1, beta: 0.4 })

type Regime = { mean: number; sd: number; until: number }
type Row = { t: number; x: number; truth: number; forecast: number; spread: number; map: number; logp: Float64Array }

/** A new regime from the generator: a mean, a noise level and how long it lasts. */
function nextRegime(r: ReturnType<typeof rng>, t: number): Regime {
  return { mean: -2.4 + 4.8 * r.uniform(), sd: 0.25 + 0.6 * r.uniform(), until: t + 40 + Math.floor(80 * r.uniform()) }
}

/**
 * Bayesian online changepoint detection (Adams and MacKay, 2007) on a live stream whose regimes shift without warning.
 * Each new point updates the posterior over the run length, the time since the last change: drawn below as a
 * heatmap, it climbs while a regime lasts and collapses to zero when one ends. Above, the data, the one-step forecast
 * the run-length mixture implies, and the changepoints read from the most probable run lengths.
 */
export function ChangepointStream() {
  const s = useFigureState({ gap: int(80, { min: 10, max: 400, label: 'expected gap (hazard 1/gap)' }) })
  const gen = useRef(rng(11))
  const regime = useRef<Regime | null>(null)
  const state = useRef<BocpdState>(bocpdInit(MODEL) as BocpdState)
  const [rows, setRows] = useState<Row[]>([])
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  const narrow = useNarrow()
  useFrames(
    shown,
    () => {
      const t = state.current.t
      if (!regime.current || t >= regime.current.until) regime.current = nextRegime(gen.current, t)
      const { mean, sd } = regime.current
      const x = mean + sd * gen.current.normal()
      const before = bocpdForecast(MODEL, state.current as never)
      state.current = bocpdUpdate(MODEL, state.current as never, x, {
        hazard: 1 / s.gap,
        threshold: 1e-7,
        maxRuns: 250,
      }) as BocpdState
      const row = runLengthRow(state.current, RUNS)
      setRows((rs) =>
        [
          ...rs,
          {
            t,
            x,
            truth: mean,
            forecast: before.mean,
            spread: 2 * Math.sqrt(Math.max(before.variance, 0)),
            map: state.current.map,
            logp: row.map((p) => Math.max(-6, Math.log10(Math.max(p, 1e-12)))),
          },
        ].slice(-WINDOW),
      )
    },
    RATE,
  )
  const ts = rows.map((r) => r.t)
  const t1 = ts.at(-1) ?? WINDOW
  const t0 = Math.max(0, t1 - WINDOW + 1)
  // Changepoints from the MAP run lengths over the window, as times.
  const changes = mapChangepoints(rows.map((r) => r.map)).map((i) => rows[i].t)
  const heat = Array.from({ length: RUNS }, (_, k) => rows.map((r) => r.logp[k]))
  const time = useAxis({ label: 't', range: [t0, t0 + WINDOW - 1] })
  const value = useAxis({ label: 'x', range: [-4, 4] })
  const run = useAxis({ label: 'run length', range: [0, RUNS - 1] })
  return (
    <div ref={box}>
      <Figure
        title="Spot the change"
        purpose="Bayesian online changepoint detection: a posterior over how long the current regime has lasted."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        aspect={0.9}
        controlsCollapsed
        readouts={
          <div className="flex flex-wrap items-center gap-2">
            <StatusText>
              {rows.length
                ? `Most probable run length ${rows.at(-1)!.map}; ${changes.length} changes found in view.`
                : 'Starting…'}
            </StatusText>
            <Button
              size="sm"
              variant="outline"
              className="ml-auto"
              onClick={() => (regime.current = nextRegime(gen.current, state.current.t))}
            >
              Shift now
            </Button>
          </div>
        }
        caption="Below: log₁₀ of the run-length posterior, with its most probable value in ink. Press Shift now and watch it reset."
      >
        <Plots rows={2} heights={[1, 1]} tight>
          <Plot x={time} y={value}>
            <Area
              name="forecast ± 2 sd"
              x={ts}
              y={rows.map((r) => r.forecast + r.spread)}
              base={rows.map((r) => r.forecast - r.spread)}
              slot={0}
              opacity={0.18}
              live
            />
            <Curve name="forecast" x={ts} y={rows.map((r) => r.forecast)} slot={0} live />
            <Curve name="true mean" x={ts} y={rows.map((r) => r.truth)} muted dashed live />
            <Points name="data" x={ts} y={rows.map((r) => r.x)} slot={1} size={4} live />
            <Segments segments={changes.map((c) => ({ from: [c, -4] as const, to: [c, 4] as const }))} emphasis live />
          </Plot>
          <Plot x={time} y={run}>
            {rows.length > 1 && (
              <Raster
                x={ts}
                y={Array.from({ length: RUNS }, (_, k) => k)}
                z={heat}
                range={[-6, 0]}
                valueLabel="log₁₀ p(run length)"
                colorBar={!narrow}
                live
              />
            )}
            <Curve name="most probable" x={ts} y={rows.map((r) => Math.min(r.map, RUNS - 1))} emphasis live />
          </Plot>
        </Plots>
      </Figure>
    </div>
  )
}
