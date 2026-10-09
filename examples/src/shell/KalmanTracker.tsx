import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { kalmanStep, parseModel } from 'aifn-compute/inference/filtering'
import {
  Curve,
  Figure,
  Handle,
  Plot,
  Points,
  slider,
  StatusText,
  useAxis,
  useFigureState,
  type Vec2,
} from 'aifn-render'
import { useMemo, useRef, useState } from 'react'
import { grid, rng } from '@examples/data'
import { useFrames, useOnScreen, useTouched } from './live'

/** Seconds of model time per frame. */
const DT = 0.05
/** Frames of history drawn. */
const KEEP = 120
/** Where the target goes on its own: a slow figure of eight. */
const path = (t: number): Vec2 => [2.6 * Math.sin(0.55 * t), 1.7 * Math.sin(1.1 * t)]
const ring = grid(0, 2 * Math.PI, 60)

/**
 * The constant-velocity model: the state is position and velocity, (x, y, vₓ, v_y); positions advance by velocity
 * × DT, velocities drift by process noise, and only the position is observed, with noise of standard deviation `r`.
 */
function model(r: number) {
  const q = 4
  return parseModel(
    {
      A: [
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ],
      C: [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
      ],
      Q: [
        [(q * DT ** 3) / 3, 0, (q * DT ** 2) / 2, 0],
        [0, (q * DT ** 3) / 3, 0, (q * DT ** 2) / 2],
        [(q * DT ** 2) / 2, 0, q * DT, 0],
        [0, (q * DT ** 2) / 2, 0, q * DT],
      ],
      R: [
        [r * r, 0],
        [0, r * r],
      ],
      m0: [0, 0, 0, 0],
      P0: [
        [4, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 4],
      ],
    },
    'KalmanTracker',
  )
}

/** The 2σ ellipse of a 2×2 covariance [[a, b], [b, d]] about (x, y). */
function ellipse(x: number, y: number, a: number, b: number, d: number) {
  const half = (a + d) / 2
  const s = Math.sqrt(((a - d) / 2) ** 2 + b * b)
  const [l1, l2] = [half + s, Math.max(half - s, 0)]
  const th = 0.5 * Math.atan2(2 * b, a - d)
  return ring.map((t) => {
    const u = 2 * Math.sqrt(l1) * Math.cos(t)
    const v = 2 * Math.sqrt(l2) * Math.sin(t)
    return [x + Math.cos(th) * u - Math.sin(th) * v, y + Math.sin(th) * u + Math.cos(th) * v]
  })
}

type Track = { truth: number[][]; obs: number[][]; est: number[][]; cov: number[] }

/**
 * A Kalman filter tracking a target from noisy position readings, one step a frame: the readings scatter, the estimate
 * follows smoothly, and its 2σ ellipse shows the filter's uncertainty. The target loops on its own; dragging it takes
 * over, and the filter has to catch up.
 */
export function KalmanTracker() {
  const s = useFigureState({ noise: slider(0.05, 1.2, 0.45, { label: 'reading noise (sd)' }) })
  const md = useMemo(() => model(s.noise), [s.noise])
  const [target, setTarget] = useState<Vec2>(path(0))
  const [track, setTrack] = useState<Track>({ truth: [], obs: [], est: [], cov: [4, 0, 4] })
  const filter = useRef<{ mean: Tensor; cov: Tensor }>({ mean: md.m0, cov: md.P0 })
  const noise = useRef(rng(7))
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  const { touched, touch } = useTouched(3)
  useFrames(shown, (t, dt) => {
    let truth = target
    if (!touched) {
      const goal = path(t)
      const k = Math.min(1, 3 * dt)
      truth = [target[0] + k * (goal[0] - target[0]), target[1] + k * (goal[1] - target[1])]
      setTarget(truth)
    }
    const y = [truth[0] + s.noise * noise.current.normal(), truth[1] + s.noise * noise.current.normal()]
    const step = kalmanStep(md, filter.current.mean, filter.current.cov, y)
    filter.current = { mean: step.mean, cov: step.cov }
    const m = toFlat(step.mean)
    const P = toRows(step.cov)
    setTrack((tr) => ({
      truth: [...tr.truth, truth].slice(-KEEP),
      obs: [...tr.obs, y].slice(-KEEP / 2),
      est: [...tr.est, [m[0], m[1]]].slice(-KEEP),
      cov: [P[0][0], P[0][1], P[1][1]],
    }))
  })
  const now = track.est.at(-1)
  const ell = now ? ellipse(now[0], now[1], ...(track.cov as [number, number, number])) : []
  const err = now ? Math.hypot(now[0] - target[0], now[1] - target[1]) : NaN
  const x = useAxis({ label: 'x', range: [-4, 4] })
  const y = useAxis({ label: 'y', range: [-3, 3], equal: x })
  return (
    <div ref={box}>
      <Figure
        title="Track with a Kalman filter"
        purpose="Noisy readings of a moving target, and the filter's estimate of where it really is."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        readouts={
          <StatusText>
            {Number.isFinite(err)
              ? `Estimate ${err.toFixed(2)} from the target; readings scatter by ${s.noise.toFixed(2)}.`
              : 'Starting…'}
          </StatusText>
        }
        caption="Drag the ink target to steer it; the filter follows. More reading noise and it trusts its model more."
      >
        <Plot x={x} y={y}>
          <Curve name="target" x={track.truth.map((p) => p[0])} y={track.truth.map((p) => p[1])} muted live />
          <Points
            name="readings"
            x={track.obs.map((p) => p[0])}
            y={track.obs.map((p) => p[1])}
            slot={1}
            size={5}
            live
          />
          <Curve name="estimate" x={track.est.map((p) => p[0])} y={track.est.map((p) => p[1])} slot={0} live />
          {ell.length > 0 && (
            <Curve name="2σ" x={ell.map((p) => p[0])} y={ell.map((p) => p[1])} slot={0} dashed silent live />
          )}
          {now && <Points name="estimate now" x={[now[0]]} y={[now[1]]} slot={0} size={10} live />}
          <Handle
            kind="point"
            at={target}
            onDrag={(p) => {
              touch()
              setTarget(p)
            }}
            label="target"
          />
        </Plot>
      </Figure>
    </div>
  )
}
