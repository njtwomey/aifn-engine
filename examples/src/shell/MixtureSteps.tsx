import { Curve, Figure, Player, Plot, Points, Shapes, StatusText, useAxis } from 'aifn-render'
import { playTime, useAutoPlay } from './autoplay'
import { DATA, ellipse, em, K, type Phase } from './gmm'

const NAMES = ['a', 'b', 'c', 'd', 'e']
const BOX = 4.5
const FRAMES = em()

/** Milliseconds at each frame while the figure plays itself: a look at the start, slow at first, then faster. */
const wait = (step: number) => {
  const f = FRAMES[step]
  return f.phase === 'start' ? 800 : f.phase === 'done' ? 2500 : f.iter <= 4 ? 220 : 90
}
/** How long the figure takes to play itself through, plus a moment on the end: its carousel slide's duration. */
export const MIXTURE_PLAY_MS = playTime(FRAMES.length, wait) + 2500

const TEXT: Record<Phase, string> = {
  start: 'all five means in one corner, unit covariances',
  iterate: 'E-step, then M-step',
  done: 'converged',
}

/** A closed polyline as one curve's coordinates. */
const xy = (pts: number[][]) => ({ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) })

/**
 * A Gaussian mixture fitted by EM from a bad start, one iteration a frame, to convergence: the ellipses are each component's 1 and 2 sd contours, and the points take the colour of their most
 * likely component. While on screen and untouched it plays itself; the player takes over once used.
 */
export function MixtureSteps() {
  const { ref, step, set } = useAutoPlay(FRAMES.length, wait)
  const f = FRAMES[step]
  const { weights, means, covs } = f.mixture
  const trails = means.map((_, k) => xy(FRAMES.slice(0, step + 1).map((g) => g.mixture.means[k])))
  const x = useAxis({ label: 'x₁', range: [-BOX, BOX] })
  const y = useAxis({ label: 'x₂', range: [-BOX, BOX], equal: x })
  return (
    <div ref={ref}>
      <Figure
        title="A Gaussian mixture, fitted by EM"
        purpose="Five components from one corner, run until the likelihood stops rising."
        hoverReadout={false}
        defaultSize="L"
        controls={
          <Player value={step} onChange={set} count={FRAMES.length} format={(s) => `iteration ${FRAMES[s].iter}`} />
        }
        readouts={
          <StatusText>{`Iteration ${f.iter}: ${TEXT[f.phase]}. Log-likelihood ${f.logLik.toFixed(1)}.`}</StatusText>
        }
      >
        <Plot x={x} y={y}>
          <Shapes
            shapes={means.map((m, k) => ({
              contours: [ellipse(m, covs[k], 2).map((p) => [p[0], p[1]] as const)],
              tone: k,
              opacity: 0.08 + 0.3 * weights[k],
            }))}
            live
          />
          <Points
            name="points"
            x={DATA.map((p) => p[0])}
            y={DATA.map((p) => p[1])}
            group={f.labels}
            groupNames={NAMES}
            live
          />
          {means.flatMap((m, k) => {
            const one = xy(ellipse(m, covs[k], 1))
            const two = xy(ellipse(m, covs[k], 2))
            return [
              <Curve key={`1-${k}`} x={one.x} y={one.y} slot={k} silent live />,
              <Curve key={`2-${k}`} x={two.x} y={two.y} slot={k} dashed silent live />,
            ]
          })}
          {trails.map((t, k) => (
            <Curve key={`t-${k}`} x={t.x} y={t.y} slot={k} thin silent live />
          ))}
          <Points
            name="means"
            x={means.map((m) => m[0])}
            y={means.map((m) => m[1])}
            group={Array.from({ length: K }, (_, k) => k)}
            groupNames={NAMES}
            shape={3}
            size={14}
            live
          />
        </Plot>
      </Figure>
    </div>
  )
}
