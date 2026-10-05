/**
 * `GymTrainer`: train any agent in any environment headlessly in the compute worker (the setup's training generator,
 * `aifn-methods/gym` `training`), watch the learning curve fill in as summaries stream, pick an episode on it (drag or
 * click the marker), and play that episode step by step: `replay` re-runs the training episode exactly as it happened
 * (from the nearest checkpoint), `evaluate` plays a fresh greedy episode of the policy as it was after that episode.
 * The environment is drawn by the renderer for its `render.kind` (`GYM_RENDERERS`).
 *
 * The page owns the environment's and agent's controls in its `useFigureState`, spreads in the shared training-run
 * row (`run: trainingRun(defaults)`: the budget in episodes or steps, and the seed), which `GymTrainer` reads from
 * `state.run`, and passes a `setup`: the compute `GymSetup` (`aifn-methods/gym` `gymSetup(envKey, envParams, agentKey,
 * agentParams)`), which carries the environment and agent as values, the same two as worker calls, and the functions
 * that re-run episodes, so this view never imports an environment or an agent. Nothing trains until Train is pressed,
 * and a changed setting marks the shown run as stale until Train is pressed again, so the reader chooses the
 * configuration before spending the compute.
 */
import { useMemo, useState, type ReactNode } from 'react'
import type { GymSetup, Training, Trajectory } from 'aifn-compute/foundation/contracts'
import { Player, StatusText } from '../controls'
import { ControlRow, Figure, type FigureProps } from '../layout'
import { call, useStreamed } from '../state'
import { Button } from '../ui/button'
import { Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '../viz'
import { GYM_RENDERERS, renderKind, WIDE_KINDS } from './registry'
import type { GymRenderer } from './renderers'
import { hasStepSeries } from './series'
import { trainingRunOf } from './trainingRun'
import { StepSeries } from './StepSeries'

export type GymTrainerProps = {
  title: string
  purpose: ReactNode
  caption?: ReactNode
  /** The page's figure state, with the training-run row as `run` (its controls appear above the trainer's). */
  state: FigureProps['state']
  setup: GymSetup
  /** A renderer for this page, instead of the one for `render.kind`. */
  renderer?: GymRenderer
  /** The evaluation episode's seed. Default 'evaluate'. */
  evaluationSeed?: number | string
  defaultSize?: 'L' | 'XL'
  /** The mode an episode opens in. Default 'replay'. */
  initialMode?: 'replay' | 'evaluate'
  /** Options passed to the renderer (`GymRenderProps.options`). */
  rendererOptions?: Readonly<Record<string, unknown>>
  /** Agent scalars (`Agent.scalars` names) drawn against episode in a row of their own, at most three. */
  scalars?: readonly string[]
  /** Extra buttons beside Train (e.g. a page's presets). */
  actions?: ReactNode
  /** The episode Player's starting speed, in steps per second (default 60). */
  playbackSpeed?: number
  /** The page's own readouts, after the trainer's (a reference return, an optimal value). */
  readouts?: ReactNode
  /** What the scene shows before an episode can be played (default an empty plot saying what to do): the grid. */
  idleScene?: ReactNode
}

type Mode = 'replay' | 'evaluate'

/** A trailing moving average over `1 / fraction` of the run (at least 1 episode). */
function smooth(y: ArrayLike<number>, total: number, fraction: number): number[] {
  const w = Math.max(1, Math.round(total * fraction))
  const out: number[] = []
  let sum = 0
  for (let i = 0; i < y.length; i++) {
    sum += y[i]
    if (i >= w) sum -= y[i - w]
    out.push(sum / Math.min(i + 1, w))
  }
  return out
}

/** A setup with the training run (budget and seed) from the figure state's `run` row. */
type RunSetup = GymSetup & { episodes?: number; steps?: number; seed: number }

const keyOf = (s: RunSetup) => JSON.stringify([s.envCall, s.agentCall, s.steps ?? s.episodes, s.seed])

export function GymTrainer({
  title,
  purpose,
  caption,
  state,
  setup,
  renderer,
  evaluationSeed = 'evaluate',
  defaultSize = 'XL',
  initialMode = 'replay',
  rendererOptions,
  scalars: scalarNames = [],
  playbackSpeed = 60,
  actions,
  idleScene,
  readouts: extraReadouts,
}: GymTrainerProps) {
  // The setup of the run shown: none until Train is pressed, then the setup current at that press. A fresh object per
  // press, so pressing again with the same settings reruns.
  const [trained, setTrained] = useState<RunSetup | null>(null)
  const run0 = trainingRunOf(state)
  const current: RunSetup = { ...setup, ...run0 }
  const stale = trained !== null && keyOf(trained) !== keyOf(current)
  const shown = trained ?? current
  const task = useMemo(
    () =>
      trained
        ? call<Training<unknown>>(
            trained.trainingAddress,
            call(trained.envCall.address, trained.envCall.params),
            call(trained.agentCall.address, trained.agentCall.params),
            trained.steps !== undefined
              ? { steps: trained.steps, seed: trained.seed }
              : { episodes: trained.episodes, seed: trained.seed },
          )
        : null,
    [trained],
  )
  const run = useStreamed(task)
  const training = run.value
  const n = training?.episodes ?? 0
  const stepsDone = training?.steps ?? 0
  const stepBudget = shown.steps
  // How far the run is, as a fraction, and in words.
  const progress = stepBudget ? stepsDone / stepBudget : n / Math.max(1, shown.episodes ?? 1)
  const progressText = stepBudget
    ? `${n} episodes · ${stepsDone.toLocaleString()} / ${stepBudget.toLocaleString()} steps`
    : `${n} / ${shown.episodes} episodes`

  // The chosen episode follows the newest one until the reader picks one.
  // A pick belongs to the run it was made on; a new run starts from its newest episode again.
  const [picked, setPicked] = useState<{ task: typeof task; episode: number } | null>(null)
  const episode = Math.min(picked?.task === task ? picked.episode : n, n)
  const [mode, setMode] = useState<Mode>(initialMode)
  const evaluationEnv = setup.evaluationEnv ?? shown.env
  const shownEnv = mode === 'evaluate' ? evaluationEnv : shown.env

  // One job at a time: while the run streams in, only the learning curves update. No episode is replayed or drawn
  // until training ends, so the main thread does not re-simulate an episode on every streamed batch.
  const busy = run.running
  const trajectory = useMemo((): Trajectory<unknown, unknown, unknown> | null => {
    if (busy || !training || !trained || episode < 1) return null
    return mode === 'replay'
      ? trained.replay(training, episode)
      : trained.evaluate(evaluationEnv, training, episode, evaluationSeed)
  }, [busy, training, episode, mode, trained, evaluationSeed, evaluationEnv])
  // The agent's state after the chosen episode, computed once on first use (by a renderer's overlay).
  const agentAfter = useMemo(() => {
    let cached: { value: unknown } | null = null
    return () => {
      if (!trained || !training) return null
      cached ??= { value: trained.agentAfter(training, episode) }
      return cached.value
    }
  }, [trained, training, episode])
  // The step belongs to the trajectory it was set on: a newly chosen episode opens at step 0.
  const [stepOf, setStepOf] = useState<{ trajectory: typeof trajectory; step: number } | null>(null)
  const step = stepOf?.trajectory === trajectory ? stepOf.step : 0
  const setStep = (s: number) => setStepOf({ trajectory, step: s })
  const steps = trajectory?.states.length ?? 1
  const at = Math.min(step, steps - 1)

  const curves = useMemo(() => {
    if (!training) return null
    const x = Array.from({ length: training.episodes }, (_, i) => i + 1)
    const scalar = Object.entries(training.scalars)[0]
    const oneStep = training.lengths.every((l) => l === 1)
    let cum = 0
    // Episodes by outcome, and the share of failures over a trailing window, when the environment reports outcomes.
    const outcome = training.outcome
    const pick = (o: number) => x.filter((e) => outcome[e - 1] === o)
    const failX = pick(-1)
    const okX = pick(1)
    const span = Number.isNaN(training.total) ? training.episodes : training.total
    const w = Math.max(1, Math.round(span / 20))
    let fails = 0
    const failShare = Array.from(outcome, (o, i) => {
      fails += o === -1 ? 1 : 0
      if (i >= w && outcome[i - w] === -1) fails -= 1
      return fails / Math.min(i + 1, w)
    })
    return {
      x,
      // One-step episodes (a bandit's rounds) pay noisy single rewards: no raw curve, and a wider average.
      returns: oneStep ? null : Array.from(training.returns),
      smoothed: smooth(training.returns, span, oneStep ? 1 / 10 : 1 / 50),
      lengths: Array.from(training.lengths),
      regret: training.regret ? Array.from(training.regret, (r) => (cum += r)) : null,
      scalar: scalar ? { name: scalar[0], y: Array.from(scalar[1]) } : null,
      outcomes:
        failX.length + okX.length > 0
          ? {
              fail: { x: failX, y: failX.map((e) => training.returns[e - 1]) },
              ok: { x: okX, y: okX.map((e) => training.returns[e - 1]) },
              share: failShare,
              window: w,
            }
          : null,
    }
  }, [training])

  const kind = renderKind(shown.env)
  const Renderer = renderer ?? GYM_RENDERERS[kind]
  // Per-step panels (state series and actions against step) for environments that declare state series.
  const withSeries = hasStepSeries(shown.env)
  const wide = WIDE_KINDS.has(kind)
  // Under a step budget the number of episodes is not known in advance: the episode axis follows the run.
  const total = stepBudget ? Math.max(1, n) : (shown.episodes ?? 1)
  const ea = useAxis({ label: 'episode', range: [0, total], key: total })
  const ra = useAxis({ label: 'return', hold: 'union', key: task })
  const ba = useAxis({ label: curves?.regret ? 'cumulative regret' : 'length (steps)', hold: 'union', key: task })
  const middle = curves?.outcomes
    ? `failure share (last ${curves.outcomes.window})`
    : (curves?.scalar?.name ?? 'reward')
  const sa = useAxis({
    label: middle,
    hold: 'union',
    key: task,
    ...(curves?.outcomes && { range: [0, 1] as const }),
  })
  const sx = useAxis({ label: 'step' })
  // One axis per extra scalar panel (a fixed number of hooks).
  const extra = scalarNames.slice(0, 3)
  const xa = [
    useAxis({ label: extra[0] ?? '', hold: 'union', key: task }),
    useAxis({ label: extra[1] ?? '', hold: 'union', key: task }),
    useAxis({ label: extra[2] ?? '', hold: 'union', key: task }),
  ]
  const pick = (v: number) => setPicked({ task, episode: Math.max(1, Math.min(n, Math.round(v))) })
  // The marker stays on the chart (adding and removing a handle layer mid-stream would re-patch a missing series); it
  // only ignores drags while training runs.
  const marker = n > 0 && <Handle kind="x" at={episode} label="episode" onDrag={busy ? () => {} : pick} />
  const extraPlots =
    extra.length > 0 ? (
      <Plots cols={extra.length} scale={withSeries ? 0.3 : 0.4}>
        {extra.map((name, i) => {
          const y = training?.scalars[name]
          return (
            <Plot key={name} x={ea} y={xa[i]} legend={false}>
              {curves && y && <Curve name={name} x={curves.x} y={Array.from(y)} slot={i + 1} />}
              {marker}
            </Plot>
          )
        })}
      </Plots>
    ) : null

  const rewardsSoFar = trajectory ? trajectory.rewards.slice(0, at).reduce((a, b) => a + b, 0) : 0
  // At the last step of an episode that the environment calls a success or a failure, the scene takes that tone.
  const ending = trajectory?.ending ?? null
  const end = ending && at === steps - 1 ? ending : null
  const T = steps - 1
  const outcomeText = !trajectory
    ? '—'
    : ending
      ? ending.success
        ? /\bsteps?\b/.test(ending.reason)
          ? ending.reason
          : `${ending.reason} at step ${T}`
        : `failed at step ${T}: ${ending.reason}`
      : trajectory.reachedTerminal
        ? `terminal at step ${T}`
        : `truncated at step ${T}`
  const scene =
    training && trajectory && Renderer ? (
      <Renderer
        env={shownEnv}
        trajectory={trajectory}
        step={at}
        training={training}
        episode={episode}
        agentAfter={agentAfter}
        options={rendererOptions}
        end={end}
      />
    ) : idleScene && !trajectory ? (
      idleScene
    ) : (
      <Plot
        x={sx}
        y={sa}
        title={
          !trained
            ? 'press Train to start'
            : busy
              ? 'training: pick an episode on the curves when it finishes'
              : 'waiting for the first episodes'
        }
      />
    )
  const returnPlot = (
    <Plot x={ea} y={ra} title="return per episode">
      {curves?.returns && <Curve name="return" x={curves.x} y={curves.returns} muted thin />}
      {curves && <Curve name="moving average" x={curves.x} y={curves.smoothed} slot={0} />}
      {curves?.outcomes && (
        <Points name="failed" x={curves.outcomes.fail.x} y={curves.outcomes.fail.y} tone="destructive" thin />
      )}
      {curves?.outcomes && (
        <Points name="succeeded" x={curves.outcomes.ok.x} y={curves.outcomes.ok.y} tone="success" thin />
      )}
      {marker}
    </Plot>
  )
  const middlePlot = (
    <Plot x={curves?.outcomes || curves?.scalar ? ea : sx} y={sa} legend={false}>
      {curves?.outcomes ? (
        <Curve name={middle} x={curves.x} y={curves.outcomes.share} tone="destructive" />
      ) : curves?.scalar ? (
        <Curve name={curves.scalar.name} x={curves.x} y={curves.scalar.y} slot={2} />
      ) : (
        trajectory && (
          <Curve
            name="reward per step"
            x={trajectory.rewards.map((_, i) => i + 1)}
            y={trajectory.rewards}
            slot={2}
            showPoints
          />
        )
      )}
      {(curves?.outcomes || curves?.scalar) && marker}
    </Plot>
  )
  const lastPlot = (
    <Plot x={ea} y={ba} legend={false}>
      {curves && (
        <Curve
          name={curves.regret ? 'cumulative regret' : 'length'}
          x={curves.x}
          y={curves.regret ?? curves.lengths}
          slot={3}
        />
      )}
      {marker}
    </Plot>
  )
  return (
    <Figure
      title={title}
      purpose={purpose}
      state={state}
      defaultSize={defaultSize}
      controls={
        <>
          <ControlRow label="train">
            <div className="flex flex-wrap items-center gap-3">
              {run.running ? (
                <Button size="sm" variant="destructive" aria-label="Stop" onClick={run.stop}>
                  Stop
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant={!trained || stale ? 'default' : 'outline'}
                  aria-label="Train"
                  onClick={() => setTrained({ ...current })}
                >
                  {trained ? 'Retrain' : 'Train'}
                </Button>
              )}
              {actions}
              <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
                <div className="h-full bg-primary" style={{ width: `${100 * Math.min(1, progress)}%` }} />
              </div>
              <StatusText tone={run.error ? 'error' : !trained || stale ? 'attention' : 'muted'}>
                {!trained
                  ? 'Not trained yet: choose the settings, then press Train.'
                  : run.error
                    ? `failed: ${run.error}`
                    : stale
                      ? `Settings changed since this run (${progressText} shown): press Retrain to train with them.`
                      : run.stopped
                        ? `stopped at ${n} episodes / ${stepsDone.toLocaleString()} steps`
                        : `${progressText}${run.running ? '…' : ''}`}
              </StatusText>
            </div>
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="episode" value={episode} />
          <Readout label="step" value={`${at} / ${steps - 1}`} />
          <Readout label="reward so far" value={rewardsSoFar.toFixed(2)} />
          <Readout label="episode return" value={trajectory ? trajectory.episodeReturn.toFixed(2) : '—'} />
          <Readout label="outcome" value={outcomeText} />
          {extraReadouts}
        </>
      }
      caption={
        <>
          {caption} Trained headlessly in the worker by aifn <code>training</code> (
          {stepBudget ? `${stepBudget.toLocaleString()} steps` : `${total} episodes`}, seed {String(shown.seed)},
          checkpoints every {training?.every ?? '…'} episodes); drag or click the episode marker to pick an episode,
          then play it: replay re-runs the training episode from the nearest checkpoint, evaluate plays the greedy
          policy as of that episode on seed {String(evaluationSeed)}.
        </>
      }
    >
      {/* The flow, top to bottom: how training went, then the episode chosen on it, then that episode played. */}
      <Plots cols={3} scale={wide ? 0.36 : 0.5}>
        {returnPlot}
        {middlePlot}
        {lastPlot}
      </Plots>
      {extraPlots}
      <div className="flex flex-col gap-3">
        <ControlRow label={`episode ${episode}`}>
          <div className="flex items-center gap-1">
            {(['replay', 'evaluate'] as const).map((m) => (
              <Button
                key={m}
                size="sm"
                variant={mode === m ? 'default' : 'outline'}
                aria-label={m}
                aria-pressed={mode === m}
                onClick={() => setMode(m)}
              >
                {m === 'replay' ? 'replay (as trained)' : 'evaluate (greedy)'}
              </Button>
            ))}
          </div>
          <Player label="step" value={at} onChange={setStep} count={steps} defaultSpeed={playbackSpeed} />
        </ControlRow>
        {end && (
          <div
            role="status"
            className={
              end.success
                ? 'col-span-full w-fit rounded-md border border-success/40 bg-success/10 px-3 py-1.5 text-sm text-success'
                : 'col-span-full w-fit rounded-md border border-destructive/40 bg-destructive/10 px-3 py-1.5 text-sm text-destructive'
            }
          >
            {end.success ? '✓ ' : '✗ '}
            {outcomeText}
          </div>
        )}
      </div>
      <Plots cols={1} scale={wide ? (withSeries ? 0.26 : 0.45) : 0.55}>
        {scene}
      </Plots>
      {withSeries && trajectory && (
        <StepSeries env={shownEnv} trajectory={trajectory} step={at} onStep={setStep} scale={0.44} />
      )}
    </Figure>
  )
}
