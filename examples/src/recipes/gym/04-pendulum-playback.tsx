import { Figure, Player, usePlayhead } from 'aifn-render'
import { PendulumView } from 'aifn-render/gym'
import { runEpisode } from 'aifn-methods/gym'
import { swingUpAgent } from 'aifn-methods/gym/agents'
import { pendulumEnvironment } from 'aifn-methods/gym/environments'
import { stream } from 'aifn-compute/foundation/random'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Environment playback',
  question: 'How do I play back an episode of a control environment?',
  explain:
    '`runEpisode` (aifn-methods) gives a trajectory of environment states; `PendulumView` (or `CartPoleView`) draws one state from the environment’s `render` spec, with a faint trail of recent states. A `Player` walks the steps.',
}

const env = pendulumEnvironment({ start: { theta: Math.PI, thetaDot: 0 } })
const agent = swingUpAgent()
const { trajectory } = runEpisode(env, agent, agent.init(env, stream('init')), stream(1), 'greedy')
const states = trajectory.states

export default function PendulumPlayback() {
  const [k, setK] = usePlayhead(states.length)
  return (
    <Figure
      title="Swing-up of a pendulum"
      purpose="Energy pumping swings the pendulum up from hanging; a linear controller then balances it."
      controls={<Player value={k} onChange={setK} count={states.length} label="step" />}
    >
      <PendulumView render={env.render} state={states[k]} trail={states.slice(Math.max(0, k - 20), k + 1)} />
    </Figure>
  )
}
