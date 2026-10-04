import { useMemo } from 'react'
import { float, useFigureState } from 'aifn-render'
import { GymTrainer, trainingRun, type GridRendererOptions } from 'aifn-render/gym'
import { gymSetup } from 'aifn-methods/gym'
import { greedyPolicy, valuesFromQ, type TdAgentState } from 'aifn-methods/gym/agents'
import type { MdpEnvironment } from 'aifn-methods/gym/environments'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Gym trainer',
  question: 'How do I train an agent in an environment and play back any episode?',
  explain:
    '`gymSetup(envKey, envParams, agentKey, agentParams)` (aifn-methods) builds a compute `GymSetup` from the gym registries; `GymTrainer` trains it in the worker when Train is pressed, streams the curves, and replays or evaluates the episode picked on them. Spread `trainingRun(defaults)` into the state as `run`; a grid renderer’s `overlay` draws what the agent knew after the episode.',
}

export default function Trainer() {
  const state = useFigureState({
    epsilon: float(0.1, { label: 'exploration ε', ge: 0, le: 1, suggestions: [0.05, 0.1, 0.2] }),
    run: trainingRun({ episodes: 200 }),
  })
  const setup = useMemo(
    () => gymSetup('mazeEnvironment', { layout: 'small' }, 'qLearningAgent', { epsilon: state.epsilon }),
    [state.epsilon],
  )
  // The learnt values max_a Q(s, a) and greedy policy after the chosen episode, under its path.
  const options = useMemo((): GridRendererOptions => {
    const { model } = setup.env as MdpEnvironment
    return {
      overlay: (agent) => ({
        value: { values: valuesFromQ(model, (agent as TdAgentState).Q).data, label: 'max Q', fillOpacity: 0.6 },
        policy: greedyPolicy(model, (agent as TdAgentState).Q).data,
      }),
    }
  }, [setup])
  return (
    <GymTrainer
      title="Q-learning in a small maze"
      purpose="Train, then drag the episode marker and play the episode."
      state={state}
      setup={setup}
      rendererOptions={options}
      defaultSize="L"
    />
  )
}
