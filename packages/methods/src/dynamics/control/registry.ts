/**
 * The registry of `aifn-methods/dynamics/control`: the PID loop as a traceable algorithm.
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn-compute/foundation/registry'
import * as pid from './pid'

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/control')

algorithm(
  {
    key: 'pidLoop',
    name: 'PID control loop',
    summary:
      'A discrete PID controller around a SISO plant, with a filtered derivative, actuator limits and anti-windup.',
    problem: 'lti-system',
    state: { iterate: 'x', objective: 'e', flags: ['diverged'] },
    notes: ['proportional-integral-derivative-control'],
    cite: ['astrom2021', 'ziegler1942'],
  },
  pid.pidLoop,
)

/** The algorithms of the module, keyed by factory name. */
export const pidAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', pid) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
