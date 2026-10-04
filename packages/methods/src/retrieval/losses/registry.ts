/**
 * The registry of the ranking and retrieval losses, keyed by `info.key` (the application counterpart of
 * `aifn-compute/learning/losses`' `lossRegistry`).
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { Loss } from 'aifn-compute/learning/losses'
import * as ranking from './ranking'
import * as retrieval from './retrieval'

/** Every ranking and retrieval loss, keyed by its `info.key`. */
export const retrievalLossRegistry: Readonly<Record<string, Loss>> = entries(
  'loss',
  ranking,
  retrieval,
) as unknown as Readonly<Record<string, Loss>>
