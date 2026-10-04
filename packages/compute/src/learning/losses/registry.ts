/**
 * The registry of every loss in the module, keyed by `info.key`, so that the lab and training code can list losses and
 * filter them by family, input kind or note.
 */

import * as adversarial from './adversarial'
import * as classification from './classification'
import { entries } from 'aifn-compute/foundation/registry'
import { type Loss, type LossFamily, type LossInput } from './core'
import * as divergence from './divergence'
import * as energy from './energy'
import * as mixture from './mixture'
import * as regression from './regression'
import * as representation from './representation'
import * as weak from './weak'
import { DomainError } from 'aifn-compute/foundation/errors'

const modules = [classification, regression, divergence, representation, adversarial, energy, mixture, weak]

/** Every loss, keyed by its `info.key`. */
export const lossRegistry: Readonly<Record<string, Loss>> = entries('loss', ...modules) as unknown as Readonly<
  Record<string, Loss>
>

/** The losses matching every given filter, in registry order (by family, then definition order). */
export function listLosses(filter: { family?: LossFamily; inputs?: LossInput; note?: string } = {}): Loss[] {
  return Object.values(lossRegistry).filter(
    (l) =>
      (filter.family === undefined || l.info.family === filter.family) &&
      (filter.inputs === undefined || l.info.inputs === filter.inputs) &&
      (filter.note === undefined || (l.info.notes ?? []).includes(filter.note)),
  )
}

/** The loss with this key; throws for an unknown key. */
export function getLoss(key: string): Loss {
  const l = lossRegistry[key]
  if (!l) throw new DomainError('losses', `losses: no loss '${key}'`)
  return l
}
