/**
 * The functions and layers of `aifn-compute/nn/experts`, registered with the notes that define them. The layer is a
 * construction; routing is a transform; the auxiliary losses and the statistics are properties of a routing.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as layer from './layer'
import * as losses from './losses'
import * as routing from './routing'

const fn = definer<FunctionInfo>('function', 'nn/experts')
const NOTES = ['mixture-of-experts']

fn(
  {
    key: 'route',
    name: 'Mixture-of-experts routing',
    role: 'transform',
    summary:
      'Combine weights and dispatch masks from router logits: softmax, top-k, noisy top-k, Switch top-1 or expert choice, with capacity and token dropping.',
    notes: NOTES,
    cite: ['shazeer2017', 'fedus2022', 'lepikhin2021', 'jacobs1991'],
  },
  routing.route,
)
fn(
  {
    key: 'denseRouting',
    name: 'Dense routing from probabilities',
    role: 'construction',
    summary: 'A routing in which every expert sees every token, from gate probabilities such as a hierarchical gate’s.',
    notes: NOTES,
    cite: ['jacobs1991'],
  },
  routing.denseRouting,
)
fn(
  {
    key: 'expertCapacity',
    name: 'Expert capacity',
    tex: 'C = \\lceil c\\,Tk/N \\rceil',
    role: 'property',
    summary: 'The assignments each expert may take: the capacity factor times the even share of T·k assignments.',
    notes: NOTES,
    cite: ['fedus2022', 'lepikhin2021'],
  },
  routing.expertCapacity,
)
fn(
  {
    key: 'MixtureOfExperts',
    name: 'Mixture-of-experts layer',
    role: 'construction',
    summary:
      'A router and N expert layers: each token’s output is the weighted sum of the experts the gate chose for it.',
    notes: NOTES,
    cite: ['shazeer2017', 'fedus2022', 'jacobs1991'],
  },
  layer.MixtureOfExperts,
)
fn(
  {
    key: 'loadBalancingLoss',
    name: 'Load-balancing loss',
    tex: 'N\\sum_i f_i P_i',
    role: 'property',
    summary:
      'The Switch auxiliary loss: N times the dot product of each expert’s share of assignments and its mean probability.',
    notes: NOTES,
    cite: ['fedus2022'],
  },
  losses.loadBalancingLoss,
)
fn(
  {
    key: 'importanceLoss',
    name: 'Importance loss',
    tex: '\\mathrm{CV}(\\mathrm{Importance})^2',
    role: 'property',
    summary: 'The squared coefficient of variation of the experts’ total gate weights over a batch.',
    notes: NOTES,
    cite: ['shazeer2017'],
  },
  losses.importanceLoss,
)
fn(
  {
    key: 'routerZLoss',
    name: 'Router z-loss',
    tex: '\\tfrac1T\\sum_t (\\log\\sum_i e^{z_{ti}})^2',
    role: 'property',
    summary: 'The mean squared log-partition of the router’s logits, which keeps them small and the softmax stable.',
    notes: NOTES,
  },
  losses.routerZLoss,
)
fn(
  {
    key: 'routingStatistics',
    name: 'Routing statistics',
    role: 'property',
    summary: 'Load and mean probability per expert, router entropy, idle experts and the share of assignments dropped.',
    notes: NOTES,
  },
  losses.routingStatistics,
)

type Table = Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

/** Every function and layer constructor of the module, keyed by name. */
export const expertFunctions = entries<FunctionInfo>('function', routing, layer, losses) as Table
