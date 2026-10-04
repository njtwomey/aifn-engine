/**
 * The registry of `aifn-methods/neural/ode-mixtures`: stochastic vector field mixtures, their losses, realised paths
 * and per-instance work, the streamed runs and the forward-evaluation study, as functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as efficiency from './efficiency'
import * as losses from './losses'
import * as model from './model'
import * as run from './run'
import * as sampling from './sampling'
import * as tasks from './tasks'

const fn = definer<FunctionInfo>('function', 'neural/ode-mixtures')
const NODE = 'neural-ordinary-differential-equations'
const CITE = ['chen2018', 'dupont2019']

fn(
  {
    key: 'svfm',
    name: 'Stochastic vector field mixture',
    summary:
      'A neural ODE whose VF is one of K (stochastic) components, chosen by pick and stick or forward filtering over the integration interval.',
    role: 'construction',
    notes: [NODE, 'gaussian-mixture-model', 'hidden-markov-model', 'log-normal-distribution'],
    cite: [...CITE, 'rabiner1989'],
  },
  model.svfm,
)
fn(
  {
    key: 'svfmObjective',
    name: 'SVFM objective (MDLoss, TLoss, VLoss, FLoss)',
    summary:
      'The predictive loss (mixture density, cross-entropy or squared error; FLoss along paths) plus λ times the transport and variance losses.',
    role: 'construction',
    notes: [NODE, 'gaussian-mixture-model', 'cubic-spline-interpolation'],
    cite: CITE,
  },
  losses.svfmObjective,
)
fn(
  {
    key: 'transportLoss',
    name: 'Transportation loss (TLoss)',
    summary: 'The mean squared distance a state travels per grid interval, π-weighted over components.',
    role: 'property',
    notes: [NODE],
    cite: CITE,
  },
  losses.transportLoss,
)
fn(
  {
    key: 'varianceLoss',
    name: 'Variance loss (VLoss)',
    summary: 'The mean squared deviation of the VF along a path from its average, π-weighted over components.',
    role: 'property',
    notes: [NODE, 'numerical-ode-solvers'],
    cite: [...CITE, 'dormand1980'],
  },
  losses.varianceLoss,
)
fn(
  {
    key: 'mixtureDensityLoss',
    name: 'Mixture density loss (MDLoss) of the state',
    summary: 'The negative log density of targets under the mixture over components at a grid time.',
    role: 'property',
    notes: [NODE, 'gaussian-mixture-model'],
    cite: CITE,
  },
  losses.mixtureDensityLoss,
)
fn(
  {
    key: 'interpolatePaths',
    name: 'Path targets on the grid',
    summary: 'Sampled paths interpolated at the grid times by a not-a-knot cubic spline (the targets of FLoss).',
    role: 'transform',
    notes: ['cubic-spline-interpolation'],
    cite: [],
  },
  losses.interpolatePaths,
)
fn(
  {
    key: 'samplePaths',
    name: 'Realised SVFM paths',
    summary:
      'Paths with frozen randomness per instance: components drawn from π, the stochastic VF sampled once per solve.',
    role: 'simulation',
    random: true,
    notes: [NODE, 'reparameterisation-trick', 'gumbel-max-trick'],
    cite: [...CITE, 'jang2017'],
  },
  sampling.samplePaths,
)
fn(
  {
    key: 'instanceWork',
    name: 'Per-instance function evaluations',
    summary: 'The NFE of each instance solved alone by Dormand–Prince, and of the batch solved as one system.',
    role: 'property',
    notes: [NODE, 'numerical-ode-solvers'],
    cite: [...CITE, 'dormand1980'],
  },
  sampling.instanceWork,
)
fn(
  {
    key: 'svfmRun',
    name: 'Streamed SVFM training run',
    summary:
      'Train an SVFM or a baseline (classification, end targets or FLoss forecasting) and yield realised paths, component posteriors, fields and NFE.',
    role: 'simulation',
    random: true,
    notes: [NODE, 'gaussian-mixture-model'],
    cite: [...CITE, 'kingma2015'],
  },
  run.svfmRun,
)
fn(
  {
    key: 'nfeStudy',
    name: 'Forward-evaluation study',
    summary:
      'Train VF, VF with TVLoss and SVFM models, then count each instance’s function evaluations over tolerances beside its VF variance.',
    role: 'simulation',
    random: true,
    notes: [NODE, 'numerical-ode-solvers'],
    cite: [...CITE, 'dormand1980'],
  },
  efficiency.nfeStudy,
)
fn(
  {
    key: 'walkTask',
    name: 'Walks as a forecasting task',
    summary:
      'Sampled walks with the hour of day as starts, paths at regular times and an optional cyclic time context.',
    role: 'construction',
    notes: [NODE],
    cite: [],
  },
  tasks.walkTask,
)
fn(
  {
    key: 'endpointTask',
    name: 'Start–target pairs as a task',
    summary: '1-d starts and end targets, coloured by the side of the start or of the target.',
    role: 'construction',
    notes: [NODE],
    cite: [],
  },
  tasks.endpointTask,
)
fn(
  {
    key: 'classificationTask',
    name: 'Labelled points as a task',
    summary: 'Points and class labels as the starts and labels of an SVFM classification run.',
    role: 'construction',
    notes: [NODE],
    cite: [],
  },
  tasks.classificationTask,
)

/** The functions of the module, keyed by name. */
export const odeMixtureFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', model, losses, sampling, run, tasks, efficiency) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
