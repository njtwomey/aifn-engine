/**
 * The heat equation $u_t = D u_{xx}$ on an irregular 1-D mesh written as message passing, the computation a graph or
 * mesh neural operator learns. Each interior node $i$ receives from its neighbours $j$ the messages
 * $m_{ij} = w_{ij}(u_j - u_i)$ and updates $u_i \leftarrow u_i + \Delta t\,D \sum_j m_{ij}$ (explicit Euler). With
 * the geometric weights $w_{i,i\pm1} = 2/(h_\pm(h_- + h_+))$, where $h_-$ and $h_+$ are the lengths of the edges to
 * the left and right, the aggregate is the standard second difference on a non-uniform grid (LeVeque, 2007, "Finite
 * Difference Methods for Ordinary and Partial Differential Equations", §1.5). With the uniform weights $1/\bar{h}^2$
 * ($\bar{h}$ the mean edge length) the messages ignore the edge lengths, as a message function without relative
 * positions in its edge features would. The ends are Dirichlet nodes held at 0. The point is the message-passing view
 * and the uniform-weight baseline; `heatEquation` of `aifn-methods/dynamics/pde` solves the uniform-grid problem by the
 * method of lines.
 */

/** How messages are weighted: by the edge lengths (`geometric`), or the same for every edge (`uniform`). */
export type MessageWeights = 'geometric' | 'uniform'

/** Options of `meshHeatRollout`. */
export type MeshHeatOptions = {
  /** The diffusivity $D > 0$. */
  diffusivity: number
  /** The final time. */
  time: number
  /** The number of states kept, evenly spaced in time from 0 to `time` (default 41). */
  frames?: number
  /** How the messages are weighted. */
  weights: MessageWeights
}

/**
 * The two message weights of each interior node, to the left and right neighbour (0 at the ends).
 *
 * @param x The sorted mesh, ends included.
 * @param weights Geometric (by the edge lengths) or uniform weights.
 * @returns The `left` and `right` weight of every node.
 *
 * @example Geometric weights give u'' = 2 for u = x^2 on an irregular mesh
 * const x = [0, 0.1, 0.35, 0.4, 0.7, 1]
 * const { left, right } = messageWeights(x, 'geometric')
 * print('u\'\' at node 2 =', left[2] * (x[1] ** 2 - x[2] ** 2) + right[2] * (x[3] ** 2 - x[2] ** 2))
 */
export function messageWeights(x: readonly number[], weights: MessageWeights): { left: number[]; right: number[] } {
  const n = x.length
  const mean = (x[n - 1] - x[0]) / (n - 1)
  const left = new Array<number>(n).fill(0)
  const right = new Array<number>(n).fill(0)
  for (let i = 1; i < n - 1; i++) {
    const hl = x[i] - x[i - 1]
    const hr = x[i + 1] - x[i]
    if (weights === 'geometric') {
      left[i] = 2 / (hl * (hl + hr))
      right[i] = 2 / (hr * (hl + hr))
    } else {
      left[i] = 1 / mean ** 2
      right[i] = 1 / mean ** 2
    }
  }
  return { left, right }
}

/**
 * The heat equation on the sorted mesh `x` (ends included, held at 0) from the values `u0`, by explicit message
 * passing. The time step is at most $0.4/(D \max_i (w_{i,i-1} + w_{i,i+1}))$, inside the explicit stability limit,
 * and divides the time between frames evenly.
 *
 * @param x The sorted mesh, ends included.
 * @param u0 The initial values at the nodes (the ends are set to 0).
 * @param options The diffusivity, the final time, the number of frames and the message weights.
 * @param options.diffusivity The diffusivity $D > 0$.
 * @param options.time The final time.
 * @param options.frames The number of states kept, evenly spaced in time from 0 to `time`.
 * @param options.weights How the messages are weighted.
 * @returns The kept `states`, one array per frame, with their `times` and the total number of `steps`.
 *
 * @example A sine on an uneven mesh decays as exp(-D pi^2 t)
 * const x = [0, 0.1, 0.25, 0.5, 0.6, 0.8, 1]
 * const options = { diffusivity: 0.1, time: 1, frames: 3, weights: 'geometric' }
 * const run = meshHeatRollout(x, x.map((v) => Math.sin(Math.PI * v)), options)
 * print('u(0.5) at t = 1:', run.states[2][3], ' exact', Math.exp(-0.1 * Math.PI ** 2))
 */
export function meshHeatRollout(
  x: readonly number[],
  u0: readonly number[],
  { diffusivity, time, frames = 41, weights }: MeshHeatOptions,
): { times: number[]; states: number[][]; steps: number } {
  const n = x.length
  const { left, right } = messageWeights(x, weights)
  let largest = 0
  for (let i = 1; i < n - 1; i++) largest = Math.max(largest, left[i] + right[i])
  const perFrame = Math.max(1, Math.ceil(time / (frames - 1) / (0.4 / (diffusivity * largest))))
  const dt = time / (frames - 1) / perFrame
  let u = u0.slice()
  u[0] = 0
  u[n - 1] = 0
  const states = [u.slice()]
  const times = [0]
  for (let f = 1; f < frames; f++) {
    for (let s = 0; s < perFrame; s++) {
      const next = u.slice()
      for (let i = 1; i < n - 1; i++) {
        const aggregate = left[i] * (u[i - 1] - u[i]) + right[i] * (u[i + 1] - u[i])
        next[i] = u[i] + dt * diffusivity * aggregate
      }
      u = next
    }
    states.push(u.slice())
    times.push((f * time) / (frames - 1))
  }
  return { times, states, steps: perFrame * (frames - 1) }
}
