/**
 * TrueSkill (Herbrich, Minka and Graepel, 2007): Bayesian skill ratings with a Gaussian belief
 * $\Gauss(\mu, \sigma^2)$ per player, three ways. `trueSkillUpdate` is the closed-form two-player update of one game
 * (Moser, 2010), `trueSkillEp` runs expectation propagation over a fixed set of matches with that update as its
 * site step, and `trueSkillModel` states the same factor graph in the model language, for
 * `modelExpectationPropagation`.
 *
 * In every form a player's performance is their skill plus $\Gauss(0, \beta^2)$ noise, and a game observes the
 * difference $d$ of the two performances through an interval: a win is $d > \varepsilon$ and a draw
 * $\lvert d \rvert < \varepsilon$, with the draw margin $\varepsilon$ from `drawMargin`.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { fromData, tensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { intervalTilted } from 'aifn-compute/inference/expectation-propagation'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { dist, model, type Model } from 'aifn-compute/inference/model'

// ── TrueSkill ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** A skill belief $\Gauss(\mu, \sigma^2)$: `mean` is $\mu$ and `sd` is $\sigma$. */
export interface Rating {
  /** The mean skill $\mu$. */
  mean: number
  /** The standard deviation $\sigma$ of the belief. */
  sd: number
}

/**
 * TrueSkill's published defaults: $\mu_0 = 25$, $\sigma_0 = 25/3$, $\beta = 25/6$, $\tau = 25/300$, draw
 * probability 0.1.
 */
export const TRUESKILL_DEFAULTS = { mean: 25, sd: 25 / 3, beta: 25 / 6, tau: 25 / 300, drawProbability: 0.1 } as const

/**
 * The draw margin $\varepsilon$ for a draw probability $p$ between two equally skilled players:
 * $p = 2\Phi(\varepsilon / (\sqrt{n}\,\beta)) - 1$, so $\varepsilon = \Phi^{-1}((p + 1)/2) \sqrt{n}\,\beta$ with $n$
 * players in the game.
 *
 * @param drawProbability The probability $p$ of a draw between equal players, in $[0, 1)$.
 * @param beta The performance noise $\beta$.
 * @param players The number $n$ of players in the game, all teams together.
 * @returns The margin $\varepsilon$, in skill units.
 *
 * @example TrueSkill's default margin
 * print('epsilon:', drawMargin(0.1, 25 / 6))
 * print('more draws, wider margin:', drawMargin(0.3, 25 / 6))
 */
export function drawMargin(drawProbability: number, beta: number, players = 2): number {
  return (normalQuantile((drawProbability + 1) / 2) as number) * Math.sqrt(players) * beta
}

/** Options of a TrueSkill update. */
export interface TrueSkillOptions {
  /** The performance noise $\beta$ (default 25/6). */
  beta?: number
  /**
   * The dynamics noise $\tau$, whose square is added to each variance before the game (default 25/300; 0 in EP over a
   * fixed match set).
   */
  tau?: number
  /** The draw margin $\varepsilon$ (default `drawMargin` of the draw probability 0.1 and $\beta$). */
  drawMargin?: number
}

/** The result of one two-player TrueSkill update, with the intermediate quantities. */
export interface TrueSkillUpdate {
  /** Player 1's rating after the game. */
  player1: Rating
  /** Player 2's rating after the game. */
  player2: Rating
  /** $c = \sqrt{2\beta^2 + \sigma_1^2 + \sigma_2^2}$, with the variances after $\tau$: the spread of $d$. */
  c: number
  /** $t = \mu_1 - \mu_2$, the mean of $d$ before the game. */
  t: number
  /** The mean correction $v$ of TrueSkill's notation. */
  v: number
  /** The variance correction $w$ of TrueSkill's notation. */
  w: number
  /** The probability of the observed outcome under the prior. */
  probability: number
}

/**
 * One TrueSkill update of a two-player game from player 1's side (Herbrich, Minka and Graepel, 2007; Moser, 2010).
 * With $\sigma_i^2 \gets \sigma_i^2 + \tau^2$, the performance difference $d = p_1 - p_2$ has the cavity
 * $\Gauss(t, c^2)$, $t = \mu_1 - \mu_2$, $c^2 = 2\beta^2 + \sigma_1^2 + \sigma_2^2$. The outcome is an interval factor
 * on $d$: a win is $d > \varepsilon$, a loss $d < -\varepsilon$, a draw $\lvert d \rvert < \varepsilon$. Its
 * truncated-normal moments $(\hat{m}, \hat{v})$ (`intervalTilted`) give
 * $\mu_1 \gets \mu_1 + \sigma_1^2(\hat{m} - t)/c^2$, $\mu_2 \gets \mu_2 - \sigma_2^2(\hat{m} - t)/c^2$ and
 * $\sigma_i^2 \gets \sigma_i^2 + \sigma_i^4(\hat{v} - c^2)/c^4$. In TrueSkill's notation
 * $v = \pm(\hat{m} - t)/c$ (the sign of the outcome; $+$ for a draw) and $w = 1 - \hat{v}/c^2$, which are
 * $v_{\text{win}}, w_{\text{win}}$ of $(\pm t - \varepsilon)/c$ and $v_{\text{draw}}, w_{\text{draw}}$ of
 * $(t/c, \varepsilon/c)$.
 *
 * @param r1 Player 1's rating before the game (before the dynamics noise is added).
 * @param r2 Player 2's rating before the game.
 * @param outcome The result from player 1's side.
 * @param options The performance noise $\beta$, the dynamics noise $\tau$ and the draw margin $\varepsilon$.
 * @returns Both new ratings, with $c$, $t$, $v$, $w$ and the prior probability of the outcome.
 *
 * @example One game between two new players
 * const fresh = { mean: 25, sd: 25 / 3 }
 * const win = trueSkillUpdate(fresh, fresh, 'win')
 * print('winner:', win.player1)
 * print('loser:', win.player2)
 * print('P(win) before the game:', win.probability)
 *
 * @example A draw between unequal players narrows the gap
 * const draw = trueSkillUpdate({ mean: 30, sd: 4 }, { mean: 22, sd: 6 }, 'draw')
 * print('stronger:', draw.player1)
 * print('weaker:', draw.player2)
 * print('P(draw) before the game:', draw.probability)
 */
export function trueSkillUpdate(
  r1: Rating,
  r2: Rating,
  outcome: 'win' | 'loss' | 'draw',
  options: TrueSkillOptions = {},
): TrueSkillUpdate {
  const beta = options.beta ?? TRUESKILL_DEFAULTS.beta
  const tau = options.tau ?? TRUESKILL_DEFAULTS.tau
  const eps = options.drawMargin ?? drawMargin(TRUESKILL_DEFAULTS.drawProbability, beta)
  const var1 = r1.sd ** 2 + tau ** 2
  const var2 = r2.sd ** 2 + tau ** 2
  const c2 = 2 * beta * beta + var1 + var2
  const c = Math.sqrt(c2)
  const t = r1.mean - r2.mean
  const [lower, upper] = outcome === 'win' ? [eps, Infinity] : outcome === 'loss' ? [-Infinity, -eps] : [-eps, eps]
  const d = intervalTilted(t, c2, lower, upper)
  const shift = (d.mean - t) / c2
  const shrink = (d.variance - c2) / (c2 * c2)
  return {
    player1: { mean: r1.mean + var1 * shift, sd: Math.sqrt(var1 + var1 * var1 * shrink) },
    player2: { mean: r2.mean - var2 * shift, sd: Math.sqrt(var2 + var2 * var2 * shrink) },
    c,
    t,
    v: ((outcome === 'loss' ? -1 : 1) * (d.mean - t)) / c,
    w: 1 - d.variance / c2,
    probability: Math.exp(d.logZ),
  }
}

/**
 * TrueSkill in the model language (Herbrich, Minka and Graepel, 2007, fig. 1): a plate of $P$ players with skills
 * $s_p \sim \Gauss(\mu_0, \sigma_0^2)$, and a plate of $G$ games, each naming its first player $w_g$, second player
 * $l_g$ and whether it was a draw $\delta_g \in \{0, 1\}$ (per-game constants). The performances are
 * $p_g \sim \Gauss(s_{w_g}, \beta^2)$ and $q_g \sim \Gauss(s_{l_g}, \beta^2)$, their difference $d_g = p_g - q_g$,
 * and the outcome is an interval on $d_g$: the first player won when $d_g > \varepsilon$, and the game was drawn when
 * $-\varepsilon < d_g < \varepsilon$ (bounds chosen by $\delta_g$ with `index`). Observing
 * $y_g = 1 \sim \Bern(\indicator(\text{lower} < d_g < \text{upper}))$ states the interval, so
 * `modelExpectationPropagation` runs TrueSkill's message passing on this graph, and agrees with `trueSkillEp`. The
 * draw margin $\varepsilon$ defaults to 0 here (the win-only model of the original figure). Bind the sizes `P` and
 * `G`, the constants `w`, `l` and `δ` (one entry per game) and the data `y` (all 1).
 *
 * @param options The prior `mean` $\mu_0$ and `sd` $\sigma_0$ of every skill (default 25 and 25/3), the performance
 *   noise `beta` (default 25/6) and the `drawMargin` $\varepsilon$ (default 0).
 * @returns The model, with the plates `players` and `games`.
 *
 * @example The nodes of the model
 * const m = trueSkillModel()
 * print('plates:', m.groups.map((g) => g.name))
 * print('nodes:', m.attributes.map((n) => `${n.name} (${n.role})`))
 */
export function trueSkillModel(
  options: { mean?: number; sd?: number; beta?: number; drawMargin?: number } = {},
): Model {
  const {
    mean = TRUESKILL_DEFAULTS.mean,
    sd = TRUESKILL_DEFAULTS.sd,
    beta = TRUESKILL_DEFAULTS.beta,
    drawMargin: eps = 0,
  } = options
  return model('TrueSkill', (m) => {
    const players = m.plate('players', 'P', { label: 'P', index: 'p' })
    const games = m.plate('games', 'G', { label: 'G', index: 'g' })
    const skill = players.variable('s', dist.Normal(mean, sd), { label: 's_p' })
    const winner = games.constant('w', undefined, { label: 'w_g' })
    const loser = games.constant('l', undefined, { label: 'l_g' })
    const drawn = games.constant('δ', undefined, { label: '\\delta_g' })
    const p = games.variable('p', dist.Normal(skill.at(winner), beta), { label: 'p_g' })
    const q = games.variable('q', dist.Normal(skill.at(loser), beta), { label: 'q_g' })
    const d = games.deterministic('d', 'difference', [p, q], { label: 'd_g' })
    const lower = games.deterministic('lower', 'index', [tensor([eps, -eps]), drawn])
    const upper = games.deterministic('upper', 'index', [tensor([Infinity, eps]), drawn])
    const inside = games.deterministic('inside', 'interval', [d, lower, upper], {
      label: '\\mathbb{1}_g',
    })
    games.observed('y', dist.Bernoulli(inside), { label: 'y_g' })
  })
}

/** A match between two players: the first won, or it was a draw. */
export interface Match {
  /** The first player's index: the winner, or either player of a draw. */
  winner: number
  /** The second player's index. */
  loser: number
  /** Whether the match was drawn (default false). */
  draw?: boolean
}

/** Options of {@link trueSkillEp}. */
export interface TrueSkillEpOptions {
  /** Prior ratings, one per player. */
  players: readonly Rating[]
  /** The matches, each between two players indexed into `players`, all treated as simultaneous. */
  matches: readonly Match[]
  /** The performance noise $\beta$ (default 25/6). */
  beta?: number
  /** The draw margin $\varepsilon$ (default `drawMargin` of the draw probability 0.1 and $\beta$). */
  drawMargin?: number
  /** Weight of the old site, in $[0, 1)$. Default 0. */
  damping?: number
  /** A sweep that changes no site parameter by this much or more has converged (default 1e-8). */
  tolerance?: number
}

/** The state of EP over a fixed set of matches (skills constant over the matches, as in TrueSkill through time). */
export interface TrueSkillEpState extends Status {
  /** Match updates done. */
  t: number
  /**
   * Site precisions per match, $M \times 2$ (column 0 on the first-listed player, 1 on the second): what each match
   * adds to its players' precisions.
   */
  sitePrecision: Tensor
  /** Site shifts (precision times mean) per match, $M \times 2$, in the layout of `sitePrecision`. */
  siteShift: Tensor
  /** Posterior means per player. */
  means: Tensor
  /** Posterior standard deviations per player. */
  sds: Tensor
  /** The match updated last ($-1$ at the start). */
  match: number
  /** The details of the last update, or null at the start and when the update was skipped. */
  update: TrueSkillUpdate | null
  /** False when the last update was skipped (a cavity precision not positive) or gave a non-finite site. */
  ok: boolean
  /** Completed sweeps over the matches. */
  sweep: number
  /** The match the next step updates. */
  position: number
  /** The largest change of a site parameter in the last step. */
  change: number
  /** The largest change so far in the current sweep. */
  sweepChange: number
  /** The largest change in the last completed sweep (infinity before the first). */
  lastSweepChange: number
  /** Whether a whole sweep changed no site parameter by `tolerance` or more (true at once with no matches). */
  converged: boolean
}

/**
 * The players' posteriors in natural parameters: the prior precision and shift plus every match's sites.
 *
 * @param players The prior ratings, one per player.
 * @param matches The matches whose sites are added.
 * @param tau The site precisions, two per match (entry $2k$ on match $k$'s first player, $2k + 1$ on its second).
 * @param nu The site shifts (precision times mean), in the layout of `tau`.
 * @returns The precision `P` and the shift `N` of each player's posterior.
 */
function marginals(players: readonly Rating[], matches: readonly Match[], tau: Float64Array, nu: Float64Array) {
  const P = players.map((r) => 1 / r.sd ** 2)
  const N = players.map((r) => r.mean / r.sd ** 2)
  matches.forEach((m, k) => {
    P[m.winner] += tau[2 * k]
    N[m.winner] += nu[2 * k]
    P[m.loser] += tau[2 * k + 1]
    N[m.loser] += nu[2 * k + 1]
  })
  return { P, N }
}

/**
 * EP for TrueSkill over a fixed set of matches: one match per step. The cavity of each player removes the match's
 * site; the match factor is applied by the two-player update (with $\tau = 0$); the new sites are the updated ratings
 * divided by the cavities. The first sweep is TrueSkill's online (ADF) pass without dynamics; later sweeps let early
 * matches learn from later ones (Herbrich, Minka and Graepel, 2007; Dangauthier et al., 2008). A match whose cavity
 * is not a proper Gaussian is skipped and flagged with `ok`. No start: run it with `run(alg, undefined, steps)`, and it
 * stops once a sweep has converged.
 *
 * @param o The prior ratings, the matches, $\beta$, the draw margin, the damping and the tolerance.
 * @returns The algorithm, whose state holds the sites and the posterior means and standard deviations.
 *
 * @example A cycle of three games, to convergence
 * // 0 beats 1, 1 beats 2, and 2 draws with 0.
 * const fresh = { mean: 25, sd: 25 / 3 }
 * const matches = [{ winner: 0, loser: 1 }, { winner: 1, loser: 2 }, { winner: 2, loser: 0, draw: true }]
 * const s = run(trueSkillEp({ players: [fresh, fresh, fresh], matches }), undefined, 1000)
 * print('sweeps:', s.sweep, 'converged:', s.converged)
 * print('means:', s.means)
 * print('sds:', s.sds)
 *
 * @example Later sweeps revise the online pass
 * const fresh = { mean: 25, sd: 25 / 3 }
 * const matches = [{ winner: 0, loser: 1 }, { winner: 1, loser: 2 }]
 * const alg = trueSkillEp({ players: [fresh, fresh, fresh], matches })
 * print('after one sweep (online TrueSkill):', run(alg, undefined, 2).means)
 * print('at convergence:', run(alg, undefined, 1000).means)
 */
export function trueSkillEp(o: TrueSkillEpOptions): Algorithm<void, TrueSkillEpState> {
  const { players, matches } = o
  const M = matches.length
  const beta = o.beta ?? TRUESKILL_DEFAULTS.beta
  const margin = o.drawMargin ?? drawMargin(TRUESKILL_DEFAULTS.drawProbability, beta)
  const damping = o.damping ?? 0
  const tolerance = o.tolerance ?? 1e-8
  return {
    name: 'ep.trueskill',
    init: () => {
      const tau = new Float64Array(2 * M)
      const { P, N } = marginals(players, matches, tau, tau)
      return {
        t: 0,
        sitePrecision: fromData(new Float64Array(2 * M), [M, 2]),
        siteShift: fromData(new Float64Array(2 * M), [M, 2]),
        means: fromData(
          Float64Array.from(P, (p, i) => N[i] / p),
          [P.length],
        ),
        sds: fromData(
          Float64Array.from(P, (p) => Math.sqrt(1 / p)),
          [P.length],
        ),
        match: -1,
        update: null,
        ok: true,
        sweep: 0,
        position: 0,
        change: 0,
        sweepChange: 0,
        lastSweepChange: Infinity,
        converged: M === 0,
      }
    },
    step: (s) => {
      if (s.converged) return { ...s, t: s.t + 1 }
      const k = s.position
      const m = matches[k]
      const tau = Float64Array.from(s.sitePrecision.data)
      const nu = Float64Array.from(s.siteShift.data)
      const { P, N } = marginals(players, matches, tau, nu)
      const cav = [
        { p: P[m.winner] - tau[2 * k], n: N[m.winner] - nu[2 * k] },
        { p: P[m.loser] - tau[2 * k + 1], n: N[m.loser] - nu[2 * k + 1] },
      ]
      let ok = cav.every((c) => c.p > 0)
      let change = 0
      let update: TrueSkillUpdate | null = null
      if (ok) {
        update = trueSkillUpdate(
          { mean: cav[0].n / cav[0].p, sd: Math.sqrt(1 / cav[0].p) },
          { mean: cav[1].n / cav[1].p, sd: Math.sqrt(1 / cav[1].p) },
          m.draw ? 'draw' : 'win',
          { beta, tau: 0, drawMargin: margin },
        )
        const after = [update.player1, update.player2]
        after.forEach((r, j) => {
          const nt = 1 / r.sd ** 2 - cav[j].p
          const nn = r.mean / r.sd ** 2 - cav[j].n
          const dt = (1 - damping) * nt + damping * tau[2 * k + j]
          const dn = (1 - damping) * nn + damping * nu[2 * k + j]
          change = Math.max(change, Math.abs(dt - tau[2 * k + j]), Math.abs(dn - nu[2 * k + j]))
          tau[2 * k + j] = dt
          nu[2 * k + j] = dn
        })
        ok = Number.isFinite(change)
      }
      const mg = marginals(players, matches, tau, nu)
      let position = k + 1
      let { sweep, sweepChange, lastSweepChange } = s
      let converged: boolean = s.converged
      sweepChange = Math.max(sweepChange, change)
      if (position >= matches.length) {
        converged = sweepChange < tolerance
        lastSweepChange = sweepChange
        sweepChange = 0
        position = 0
        sweep++
      }
      return {
        ...s,
        t: s.t + 1,
        sitePrecision: fromData(tau, s.sitePrecision.shape),
        siteShift: fromData(nu, s.siteShift.shape),
        means: fromData(
          Float64Array.from(mg.P, (p, i) => mg.N[i] / p),
          [mg.P.length],
        ),
        sds: fromData(
          Float64Array.from(mg.P, (p) => Math.sqrt(1 / p)),
          [mg.P.length],
        ),
        match: k,
        update,
        ok,
        sweep,
        position,
        change,
        sweepChange,
        lastSweepChange,
        converged,
      }
    },
  }
}
