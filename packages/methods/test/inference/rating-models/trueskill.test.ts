import { describe, expect, it } from 'vitest'
import { drawMargin, trueSkillEp, trueSkillModel, trueSkillUpdate } from 'aifn-methods/inference/rating-models'
import { logJoint } from 'aifn-compute/inference/model'
import { modelExpectationPropagation } from 'aifn-compute/inference/expectation-propagation'
import { Normal } from 'aifn-compute/probability/distributions'
import { normalCdf, normalPdf, truncatedNormalVDraw, truncatedNormalWDraw } from 'aifn-compute/numerics/special'
import { run } from 'aifn-compute/foundation/trace'

describe('TrueSkill', () => {
  it('reproduces the published two-player updates (trueskill.org defaults)', () => {
    const r = { mean: 25, sd: 25 / 3 }
    const win = trueSkillUpdate(r, r, 'win')
    expect(win.player1.mean).toBeCloseTo(29.396, 3)
    expect(win.player1.sd).toBeCloseTo(7.171, 3)
    expect(win.player2.mean).toBeCloseTo(20.604, 3)
    const draw = trueSkillUpdate(r, r, 'draw')
    expect(draw.player1.mean).toBeCloseTo(25, 10)
    expect(draw.player1.sd).toBeCloseTo(6.458, 3)
  })
  it('matches the formulas v = φ/Φ, w = v(v + t) written out', () => {
    const r1 = { mean: 30, sd: 4 }
    const r2 = { mean: 22, sd: 6 }
    const beta = 4
    const eps = drawMargin(0.1, beta)
    const u = trueSkillUpdate(r1, r2, 'loss', { beta, tau: 0, drawMargin: eps })
    const c = Math.sqrt(2 * beta * beta + 16 + 36)
    const t = -(30 - 22) / c - eps / c
    const v = (normalPdf(t) as number) / (normalCdf(t) as number)
    const w = v * (v + t)
    expect(u.player1.mean).toBeCloseTo(30 - (16 * v) / c, 12)
    expect(u.player2.mean).toBeCloseTo(22 + (36 * v) / c, 12)
    expect(u.player1.sd).toBeCloseTo(Math.sqrt(16 * (1 - (16 / (c * c)) * w)), 12)
    expect(u.probability).toBeCloseTo(normalCdf(t) as number, 12)
  })
  it('EP on one match is the one-shot update; on a match set it converges', () => {
    const p = { mean: 25, sd: 25 / 3 }
    const one = run(trueSkillEp({ players: [p, p], matches: [{ winner: 0, loser: 1 }] }), undefined, 10)
    const direct = trueSkillUpdate(p, p, 'win', { tau: 0 })
    expect(one.means.data[0]).toBeCloseTo(direct.player1.mean, 10)
    expect(one.sds.data[1]).toBeCloseTo(direct.player2.sd, 10)
    const matches = [
      { winner: 0, loser: 1 },
      { winner: 1, loser: 2 },
      { winner: 0, loser: 2 },
      { winner: 2, loser: 1, draw: true },
    ]
    const s = run(trueSkillEp({ players: [p, p, p], matches }), undefined, 1000)
    expect(s.converged).toBe(true)
    expect(s.means.data[0]).toBeGreaterThan(s.means.data[1])
  })
  it('a draw through the interval factor is v_draw, w_draw written out', () => {
    const r1 = { mean: 31, sd: 5 }
    const r2 = { mean: 24, sd: 3 }
    const beta = 4
    const eps = drawMargin(0.15, beta)
    const u = trueSkillUpdate(r1, r2, 'draw', { beta, tau: 0, drawMargin: eps })
    const c = Math.sqrt(2 * beta * beta + 25 + 9)
    const v = truncatedNormalVDraw(7 / c, eps / c) as number
    const w = truncatedNormalWDraw(7 / c, eps / c) as number
    expect(u.v).toBeCloseTo(v, 12)
    expect(u.w).toBeCloseTo(w, 12)
    expect(u.player1.mean).toBeCloseTo(31 + (25 * v) / c, 12)
    expect(u.player2.sd).toBeCloseTo(Math.sqrt(9 * (1 - (9 / (c * c)) * w)), 12)
    expect(u.probability).toBeCloseTo((normalCdf((eps - 7) / c) as number) - (normalCdf((-eps - 7) / c) as number), 12)
  })
  it('the model language states wins and draws: performances and an interval on their difference', () => {
    const beta = 4
    const eps = drawMargin(0.2, beta)
    const m = trueSkillModel({ mean: 25, sd: 25 / 3, beta, drawMargin: eps })
    const skills = [27, 22.5, 24]
    const games = { w: [0, 1, 0], l: [1, 2, 2], δ: [0, 1, 0] }
    const perf = { p: [28, 23, 27], q: [21, 22.1, 24.9] }
    const bindings = { sizes: { P: 3, G: 3 }, constants: games, data: { y: [1, 1, 1] } }
    const values: Record<string, number> = {}
    skills.forEach((s, i) => (values[`s[${i}]`] = s))
    for (let g = 0; g < 3; g++) {
      values[`p[${g}]`] = perf.p[g]
      values[`q[${g}]`] = perf.q[g]
    }
    let want = skills.reduce((a, s) => a + (Normal(25, 25 / 3).logProb(s) as number), 0)
    for (let g = 0; g < 3; g++) {
      want += Normal(skills[games.w[g]], beta).logProb(perf.p[g]) as number
      want += Normal(skills[games.l[g]], beta).logProb(perf.q[g]) as number
    }
    // The differences 7 and 2.1 exceed ε (wins) and |0.9| < ε (the draw): every indicator is 1.
    expect(eps).toBeGreaterThan(0.9)
    expect(eps).toBeLessThan(2.1)
    expect(logJoint(m, values, bindings)).toBeCloseTo(want, 12)
    // A win by less than the margin has probability 0.
    expect(logJoint(m, { ...values, 'p[2]': 25.5 }, bindings)).toBe(-Infinity)
  })
  it('EP over the model is trueSkillEp: the same ratings on a match set with draws', () => {
    const beta = 25 / 6
    const eps = drawMargin(0.1, beta)
    const p = { mean: 25, sd: 25 / 3 }
    const matches = [
      { winner: 0, loser: 1 },
      { winner: 1, loser: 2 },
      { winner: 0, loser: 2 },
      { winner: 2, loser: 1, draw: true },
      { winner: 3, loser: 0, draw: true },
      { winner: 3, loser: 2 },
    ]
    const direct = run(
      trueSkillEp({ players: [p, p, p, p], matches, beta, drawMargin: eps, tolerance: 1e-13 }),
      undefined,
      100_000,
    )
    expect(direct.converged).toBe(true)
    const bindings = {
      sizes: { P: 4, G: matches.length },
      constants: {
        w: matches.map((m) => m.winner),
        l: matches.map((m) => m.loser),
        δ: matches.map((m) => (m.draw ? 1 : 0)),
      },
      data: { y: matches.map(() => 1) },
    }
    const m = trueSkillModel({ mean: 25, sd: 25 / 3, beta, drawMargin: eps })
    const ep = run(modelExpectationPropagation(m, bindings, { tolerance: 1e-13 }), undefined, 10_000)
    expect(ep.converged).toBe(true)
    for (let i = 0; i < 4; i++) {
      const k = ep.keys.indexOf(`s[${i}]`)
      expect(ep.means.data[k]).toBeCloseTo(direct.means.data[i], 8)
      expect(Math.sqrt(ep.variances.data[k])).toBeCloseTo(direct.sds.data[i], 8)
    }
    // One sweep over one match is the one-shot update.
    const one = run(
      modelExpectationPropagation(m, {
        sizes: { P: 2, G: 1 },
        constants: { w: [0], l: [1], δ: [1] },
        data: { y: [1] },
      }),
      undefined,
      1,
    )
    const u = trueSkillUpdate(p, { mean: 25, sd: 25 / 3 }, 'draw', { beta, tau: 0, drawMargin: eps })
    expect(one.means.data[one.keys.indexOf('s[0]')]).toBeCloseTo(u.player1.mean, 10)
    expect(Math.sqrt(one.variances.data[one.keys.indexOf('s[1]')])).toBeCloseTo(u.player2.sd, 10)
  })
})
