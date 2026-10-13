/**
 * Thresholding rules: the solutions of the univariate penalised least-squares problem
 *
 * $$\hat\theta(z) = \argmin_\theta \tfrac12 (z - \theta)^2 + p_\lambda(\theta),$$
 *
 * applied elementwise, which is the penalised estimate under an orthonormal design with unit noise variance (and the
 * prox of $p_\lambda$ at step 1). Soft thresholding is the lasso's, hard thresholding the $\ell_0$ penalty's
 * $p_\lambda(\theta) = \tfrac12\lambda^2 [\theta \ne 0]$; SCAD (Fan and Li, 2001, "Variable selection via nonconcave
 * penalized likelihood and its oracle properties", JASA 96(456)) and MCP (Zhang, 2010, "Nearly unbiased variable
 * selection under minimax concave penalty", Ann. Statist. 38(2)) bend the lasso penalty flat beyond $\gamma\lambda$, so
 * large values are left unshrunk while small ones are still set to zero. MCP's rule is the firm shrinkage of Gao and
 * Bruce (1997) with thresholds $\lambda$ and $\gamma\lambda$.
 *
 * Soft thresholding is also `proxL1`; `hardThreshold` of `aifn-compute/signal/sparse` keeps a given number of the
 * largest entries rather than those above a level.
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'

/** The rules of `thresholding`. */
export type ThresholdRule = 'soft' | 'hard' | 'scad' | 'mcp'

/** Options of `thresholding` and `thresholdingPenalty`. */
export type ThresholdOptions = {
  /** The rule (default `'soft'`). */
  rule?: ThresholdRule
  /** The threshold $\lambda \ge 0$. */
  lambda: number
  /**
   * The concavity $\gamma$ of SCAD (above 2, default 3.7 as Fan and Li suggest) or MCP (above 1, default 3); ignored
   * by the soft and hard rules. Larger values are closer to the lasso.
   */
  gamma?: number
}

/**
 * Check the options and fill in $\gamma$.
 *
 * @param options The options as given.
 * @param where The caller's name for error messages.
 * @returns The rule, $\lambda$ and $\gamma$.
 */
function parse(options: ThresholdOptions, where: string): { rule: ThresholdRule; lambda: number; gamma: number } {
  const { rule = 'soft', lambda } = options
  if (!(lambda >= 0 && Number.isFinite(lambda)))
    throw new DomainError(where, `${where}: lambda must be finite and non-negative, got ${lambda}`)
  if (!['soft', 'hard', 'scad', 'mcp'].includes(rule)) throw new DomainError(where, `${where}: unknown rule ${rule}`)
  const gamma = options.gamma ?? (rule === 'scad' ? 3.7 : 3)
  if (rule === 'scad' && !(gamma > 2)) throw new DomainError(where, `${where}: SCAD needs gamma > 2, got ${gamma}`)
  if (rule === 'mcp' && !(gamma > 1)) throw new DomainError(where, `${where}: MCP needs gamma > 1, got ${gamma}`)
  return { rule, lambda, gamma }
}

/**
 * Elementwise thresholding: each $z_i$ mapped to $\argmin_\theta \tfrac12 (z_i - \theta)^2 + p_\lambda(\theta)$.
 * With $a = \lvert z \rvert$:
 *
 * - soft: $\sgn(z)\max(a - \lambda, 0)$;
 * - hard: $z$ when $a > \lambda$, else 0;
 * - SCAD: soft for $a \le 2\lambda$, $((\gamma - 1)z - \sgn(z)\gamma\lambda)/(\gamma - 2)$ for
 *   $2\lambda < a \le \gamma\lambda$, and $z$ beyond;
 * - MCP: $\sgn(z)\max(a - \lambda, 0)/(1 - 1/\gamma)$ for $a \le \gamma\lambda$, and $z$ beyond.
 *
 * Soft, SCAD and MCP are continuous; hard jumps at $\lambda$ (where 0 and $z$ tie, and 0 is returned). Throws
 * `DomainError` for a negative $\lambda$, an unknown rule, or $\gamma$ out of range ($\gamma > 2$ for SCAD,
 * $\gamma > 1$ for MCP).
 *
 * @param z The values $z_i$ to threshold. Not modified.
 * @param options The rule, threshold $\lambda$ and concavity $\gamma$, as `ThresholdOptions`.
 * @returns The thresholded values, one per entry of `z`.
 *
 * @example The four rules on the same values
 * const z = tensor([0.5, 1.5, 2.5, 4, 6])
 * for (const rule of ['soft', 'hard', 'scad', 'mcp']) print(rule, thresholding(z, { rule, lambda: 1 }))
 */
export function thresholding(z: VectorLike, options: ThresholdOptions): Tensor {
  const { rule, lambda, gamma } = parse(options, 'thresholding')
  const rows = dense.toF64(z, 'thresholding')
  return dense.vec(
    rows.map((zi) => {
      const a = Math.abs(zi)
      const sign = Math.sign(zi)
      switch (rule) {
        case 'soft':
          return sign * Math.max(a - lambda, 0)
        case 'hard':
          return a > lambda ? zi : 0
        case 'scad':
          if (a <= 2 * lambda) return sign * Math.max(a - lambda, 0)
          if (a <= gamma * lambda) return ((gamma - 1) * zi - sign * gamma * lambda) / (gamma - 2)
          return zi
        case 'mcp':
          return a <= gamma * lambda ? (sign * Math.max(a - lambda, 0)) / (1 - 1 / gamma) : zi
      }
    }),
  )
}

/**
 * The penalty $p_\lambda(\theta)$ whose univariate problem `thresholding` solves, elementwise. With
 * $a = \lvert \theta \rvert$:
 *
 * - soft (lasso): $\lambda a$;
 * - hard ($\ell_0$): $\tfrac12\lambda^2$ for $\theta \ne 0$, else 0;
 * - SCAD: $\lambda a$ for $a \le \lambda$, $(2\gamma\lambda a - a^2 - \lambda^2)/(2(\gamma - 1))$ for
 *   $\lambda < a \le \gamma\lambda$, and $\lambda^2(\gamma + 1)/2$ beyond;
 * - MCP: $\lambda a - a^2/(2\gamma)$ for $a \le \gamma\lambda$, and $\gamma\lambda^2/2$ beyond.
 *
 * SCAD and MCP agree with the lasso near 0 and are constant beyond $\gamma\lambda$. Throws as `thresholding`.
 *
 * @param theta The values $\theta_i$ at which to evaluate the penalty. Not modified.
 * @param options The rule, $\lambda$ and $\gamma$, as `ThresholdOptions`.
 * @returns The penalty of each entry.
 *
 * @example Penalty curves: the lasso keeps rising, SCAD and MCP level off
 * const theta = tensor([0, 1, 2, 3, 4, 5])
 * for (const rule of ['soft', 'hard', 'scad', 'mcp']) print(rule, thresholdingPenalty(theta, { rule, lambda: 1 }))
 */
export function thresholdingPenalty(theta: VectorLike, options: ThresholdOptions): Tensor {
  const { rule, lambda, gamma } = parse(options, 'thresholdingPenalty')
  return dense.vec(
    dense.toF64(theta, 'thresholdingPenalty').map((t) => {
      const a = Math.abs(t)
      switch (rule) {
        case 'soft':
          return lambda * a
        case 'hard':
          return a === 0 ? 0 : 0.5 * lambda * lambda
        case 'scad':
          if (a <= lambda) return lambda * a
          if (a <= gamma * lambda) return (2 * gamma * lambda * a - a * a - lambda * lambda) / (2 * (gamma - 1))
          return (lambda * lambda * (gamma + 1)) / 2
        case 'mcp':
          return a <= gamma * lambda ? lambda * a - (a * a) / (2 * gamma) : (gamma * lambda * lambda) / 2
      }
    }),
  )
}
