/**
 * Elliptic integrals and Jacobi elliptic functions, with the parameter m = k² as in scipy.special.
 *
 * Method: Carlson's symmetric integrals R_F and R_D by the duplication theorem (Carlson, 1995, "Numerical computation
 * of real or complex elliptic integrals", Numer. Algorithms 10), which give K(m) = R_F(0, 1 − m, 1),
 * E(m) = R_F(0, 1 − m, 1) − (m/3) R_D(0, 1 − m, 1) and the incomplete F(φ | m) = sin φ R_F(cos²φ, 1 − m sin²φ, 1),
 * accurate near m = 1 when the complement 1 − m is passed directly (`ellipkm1`); and the Jacobi functions sn, cn, dn
 * by the descending Landen (arithmetic–geometric mean) transformation (Abramowitz and Stegun, 1964, 16.4), as Cephes'
 * `ellpj`, which scipy wraps.
 */

/** Carlson's R_F(x, y, z) = ½ ∫₀^∞ dt / √((t + x)(t + y)(t + z)) for x, y, z ≥ 0 with at most one zero. */
export function carlsonRF(x: number, y: number, z: number): number {
  if (x < 0 || y < 0 || z < 0 || Number.isNaN(x + y + z)) return NaN
  for (let i = 0; i < 100; i++) {
    const mu = (x + y + z) / 3
    const dx = 1 - x / mu
    const dy = 1 - y / mu
    const dz = 1 - z / mu
    if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) < 1e-4) {
      // Series to fifth order: error below 1e-20 at this tolerance.
      const e2 = dx * dy - dz * dz
      const e3 = dx * dy * dz
      return (1 - e2 / 10 + e3 / 14 + (e2 * e2) / 24 - (3 * e2 * e3) / 44) / Math.sqrt(mu)
    }
    const [sx, sy, sz] = [Math.sqrt(x), Math.sqrt(y), Math.sqrt(z)]
    const lambda = sx * (sy + sz) + sy * sz
    x = (x + lambda) / 4
    y = (y + lambda) / 4
    z = (z + lambda) / 4
  }
  return NaN
}

/** Carlson's R_D(x, y, z) = (3/2) ∫₀^∞ dt / ((t + z) √((t + x)(t + y)(t + z))), x, y ≥ 0 (not both 0), z > 0. */
export function carlsonRD(x: number, y: number, z: number): number {
  if (x < 0 || y < 0 || !(z > 0)) return NaN
  let sum = 0
  let factor = 1
  for (let i = 0; i < 100; i++) {
    const mu = (x + y + 3 * z) / 5
    const dx = 1 - x / mu
    const dy = 1 - y / mu
    const dz = 1 - z / mu
    if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) < 1e-4) {
      const ea = dx * dy
      const eb = dz * dz
      const ec = ea - eb
      const ed = ea - 6 * eb
      const ee = ed + ec + ec
      const s =
        1 +
        ed * (-3 / 14 + (9 / 88) * ed - (9 / 52) * dz * ee) +
        dz * ((1 / 6) * ee + dz * (-(9 / 22) * ec + dz * (3 / 26) * ea))
      return 3 * sum + (factor * s) / (mu * Math.sqrt(mu))
    }
    const [sx, sy, sz] = [Math.sqrt(x), Math.sqrt(y), Math.sqrt(z)]
    const lambda = sx * (sy + sz) + sy * sz
    sum += factor / (sz * (z + lambda))
    factor /= 4
    x = (x + lambda) / 4
    y = (y + lambda) / 4
    z = (z + lambda) / 4
  }
  return NaN
}

/** The complete elliptic integral of the first kind K(m) = ∫₀^{π/2} dθ / √(1 − m sin²θ), m < 1 (∞ at 1). */
export function ellipk(m: number): number {
  if (m === 1) return Infinity
  if (m > 1) return NaN
  return carlsonRF(0, 1 - m, 1)
}

/** K(1 − p) for small p ≥ 0 without the cancellation of 1 − (1 − p), as `scipy.special.ellipkm1`. */
export function ellipkm1(p: number): number {
  if (p === 0) return Infinity
  if (p < 0) return NaN
  return carlsonRF(0, p, 1)
}

/** The complete elliptic integral of the second kind E(m) = ∫₀^{π/2} √(1 − m sin²θ) dθ, m ≤ 1. */
export function ellipe(m: number): number {
  if (m === 1) return 1
  if (m > 1) return NaN
  const y = 1 - m
  return carlsonRF(0, y, 1) - (m / 3) * carlsonRD(0, y, 1)
}

/**
 * The incomplete elliptic integral of the first kind F(φ | m) = ∫₀^φ dθ / √(1 − m sin²θ) for |φ| ≤ π/2, as
 * `scipy.special.ellipkinc`. `complement` (default 1 − m) may be given directly when m is near 1.
 */
export function ellipf(phi: number, m: number, complement = 1 - m): number {
  if (phi === 0) return 0
  const s = Math.sin(phi)
  const c = Math.cos(phi)
  // 1 − m sin²φ = cos²φ + (1 − m) sin²φ, accurate when 1 − m is tiny.
  return s * carlsonRF(c * c, c * c + complement * s * s, 1)
}

/** The Jacobi elliptic functions at (u, m) and the amplitude φ = am(u | m). */
export type Jacobi = { sn: number; cn: number; dn: number; ph: number }

/**
 * The Jacobi elliptic functions sn, cn, dn and the amplitude φ of u with parameter 0 ≤ m ≤ 1, as
 * `scipy.special.ellipj`: by the descending Landen transformation (AGM), with series near m = 0 and m = 1.
 */
export function ellipj(u: number, m: number): Jacobi {
  if (!(m >= 0 && m <= 1) || Number.isNaN(u)) return { sn: NaN, cn: NaN, dn: NaN, ph: NaN }
  if (m < 1e-9) {
    const t = Math.sin(u)
    const b = Math.cos(u)
    const ai = 0.25 * m * (u - t * b)
    return { sn: t - ai * b, cn: b + ai * t, ph: u - ai, dn: 1 - 0.5 * m * t * t }
  }
  if (m >= 0.9999999999) {
    const ai0 = 0.25 * (1 - m)
    const b = Math.cosh(u)
    const t = Math.tanh(u)
    const phi = 1 / b
    const twon = b * Math.sinh(u)
    const sn = t + (ai0 * (twon - u)) / (b * b)
    const ph = 2 * Math.atan(Math.exp(u)) - Math.PI / 2 + (ai0 * (twon - u)) / b
    const ai = ai0 * t * phi
    return { sn, cn: phi - ai * (twon - u), dn: phi + ai * (twon + u), ph }
  }
  const a = [1]
  const c = [Math.sqrt(m)]
  let b = Math.sqrt(1 - m)
  let twon = 1
  let i = 0
  while (Math.abs(c[i] / a[i]) > Number.EPSILON && i < 8) {
    const ai = a[i]
    i++
    c[i] = (ai - b) / 2
    const t = Math.sqrt(ai * b)
    a[i] = (ai + b) / 2
    b = t
    twon *= 2
  }
  let phi = twon * a[i] * u
  let prev = phi
  do {
    const t = (c[i] * Math.sin(phi)) / a[i]
    prev = phi
    phi = (Math.asin(t) + phi) / 2
  } while (--i)
  const sn = Math.sin(phi)
  const cn = Math.cos(phi)
  return { sn, cn, dn: cn / Math.cos(phi - prev), ph: phi }
}
