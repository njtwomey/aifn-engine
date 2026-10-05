/**
 * Mathematical notation macros and KaTeX rendering context for AIFN render.
 *
 * Provides standard machine learning and mathematical notation out-of-the-box
 * (vectors \xvec, matrices \Xmat, Greek vectors/matrices \psivec, \Psimat, sets \Dcal,
 * probability operators \reals, \expect, etc.), along with a React context, hook,
 * and global registry so consumers can easily provide or extend symbols.
 */
import katex, { type KatexOptions } from 'katex'
import { createContext, useContext, useMemo, type ReactNode } from 'react'

const LOWER = 'abcdefghijklmnopqrstuvwxyz'.split('')
const UPPER = LOWER.map((c) => c.toUpperCase())
const GREEK_LOWER = [
  'alpha',
  'beta',
  'gamma',
  'delta',
  'epsilon',
  'varepsilon',
  'zeta',
  'eta',
  'theta',
  'vartheta',
  'iota',
  'kappa',
  'lambda',
  'mu',
  'nu',
  'xi',
  'pi',
  'rho',
  'sigma',
  'tau',
  'upsilon',
  'phi',
  'varphi',
  'chi',
  'psi',
  'omega',
]
const GREEK_UPPER = ['Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega']

const family = (names: string[], name: (s: string) => string, body: (s: string) => string) =>
  Object.fromEntries(names.map((s) => [`\\${name(s)}`, body(s)]))

/** Built-in standard mathematical and machine learning macros. */
export const defaultMathMacros: Record<string, string> = {
  // Vectors (bold lowercase & bold Greek)
  ...family(
    LOWER,
    (c) => `${c}vec`,
    (c) => `\\mathbf{${c}}`,
  ),
  ...family(
    GREEK_LOWER,
    (g) => `${g}vec`,
    (g) => `\\boldsymbol{\\${g}}`,
  ),
  ...family(
    GREEK_LOWER,
    (g) => `${g}b`,
    (g) => `\\boldsymbol{\\${g}}`,
  ),
  '\\ones': '\\mathbf{1}',
  '\\zeros': '\\mathbf{0}',

  // Matrices (bold uppercase & bold Greek)
  ...family(
    UPPER,
    (c) => `${c}mat`,
    (c) => `\\mathbf{${c}}`,
  ),
  ...family(
    GREEK_UPPER,
    (g) => `${g}mat`,
    (g) => `\\boldsymbol{\\${g}}`,
  ),
  ...family(
    GREEK_UPPER,
    (g) => `${g}b`,
    (g) => `\\boldsymbol{\\${g}}`,
  ),
  '\\eye': '\\mathbf{I}',
  '\\I': '\\mathbf{I}',
  '\\transpose': '^{\\mathsf{\\top}}',
  '\\tr': '\\top',

  // Sets and families (calligraphic capitals)
  ...family(
    UPPER,
    (c) => `${c}cal`,
    (c) => `\\mathcal{${c}}`,
  ),

  // Number systems and probability
  '\\reals': '\\mathbb{R}',
  '\\complex': '\\mathbb{C}',
  '\\rationals': '\\mathbb{Q}',
  '\\integers': '\\mathbb{Z}',
  '\\naturals': '\\mathbb{N}',
  '\\expect': '\\mathbb{E}',
  '\\prob': '\\mathbb{P}',
  '\\pr': '\\operatorname{Pr}',
  '\\indicator': '\\mathbb{I}',
  '\\iid': '\\mathrel{\\overset{\\text{iid}}{\\sim}}',
  '\\eqdef': '\\stackrel{\\mathrm{def}}{=}',

  // Operators
  '\\argmin': '\\operatorname*{arg\\,min}',
  '\\argmax': '\\operatorname*{arg\\,max}',
  '\\prox': '\\operatorname{prox}',
  '\\trace': '\\operatorname{tr}',
  '\\diag': '\\operatorname{diag}',
  '\\rank': '\\operatorname{rank}',
  '\\Span': '\\operatorname{span}',
  '\\sgn': '\\operatorname{sgn}',
  '\\var': '\\operatorname{var}',
  '\\cov': '\\operatorname{cov}',
  '\\corr': '\\operatorname{corr}',
  '\\KL': '\\operatorname{KL}',
  '\\entropy': '\\operatorname{H}',
  '\\Null': '\\operatorname{null}',
  '\\row': '\\operatorname{row}',
  '\\col': '\\operatorname{col}',
  '\\proj': '\\operatorname{proj}',
  '\\AIC': '\\operatorname{AIC}',
  '\\BIC': '\\operatorname{BIC}',

  // Delimiters
  '\\abs': '\\left\\lvert #1 \\right\\rvert',
  '\\norm': '\\left\\lVert #1 \\right\\rVert',
  '\\inner': '\\left\\langle #1,\\, #2 \\right\\rangle',
  '\\set': '\\left\\{ #1 \\right\\}',
  '\\paren': '\\left( #1 \\right)',
  '\\brack': '\\left[ #1 \\right]',
  '\\LP': '\\left(',
  '\\RP': '\\right)',
  '\\LB': '\\left[',
  '\\RB': '\\right]',
  '\\LC': '\\left\\{',
  '\\RC': '\\right\\}',
  '\\LA': '\\left\\langle',
  '\\RA': '\\right\\rangle',

  // Distributions
  '\\Gauss': '\\mathcal{N}',
  '\\Bern': '\\operatorname{Bern}',
  '\\Binom': '\\operatorname{Binom}',
  '\\Cat': '\\operatorname{Cat}',
  '\\Mult': '\\operatorname{Mult}',
  '\\Poisson': '\\operatorname{Poisson}',
  '\\Geom': '\\operatorname{Geom}',
  '\\Unif': '\\operatorname{Unif}',
  '\\Exp': '\\operatorname{Exp}',
  '\\GammaD': '\\operatorname{Gamma}',
  '\\Beta': '\\operatorname{Beta}',
  '\\Dir': '\\operatorname{Dir}',
  '\\NegBin': '\\operatorname{NegBin}',
  '\\InvGamma': '\\operatorname{InvGamma}',
  '\\LogNorm': '\\operatorname{LogNormal}',
  '\\MvGauss': '\\mathcal{N}',
  '\\Cauchy': '\\operatorname{Cauchy}',
  '\\Weibull': '\\operatorname{Weibull}',
  '\\Gumbel': '\\operatorname{Gumbel}',
  '\\StudentT': '\\operatorname{t}',
  '\\Laplace': '\\operatorname{Laplace}',
  '\\Wishart': '\\operatorname{Wishart}',
  '\\ChiSq': '\\chi^2',
  '\\Hypergeom': '\\operatorname{Hypergeom}',
  '\\FDist': '\\operatorname{F}',
  '\\GP': '\\mathcal{GP}',
}

/** Global macros registry initialized with the default ML and math macros. */
const globalMacros: Record<string, string> = { ...defaultMathMacros }

/** Register or extend global mathematical macros for KaTeX rendering across all components. */
export function registerMathMacros(macros: Record<string, string>): void {
  Object.assign(globalMacros, macros)
}

/** Get a copy of the current global mathematical macros. */
export function getGlobalMathMacros(): Record<string, string> {
  return { ...globalMacros }
}

/** React Context for scoped mathematical macros. */
export const RenderMathContext = createContext<Record<string, string> | undefined>(undefined)

/** Provider to supply or override mathematical macros for a subtree. */
export function RenderMathProvider({ macros, children }: { macros?: Record<string, string>; children: ReactNode }) {
  const parentMacros = useContext(RenderMathContext)
  const merged = useMemo(() => ({ ...(parentMacros ?? {}), ...(macros ?? {}) }), [parentMacros, macros])
  return <RenderMathContext.Provider value={merged}>{children}</RenderMathContext.Provider>
}

/**
 * Hook to retrieve effective mathematical macros (merging global registry and current React context).
 */
export function useRenderMathMacros(): Record<string, string> {
  const contextMacros = useContext(RenderMathContext)
  return useMemo(() => ({ ...globalMacros, ...(contextMacros ?? {}) }), [contextMacros])
}

const katexCache = new Map<string, string>()

/**
 * Render LaTeX using KaTeX with effective mathematical macros and caching.
 */
export function renderKatex(tex: string, options?: KatexOptions & { macros?: Record<string, string> }): string {
  const display = options?.displayMode ?? false
  const effectiveMacros = options?.macros ?? globalMacros
  const hasCustomMacros = !!options?.macros
  const cacheKey = `${display ? 'D' : 'I'}:${tex}`

  if (!hasCustomMacros && katexCache.has(cacheKey)) {
    return katexCache.get(cacheKey)!
  }

  try {
    const html = katex.renderToString(tex, {
      throwOnError: false,
      output: 'html',
      strict: 'ignore',
      ...options,
      macros: { ...effectiveMacros },
    })
    if (!hasCustomMacros) {
      katexCache.set(cacheKey, html)
    }
    return html
  } catch {
    return tex
  }
}
