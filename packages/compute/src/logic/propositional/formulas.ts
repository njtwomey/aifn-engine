/**
 * Propositional formulas as plain data, their truth values, models and entailment by enumeration, and printing.
 *
 * A formula is built from atoms and the constants $\top$, $\bot$ with $\neg$, $\land$, $\lor$, $\rightarrow$ and
 * $\leftrightarrow$. An interpretation assigns true or false to each atom; a model of a set of formulas is an
 * interpretation that makes all of them true; and a knowledge base entails a formula, $\mathrm{KB} \models \varphi$,
 * when $\varphi$ is true in every model of $\mathrm{KB}$ (standard semantics, e.g. Russell and Norvig, "Artificial
 * Intelligence: A Modern Approach", 4th ed., §7.4). Entailment is decided by enumerating all $2^n$ interpretations of
 * $n$ atoms, so the functions are meant for teaching sizes: `interpretations` refuses more than 20 atoms.
 *
 * Printing follows the usual precedence: $\neg$ binds tightest, then $\land$, $\lor$, $\rightarrow$, $\leftrightarrow$;
 * $\rightarrow$ and $\leftrightarrow$ group to the right.
 */
import { DomainError } from 'aifn-compute/foundation/errors'

/** A propositional formula, told apart by `kind`. */
export type Formula =
  | {
      /** An atom (a propositional variable). */
      readonly kind: 'atom'
      /** The atom's name. */
      readonly name: string
    }
  | {
      /** A constant, $\top$ or $\bot$. */
      readonly kind: 'const'
      /** True for $\top$, false for $\bot$. */
      readonly value: boolean
    }
  | {
      /** A negation $\neg\varphi$. */
      readonly kind: 'not'
      /** The negated formula $\varphi$. */
      readonly arg: Formula
    }
  | {
      /** A binary connective: $\land$, $\lor$, $\rightarrow$ or $\leftrightarrow$. */
      readonly kind: 'and' | 'or' | 'implies' | 'iff'
      /** The left operand. */
      readonly left: Formula
      /** The right operand. */
      readonly right: Formula
    }

/** An interpretation: the truth value of each atom, by name. */
export type Interpretation = Readonly<Record<string, boolean>>

/**
 * The atom `name`.
 *
 * @param name The atom's name.
 * @returns The formula.
 *
 * @example An atom, printed
 * print(formulaText(prop('rain')))
 */
export const prop = (name: string): Formula => ({ kind: 'atom', name })

/**
 * The constant $\top$ (true) or $\bot$ (false).
 *
 * @param value Which constant: true for $\top$, false for $\bot$.
 * @returns The formula.
 *
 * @example The two constants
 * print(formulaText(truth(true)), formulaText(truth(false)))
 */
export const truth = (value: boolean): Formula => ({ kind: 'const', value })

/**
 * The negation $\neg\varphi$.
 *
 * @param arg The formula $\varphi$ negated.
 * @returns The formula.
 *
 * @example Negate a conjunction
 * print(formulaText(negate(conjoin(prop('a'), prop('b')))))
 */
export const negate = (arg: Formula): Formula => ({ kind: 'not', arg })

/**
 * The conjunction $\varphi_1 \land \varphi_2 \land \dots$, nested to the right; $\top$ when empty.
 *
 * @param args The conjuncts, in order.
 * @returns The formula.
 *
 * @example Conjunctions of three, one and none
 * print(formulaText(conjoin(prop('a'), prop('b'), prop('c'))))
 * print(formulaText(conjoin(prop('a'))), formulaText(conjoin()))
 */
export const conjoin = (...args: Formula[]): Formula =>
  args.length === 0 ? truth(true) : args.reduceRight((r, f) => ({ kind: 'and', left: f, right: r }))

/**
 * The disjunction $\varphi_1 \lor \varphi_2 \lor \dots$, nested to the right; $\bot$ when empty.
 *
 * @param args The disjuncts, in order.
 * @returns The formula.
 *
 * @example A disjunction inside a conjunction is bracketed
 * print(formulaText(conjoin(prop('a'), disj(prop('b'), prop('c')))))
 */
export const disj = (...args: Formula[]): Formula =>
  args.length === 0 ? truth(false) : args.reduceRight((r, f) => ({ kind: 'or', left: f, right: r }))

/**
 * The implication $\varphi \rightarrow \psi$.
 *
 * @param left The antecedent $\varphi$.
 * @param right The consequent $\psi$.
 * @returns The formula.
 *
 * @example Implication groups to the right
 * print(formulaText(implies(prop('a'), implies(prop('b'), prop('c')))))
 * print(formulaText(implies(implies(prop('a'), prop('b')), prop('c'))))
 */
export const implies = (left: Formula, right: Formula): Formula => ({ kind: 'implies', left, right })

/**
 * The biconditional $\varphi \leftrightarrow \psi$.
 *
 * @param left The left side $\varphi$.
 * @param right The right side $\psi$.
 * @returns The formula.
 *
 * @example A biconditional is true when both sides agree
 * const f = iff(prop('a'), prop('b'))
 * print(interpretations(['a', 'b']).map((I) => evaluateFormula(f, I)))
 */
export const iff = (left: Formula, right: Formula): Formula => ({ kind: 'iff', left, right })

/**
 * The truth value of a formula under an interpretation. An atom the interpretation does not assign throws
 * `DomainError`.
 *
 * @param f The formula.
 * @param I The interpretation: a truth value for every atom of `f` (others are ignored).
 * @returns Whether `f` is true under `I`.
 *
 * @example Evaluate under one interpretation
 * const f = implies(prop('rain'), prop('wet'))
 * print('rain, not wet:', evaluateFormula(f, { rain: true, wet: false }))
 * print('no rain:', evaluateFormula(f, { rain: false, wet: false }))
 */
export function evaluateFormula(f: Formula, I: Interpretation): boolean {
  switch (f.kind) {
    case 'atom': {
      const v = I[f.name]
      if (v === undefined)
        throw new DomainError('evaluateFormula', `atom ${f.name} has no truth value in the interpretation`)
      return v
    }
    case 'const':
      return f.value
    case 'not':
      return !evaluateFormula(f.arg, I)
    case 'and':
      return evaluateFormula(f.left, I) && evaluateFormula(f.right, I)
    case 'or':
      return evaluateFormula(f.left, I) || evaluateFormula(f.right, I)
    case 'implies':
      return !evaluateFormula(f.left, I) || evaluateFormula(f.right, I)
    case 'iff':
      return evaluateFormula(f.left, I) === evaluateFormula(f.right, I)
  }
}

/**
 * The distinct atoms of the formulas, in order of first occurrence.
 *
 * @param formulas The formulas.
 * @returns The atoms' names.
 *
 * @example Atoms of a knowledge base
 * print(atomsOf(implies(prop('rain'), prop('wet')), disj(prop('rain'), prop('sprinkler'))))
 */
export function atomsOf(...formulas: Formula[]): string[] {
  const out: string[] = []
  const walk = (f: Formula) => {
    if (f.kind === 'atom') {
      if (!out.includes(f.name)) out.push(f.name)
    } else if (f.kind === 'not') walk(f.arg)
    else if (f.kind !== 'const') {
      walk(f.left)
      walk(f.right)
    }
  }
  formulas.forEach(walk)
  return out
}

/**
 * Every interpretation of the atoms, $2^n$ of them, in truth-table order: the first atom varies slowest, true before
 * false (the first row makes every atom true). More than 20 atoms throws `DomainError`.
 *
 * @param atoms The atoms' names, $n$ of them.
 * @returns The interpretations, the rows of the truth table in order.
 *
 * @example The rows of a truth table
 * for (const I of interpretations(['p', 'q'])) print(I)
 */
export function interpretations(atoms: readonly string[]): Interpretation[] {
  const n = atoms.length
  if (n > 20) throw new DomainError('interpretations', `${n} atoms give 2^${n} interpretations: too many to enumerate`)
  const out: Interpretation[] = []
  for (let r = 0; r < 2 ** n; r++) {
    const I: Record<string, boolean> = {}
    atoms.forEach((a, i) => {
      I[a] = ((r >> (n - 1 - i)) & 1) === 0
    })
    out.push(I)
  }
  return out
}

/**
 * The models of a set of formulas: the interpretations that make every formula true.
 *
 * @param formulas The formulas (a knowledge base), read as their conjunction.
 * @param atoms The atoms to enumerate over; they must include every atom of `formulas`.
 * @returns The indices of the models into `interpretations(atoms)`, in increasing order.
 *
 * @example An implication has three models of four
 * print(modelsOf([implies(prop('rain'), prop('wet'))], ['rain', 'wet']))
 */
export function modelsOf(formulas: readonly Formula[], atoms: readonly string[]): number[] {
  return interpretations(atoms).flatMap((I, r) => (formulas.every((f) => evaluateFormula(f, I)) ? [r] : []))
}

/** The outcome of `entails`. */
export interface Entailment {
  /** Whether $\mathrm{KB} \models \varphi$: no counter-model exists. */
  readonly entailed: boolean
  /** The models of the knowledge base, as indices into `interpretations(atoms)`. */
  readonly models: number[]
  /** The models of the knowledge base in which the query is false, as indices into `interpretations(atoms)`. */
  readonly counterModels: number[]
}

/**
 * Whether a knowledge base entails a query, $\mathrm{KB} \models \varphi$: $\varphi$ true in every model of
 * $\mathrm{KB}$, decided by enumerating the truth table. An unsatisfiable knowledge base entails everything.
 *
 * @param kb The knowledge base, read as the conjunction of its formulas.
 * @param query The query $\varphi$.
 * @param atoms The atoms to enumerate over (default: those of `kb` and `query`, in order of first occurrence).
 * @returns Whether it is entailed, with the models of the knowledge base and its counter-models.
 *
 * @example Modus ponens by cases, and affirming the consequent
 * const kb = [
 *   implies(prop('rain'), prop('wet')),
 *   implies(prop('sprinkler'), prop('wet')),
 *   disj(prop('rain'), prop('sprinkler')),
 * ]
 * print('KB entails wet:', entails(kb, prop('wet')).entailed)
 * const r = entails(kb, prop('rain'))
 * print('KB entails rain:', r.entailed, 'counter-models:', r.counterModels)
 */
export function entails(
  kb: readonly Formula[],
  query: Formula,
  atoms: readonly string[] = atomsOf(...kb, query),
): Entailment {
  const all = interpretations(atoms)
  const models = modelsOf(kb, atoms)
  const counterModels = models.filter((r) => !evaluateFormula(query, all[r]))
  return { entailed: counterModels.length === 0, models, counterModels }
}

/** The precedence of each kind of formula: higher binds tighter. */
const PRECEDENCE: Record<Formula['kind'], number> = { atom: 5, const: 5, not: 4, and: 3, or: 2, implies: 1, iff: 0 }

/** The symbols of each connective and constant, in plain text and in KaTeX. */
const SYMBOL = {
  text: { not: '¬', and: ' ∧ ', or: ' ∨ ', implies: ' → ', iff: ' ↔ ', true: '⊤', false: '⊥' },
  tex: {
    not: '\\neg ',
    and: ' \\land ',
    or: ' \\lor ',
    implies: ' \\rightarrow ',
    iff: ' \\leftrightarrow ',
    true: '\\top',
    false: '\\bot',
  },
}

/**
 * The formula as plain text or as KaTeX source, bracketed only where precedence needs it. In KaTeX atoms print as
 * upright text (`\text{rain}`).
 *
 * @param f The formula.
 * @param style `'text'` for Unicode connectives, `'tex'` for KaTeX source.
 * @returns The printed formula.
 *
 * @example Brackets only where needed
 * const f = implies(conjoin(prop('a'), disj(prop('b'), prop('c'))), negate(conjoin(prop('a'), prop('b'))))
 * print(formulaText(f))
 * print(formulaText(f, 'tex'))
 */
export function formulaText(f: Formula, style: 'text' | 'tex' = 'text'): string {
  const s = SYMBOL[style]
  // `outer` is the least precedence the context accepts without brackets.
  const go = (g: Formula, outer: number): string => {
    let body: string
    if (g.kind === 'atom') body = style === 'tex' ? `\\text{${g.name}}` : g.name
    else if (g.kind === 'const') body = g.value ? s.true : s.false
    else if (g.kind === 'not') body = `${s.not}${go(g.arg, PRECEDENCE.not)}`
    else {
      const p = PRECEDENCE[g.kind]
      // ∧ and ∨ are associative; → and ↔ group to the right, so only a left operand of the same kind is bracketed.
      const rightAssoc = g.kind === 'implies' || g.kind === 'iff'
      body = `${go(g.left, rightAssoc ? p + 0.5 : p)}${s[g.kind]}${go(g.right, p)}`
    }
    return PRECEDENCE[g.kind] < outer ? (style === 'tex' ? `\\left(${body}\\right)` : `(${body})`) : body
  }
  return go(f, -1)
}
