/**
 * `aifn-compute/logic/propositional`: propositional formulas, their models and entailment by truth table.
 *
 * - Formulas as plain data: `prop` (an atom), `truth` ($\top$ or $\bot$), `negate`, `conjoin`, `disj`, `implies` and
 *   `iff`; `atomsOf` lists a formula's atoms and `formulaText` prints it as text or KaTeX with the usual precedence.
 * - Semantics: `evaluateFormula` gives a formula's truth value under an interpretation, `interpretations` lists all
 *   $2^n$ rows of the truth table, `modelsOf` the rows that satisfy a set of formulas, and `entails` decides
 *   $\mathrm{KB} \models \varphi$ with its counter-models.
 * - `propositionalFunctions`: the module's registry entries.
 *
 * Everything is by enumeration, exponential in the number of atoms and meant for teaching sizes (at most 20 atoms).
 * Interpretations are records from atom names to truth values, and models are indices into `interpretations(atoms)`.
 */

export {
  atomsOf,
  conjoin,
  disj,
  entails,
  evaluateFormula,
  formulaText,
  iff,
  implies,
  interpretations,
  modelsOf,
  negate,
  prop,
  truth,
  type Entailment,
  type Formula,
  type Interpretation,
} from './formulas'
export { propositionalFunctions } from './registry'
