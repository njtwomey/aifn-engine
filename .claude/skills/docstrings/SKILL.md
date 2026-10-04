---
name: docstrings
description: Document aifn-engine source to the project's docstring pattern, which generates the /compute and /methods pages of the site. Use when asked to document, add docstrings, examples, parameter descriptions or maths to a function, file or module under packages/compute or packages/methods, or to bring a module "up to the pattern".
---

# Docstrings

The pages under `/compute` and `/methods` of the site are generated from the doc comments in the source
(`examples/plugins/docs.ts`). The documentation lives beside the code, so writing a docstring is writing the page.
The reference for everything below is `packages/compute/src/numerics/linalg/cholesky.ts`: read it before starting.

Edit comments only. Never change code, signatures or behaviour while documenting.

## The split: key and supporting functions

Every top-level function of a file is listed on the file's page, in two groups.

| Function                                                    | Doc comment | Parameters described | Example      |
| ----------------------------------------------------------- | ----------- | -------------------- | ------------ |
| **Key**: exported by the module's `index.ts`, so importable | required    | required             | at least one |
| **Supporting**: exported by its file only, or not exported  | required    | required             | not needed   |

Types need a doc comment, and a comment on each field.

## File and module docstrings

**Every source file opens with a doc comment**, before the imports. It heads the file's page, above its functions, so
it is the introduction a reader gets:

```ts
/**
 * Cholesky factorisation $\Amat = \Lmat\Lmat^\top$ with jitter reporting, and the solves and log-determinant built
 * on it.
 *
 * The factorisation is the column-by-column (Cholesky–Crout) algorithm of Golub and Van Loan (2013), Algorithm 4.2.2.
 * When a pivot is not positive, the smallest jitter $j = s \cdot 10^k$ ($s$ the mean diagonal) that lets
 * $\Amat + j\Imat$ factor is added and reported, ...
 */

import { ... } from 'aifn-compute/foundation/tensor'
```

- First sentence: what the file holds, in one line (it is also used as a summary).
- Then, as needed: the method and its source (author, year, section or equation), the conventions the file's
  functions share (storage layout, what is reported rather than thrown), and how its pieces fit together.
- A file of helpers says so ("Internal helpers: dense row-major working copies of matrices, ...").
- It must be a `/** */` block at the very top, followed by a blank line or the imports; otherwise it is taken for the
  doc comment of the first declaration and the file's page has no introduction.
- No `@param`, `@returns` or `@example` here: those belong to functions.

**A module's `index.ts` opens with the module's docstring**, which heads the module's page: the import path in
backticks, a one-line statement of what the module is, then what it contains by theme (a short list naming the
functions), and what holds across them (differentiability, failure reporting). Examples do not go here: put each on the
function it is about.

## The shape of a docstring

Prose, then tags, then examples, each separated by a blank ` *` line:

```ts
/**
 * Solve $\Amat\Xmat = \Bmat$ given $\Amat$'s Cholesky factor $\Lmat$ ($n \times n$) and $\Bmat$ ($n$ or
 * $n \times r$), by two triangular solves (differentiable).
 *
 * @param L The lower-triangular Cholesky factor of $\Amat$ ($n \times n$), as `cholesky` returns it. Not $\Amat$
 *   itself.
 * @param b The right-hand side: a vector of $n$ values, or an $n \times r$ matrix whose columns are solved together.
 * @returns The solution $\Xmat$, with the shape of `b`.
 *
 * @example Factor once, then solve for several right-hand sides
 * const { L } = cholesky(tensor([[4, 1], [1, 3]]))
 * print('x1 =', choleskySolve(L, tensor([1, 2])))
 * print('x2 =', choleskySolve(L, tensor([0, 1])))
 */
export function choleskySolve<L extends Value, B extends Value>(L: L, b: B): TensorResult<L | B> {
```

### Prose

Say what the function does and what a caller must know: failure behaviour (what throws, what is reported in the
result), whether it is differentiable, the source of the method (author, year, section). One or two sentences is
enough for a supporting function.

### Parameters and return value

- One `@param <name> <description>` per parameter, in order, and one `@returns <description>` (omit for `void`).
- The name, type and default come from the signature and are shown beside the description, so do not restate them.
  Say what the value **is** and is **for**: an array's shape and layout ("row-major array of $n^2$ values", "row `i`
  occupies entries `i * d` to `i * d + d - 1`"), whether it is read, modified in place or overwritten, what an index or
  count refers to, the unit or meaning of a number, what a flag switches, what happens when an optional one is left out.
- Read the function body to establish this. Do not guess from the name.
- A `where: string` parameter is the caller's name for error messages: say so.
- Names must match the signature exactly. Special cases:
  - An inline options pattern, `{ maxSweeps = 100 }: { maxSweeps?: number } = {}`: `@param options ...`, then
    `@param options.maxSweeps ...` for each field.
  - A named options type that is not destructured, `options: CholeskyOptions = {}`: one `@param options` line; the
    type's own field comments carry the detail.
  - An array pattern, `[[a, b], [c, d]]: Mat2`: choose a name (`@param m The 2×2 matrix as two rows.`); such tags are
    matched in order.

### Examples (key functions)

- `@example <title>` on its own line, then the code. The title makes it a runnable, editable cell on the page. (An
  `@example` with code on the tag's line and no title is shown as plain code and never run: avoid it.)
- In scope: every export of the function's module, and the common surface of `aifn-compute` (`tensor`, `matmul`,
  `transpose`, `grad`, `sum`, `run`, `trace`, streams, ...). Write the code as if those were imported; the page shows
  the import lines it works out.
- Show results with `print(label, value)`. A bare expression on a line of its own also shows its value. Do not write
  expected values in comments: the cell prints them.
- Keep it small and illustrative: tiny inputs whose answer a reader can check by eye, one idea per example. Good
  second examples show a failure being reported, a limit case, or the function used with its neighbours.
- Step-through algorithms are run with `run(alg, start, steps)` and read from the state it returns.
- Examples run on the page's own thread: no loops that could run long.

### Maths

Write maths as TeX between single dollar signs; it is set by KaTeX. Do not use Unicode superscripts, subscripts or
operators (`LLᵀ`, `n×n`, `Σ`) in doc comments.

- Use the shared notation macros so notation is consistent: bold matrices `\Amat`, `\Lmat`, `\Imat` (a capital +
  `mat`), bold vectors `\xvec`, `\bvec` (a lowercase letter + `vec`), Greek `\Sigmamat`, `\thetavec`. The full list
  (`\reals`, `\expect`, `\norm`, `\Gauss`, ...) is in `packages/render/src/layout/math-macros.tsx`.
- Matrices are bold capitals, vectors bold lowercase, scalars and dimensions plain italics: `$n \times n$`,
  `$\Amat\xvec = \bvec$`, `$\Lmat\Lmat^\top = \Amat + j\Imat$`, `$\log\det\Amat = 2\sum_i \log L_{ii}$`.
- Code identifiers stay in backticks, not in maths: `` `jitter` ``, `` `luSolve` ``.
- This applies to doc comments (`/** */`): prose, tags and the comments on type fields. Leave `//` comments inside
  function bodies, example code and string literals (registry `summary` and `tex` values) as they are.
- A backslash needs no escaping inside a comment.

### Formatting

Lines are at most 120 columns. A long tag wraps onto continuation lines indented two more spaces
(` *   continued text`). Prettier does not rewrap comments, so wrap by hand. British spelling; no em dashes.

## Process

1. List what a module or file lacks:

   ```bash
   node examples/check.ts --missing compute/numerics/linalg          # a module
   node examples/check.ts --missing compute/numerics/linalg/cholesky # one file
   ```

   It reports files with no opening comment, functions with no comment, parameters not described, and key functions
   without an example.

2. Read the file and its functions before writing. Work one file at a time.
3. Write the docstrings. Put an example on the function it is about, not in the module's `index.ts` comment.
4. Verify:

   ```bash
   node examples/check.ts --missing <path>   # nothing missing; 0 examples failed; 0 formulas failed
   node examples/check.ts --show             # prints every example's output: read it, the values must make sense
   npx prettier --write <files> && npx tsc -b
   git diff -- <files> | grep '^[+-]' | grep -v '^[+-]\s*\(/\*\*\|\*\|\*/\)' | grep -v '^+++\|^---'   # comments only: prints nothing
   ```

   `make examples-check` (part of `make check`) runs every example and sets every formula, and fails when one throws
   or does not set.

5. Look at the page: `make examples`, then `/compute/<family>/<module>/<file>`.

For a large module, the files are independent: split them between parallel agents, each given its file list, this
skill and `cholesky.ts` as the reference, and verify the whole module afterwards.
