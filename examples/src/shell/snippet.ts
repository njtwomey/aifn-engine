/**
 * The code shown under a recipe: the recipe file's own source (imported with `?raw`, so code and demo never drift),
 * trimmed to its key ingredient.
 *
 * - The imports are kept, except the recipe metadata's (`@examples/recipe`).
 * - `export const recipe = { … }` is dropped.
 * - When the file has `// region` … `// endregion` blocks (or `region` and `endregion` JSX comments), only those
 *   are shown after the imports (several blocks are joined by `// …`); otherwise the whole rest of the file is.
 * - A line ending in `// hide` is dropped.
 */
export function snippetOf(source: string): string {
  const lines = source.replace(/\r\n/g, '\n').split('\n')
  const imports: string[] = []
  let i = 0
  // Imports at the top (single or multi-line), skipping blank lines and a leading file comment.
  for (; i < lines.length; i++) {
    const line = lines[i]
    if (!line.trim()) continue
    if (!line.startsWith('import ')) break
    const statement = [line]
    while (!/from\s+['"][^'"]+['"]\s*;?\s*$/.test(statement[statement.length - 1]) && i + 1 < lines.length)
      statement.push(lines[++i])
    if (!/['"]@examples\/recipe['"]/.test(statement.join('\n'))) imports.push(...statement)
  }
  let rest = lines.slice(i)
  // Drop the metadata block: from `export const recipe` to the closing brace at the start of a line.
  const start = rest.findIndex((l) => /^export const recipe\b/.test(l))
  if (start >= 0) {
    let end = start
    while (end < rest.length && !/^}/.test(rest[end])) end++
    rest = [...rest.slice(0, start), ...rest.slice(end + 1)]
  }
  rest = rest.filter((l) => !/\/\/\s*hide\s*$/.test(l))
  const regions: string[][] = []
  let open: string[] | null = null
  for (const l of rest) {
    if (marker(l, 'region')) open = []
    else if (marker(l, 'endregion')) {
      if (open) regions.push(open)
      open = null
    } else open?.push(l)
  }
  const body = regions.length ? regions.map(dedent).join('\n\n// …\n\n') : rest.join('\n')
  return tidy([imports.join('\n'), body].filter((s) => s.trim()).join('\n\n'))
}

/** A `// region` line, or a JSX comment holding only `region`. */
const marker = (line: string, word: string) =>
  new RegExp(`^\\s*(//\\s*${word}\\b.*|\\{/\\*\\s*${word}\\b.*\\*/\\})\\s*$`).test(line)

/** Removes the indentation every non-blank line shares. */
function dedent(block: string[]): string {
  const indents = block.filter((l) => l.trim()).map((l) => /^ */.exec(l)![0].length)
  const cut = indents.length ? Math.min(...indents) : 0
  return block.map((l) => l.slice(cut)).join('\n')
}

/** Trims, and collapses runs of blank lines to one. */
const tidy = (s: string) => s.replace(/\n{3,}/g, '\n\n').trim() + '\n'
