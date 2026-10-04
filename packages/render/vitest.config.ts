import path from 'node:path'
import { defineConfig } from 'vitest/config'

const src = path.join(import.meta.dirname, 'src')

export default defineConfig({
  root: import.meta.dirname,
  resolve: {
    alias: [
      { find: /^@render\//, replacement: `${src}/` },
      { find: /^aifn-render\//, replacement: `${src}/` },
      { find: /^aifn-render$/, replacement: `${src}/index.ts` },
    ],
  },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
})
