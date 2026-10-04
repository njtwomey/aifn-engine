import { defineConfig } from 'vitest/config'

export default defineConfig({
  root: import.meta.dirname,
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
    // Several tests take 2–5 s alone (fits, samplers, the name lint) and miss vitest's 5 s default when the machine is
    // busy; a generous default stops load-dependent failures. Long training tests still set their own.
    testTimeout: 30_000,
  },
})
