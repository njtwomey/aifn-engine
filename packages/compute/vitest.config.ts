import { defineConfig } from 'vitest/config'

export default defineConfig({
  root: import.meta.dirname,
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
    // Several tests take 2–10 s alone (fits, samplers, the name lint) and several times that on a shared CI runner, so
    // the default is generous; it only bounds a hung test. Long training tests still set their own.
    testTimeout: 120_000,
  },
})
