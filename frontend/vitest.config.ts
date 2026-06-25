import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['tests/vitest.setup.ts'],
    include: ['tests/{unit,integration}/**/*.{test,spec}.ts'],
    exclude: ['tests/e2e/**'],
    server: {
      deps: {
        // Inline @vue-leaflet so Vite transforms its dynamic
        // import("leaflet/dist/leaflet-src.esm") and the alias below applies.
        inline: ['@vue-leaflet/vue-leaflet'],
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      // @vue-leaflet imports the extensionless ESM entry, which Node's ESM
      // resolver (used by Vitest) cannot resolve. Map it to the real file.
      'leaflet/dist/leaflet-src.esm': resolve(
        __dirname,
        'node_modules/leaflet/dist/leaflet-src.esm.js'
      ),
    },
  },
})