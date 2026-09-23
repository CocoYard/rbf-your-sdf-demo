import { defineConfig } from 'vite';

export default defineConfig({
  // Relative base so the static build works from any subdirectory (e.g. GitHub Pages).
  base: './',
  worker: { format: 'es' },
});
