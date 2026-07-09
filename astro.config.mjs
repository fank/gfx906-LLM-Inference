import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://fank.github.io',
  base: '/gfx906-LLM-Inference',
  integrations: [
    starlight({
      title: 'MI50 LLM Inference',
      description: 'LLM inference benchmarks, tuning results, and production notes for AMD MI50 (gfx906 / Vega 20) on ROCm',
      favicon: '/favicon.svg',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/fank/gfx906-LLM-Inference' },
      ],
      customCss: ['./src/styles/custom.css'],
      editLink: {
        baseUrl: 'https://github.com/fank/gfx906-LLM-Inference/edit/main/src/content/docs/',
      },
      sidebar: [
        {
          label: 'Overview',
          items: [
            { label: 'Home', slug: '' },
            { label: 'Master Report (2026-07-06)', slug: 'master-report' },
            { label: 'Test Log (2026-07-03)', slug: 'all-tests-log' },
          ],
        },
        {
          label: 'Individual Runs (2026-07-06)',
          autogenerate: { directory: 'individual-runs' },
        },
        {
          label: 'Older Reports',
          autogenerate: { directory: 'reports' },
        },
      ],
    }),
  ],
});
