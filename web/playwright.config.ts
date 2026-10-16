import { defineConfig, devices } from '@playwright/test';

const ci = Boolean(process.env['CI']);
const PORT = 5334;
export const BASE_URL = `http://localhost:${PORT}/GestureSign/`;

/**
 * Runs against `vite build --mode test` (npm run test:e2e builds it first):
 * the production build plus the landmark injection hook, with the CSP meta tag
 * in place. No test needs a real camera.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: ci,
  retries: ci ? 1 : 0,
  timeout: 45_000,
  expect: { timeout: 10_000 },
  reporter: ci ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'phone', use: { ...devices['Pixel 7'] }, grep: /@phone/ },
  ],
  webServer: {
    command: 'npm run preview:test',
    url: BASE_URL,
    reuseExistingServer: !ci,
    timeout: 60_000,
  },
});
