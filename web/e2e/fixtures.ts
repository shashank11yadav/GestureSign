import AxeBuilder from '@axe-core/playwright';
import { test as base, expect, type Page } from '@playwright/test';
import type {} from '../src/test-hook';

/** Fails the test on any page error, console error or CSP violation. */
export const test = base.extend<{ errors: string[] }>({
  errors: async ({ page }, use) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(`console: ${m.text()}`);
    });
    await use(errors);
    expect(errors, errors.join('\n')).toEqual([]);
  },
});

export { expect };

/** Open the page and wait for the model and the test injection hook. */
export async function openReady(page: Page): Promise<void> {
  await page.goto('./');
  await page.waitForFunction(() => window.__gesturesignTest?.modelReady() === true);
}

export async function injectClip(page: Page, index: number): Promise<void> {
  await page.evaluate((i) => window.__gesturesignTest?.injectClip(i), index);
}

export async function injectPartial(page: Page, index: number): Promise<void> {
  await page.evaluate((i) => window.__gesturesignTest?.injectPartial(i), index);
}

export async function injectEmpty(page: Page, frames: number): Promise<void> {
  await page.evaluate((n) => {
    window.__gesturesignTest?.injectEmpty(n);
  }, frames);
}

/** Make getUserMedia fail the way a browser does for `name` (e.g. NotAllowedError). */
export async function failCamera(page: Page, name: string): Promise<void> {
  await page.addInitScript((errorName) => {
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: () => Promise.reject(new DOMException('blocked in test', errorName)),
      },
    });
  }, name);
}

export async function expectAccessible(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
    .analyze();
  expect(
    results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`),
  ).toEqual([]);
}
