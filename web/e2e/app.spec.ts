import {
  expect,
  expectAccessible,
  failCamera,
  injectClip,
  injectEmpty,
  injectPartial,
  openReady,
  test,
} from './fixtures';

test.describe('page', () => {
  test('loads with the untrained-model notice, a strict CSP and no a11y violations @phone', async ({
    page,
    errors,
  }) => {
    await openReady(page);
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Sign to your camera');
    // The notice is shown exactly while the deployed model is the untrained stub.
    const meta = (await (await page.request.get('models/model-meta.json')).json()) as {
      metrics: unknown;
    };
    if (meta.metrics === null) {
      await expect(page.getByTestId('untrained-notice')).toContainText('Untrained preview model');
    } else {
      await expect(page.getByTestId('untrained-notice')).toHaveCount(0);
    }
    const csp = await page
      .locator('meta[http-equiv="Content-Security-Policy"]')
      .getAttribute('content');
    expect(csp).toContain("script-src 'self' 'wasm-unsafe-eval'");
    expect(csp).not.toContain("'unsafe-eval'");
    expect(csp).not.toContain("'unsafe-inline'");
    await expect(page.getByRole('button', { name: 'Start camera' })).toBeVisible();
    await expectAccessible(page);
    expect(errors).toEqual([]);
  });

  test('mode tabs work from the keyboard', async ({ page, errors }) => {
    await openReady(page);
    await page.getByRole('tab', { name: 'Recognise' }).focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('tab', { name: 'Practice' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    await expect(page.getByRole('tab', { name: 'Practice' })).toBeFocused();
    await page.keyboard.press('End');
    await expect(page.getByRole('tab', { name: 'Teach your own' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('tab', { name: 'Recognise' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    expect(errors).toEqual([]);
  });
});

test.describe('start-up', () => {
  test('downloads the landmark and sign models on page load, once each, with no WebGPU runtime', async ({
    page,
    errors,
  }) => {
    const requested: string[] = [];
    const origins = new Set<string>();
    page.context().on('request', (request) => {
      const url = new URL(request.url());
      requested.push(url.pathname);
      if (url.protocol !== 'data:' && url.protocol !== 'blob:') origins.add(url.origin);
    });
    await openReady(page);
    // MediaPipe starts before any click, so the camera's first frame is tracked at once.
    await expect
      .poll(() => requested.filter((p) => p.includes('/vendor/mediapipe/models/')).length)
      .toBeGreaterThan(0);
    await expect(page.getByRole('button', { name: 'Start camera' })).toBeVisible();
    const count = (suffix: string) => requested.filter((p) => p.endsWith(suffix)).length;
    expect(count('/models/model.onnx')).toBe(1);
    expect(count('/vendor/ort/ort-wasm-simd-threaded.wasm')).toBe(1);
    expect(requested.filter((p) => /jsep|asyncify|webgpu/i.test(p))).toEqual([]);
    // Every file comes from the site itself: no CDN, font host or analytics.
    expect([...origins]).toEqual([new URL(page.url()).origin]);
    expect(errors).toEqual([]);
  });
});

test.describe('camera problems', () => {
  const cases = [
    ['NotAllowedError', 'Camera access was blocked'],
    ['NotFoundError', 'No camera found'],
    ['NotReadableError', 'The camera is in use'],
  ] as const;
  for (const [name, message] of cases) {
    test(`${name} shows "${message}" and offers Replay @phone`, async ({ page, errors }) => {
      await failCamera(page, name);
      await openReady(page);
      await page.getByRole('button', { name: 'Start camera' }).click();
      const alert = page.getByTestId('camera-problem');
      await expect(alert).toContainText(message);
      await expect(alert.getByRole('button', { name: 'Replay a clip instead' })).toBeVisible();
      await expectAccessible(page);

      // The no-camera path works end to end.
      await alert.getByRole('button', { name: 'Replay a clip instead' }).click();
      await page.getByRole('list', { name: 'Replay clips' }).getByRole('button').first().click();
      await expect(page.getByTestId('recognise-result')).toContainText(/Not sure|You signed/, {
        timeout: 15_000,
      });
      await expect(page.getByTestId('history').getByRole('listitem')).toHaveCount(1);
      expect(errors).toEqual([]);
    });
  }
});

test.describe('model loading', () => {
  test('a missing model file shows an error with Retry, and Retry recovers', async ({ page }) => {
    await page.route('**/models/model.onnx', (route) => route.fulfill({ status: 404, body: '' }));
    await page.goto('./');
    const alert = page.getByRole('alert').filter({ hasText: 'The sign model did not load' });
    await expect(alert).toContainText('HTTP 404');
    await page.unroute('**/models/model.onnx');
    await alert.getByRole('button', { name: 'Retry' }).click();
    await expect(page.getByText('Your result appears here after each sign.')).toBeVisible();
    await page.waitForFunction(() => window.__gesturesignTest?.modelReady() === true);
    await injectClip(page, 0);
    await expect(page.getByTestId('history').getByRole('listitem')).toHaveCount(1);
  });
});

test.describe('recognise', () => {
  test('a signed clip gives a result, top-3 guesses and a history entry', async ({
    page,
    errors,
  }) => {
    await openReady(page);
    await injectClip(page, 0);
    const result = page.getByTestId('recognise-result');
    await expect(result).toContainText(/Not sure|You signed/);
    await expect(page.getByRole('list', { name: 'Top guesses' }).getByRole('listitem')).toHaveCount(
      3,
    );
    await injectClip(page, 3);
    await expect(page.getByTestId('history').getByRole('listitem')).toHaveCount(2);
    await expectAccessible(page);
    expect(errors).toEqual([]);
  });

  test('nothing in frame says so and produces no result', async ({ page, errors }) => {
    await openReady(page);
    await injectEmpty(page, 40);
    await expect(page.getByTestId('tracking-status')).toHaveText('No one in view');
    await expect(page.getByTestId('recognise-result')).toHaveText(
      'Your result appears here after each sign.',
    );
    expect(errors).toEqual([]);
  });

  test('a partly visible signer gets "Not sure" with a hint', async ({ page, errors }) => {
    await openReady(page);
    await injectPartial(page, 1);
    const result = page.getByTestId('recognise-result');
    await expect(result).toContainText('Not sure');
    await expect(result).toContainText('shoulders');
    expect(errors).toEqual([]);
  });
});

test.describe('practice', () => {
  test('an attempt is scored against the target sign @phone', async ({ page, errors }) => {
    await openReady(page);
    await page.getByRole('tab', { name: 'Practice' }).click();
    await expect(page.getByTestId('practice-target')).not.toBeEmpty();
    const link = page.getByRole('link', { name: /How to sign it/ });
    await expect(link).toHaveAttribute('href', /^https:\/\/www\.signingsavvy\.com\/search\//);
    await expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    await injectClip(page, 2);
    const score = page.getByTestId('practice-score');
    await expect(score).toContainText('%');
    await expect(score).toContainText('Attempt 1');
    await page.getByRole('button', { name: 'Another sign' }).click();
    await expect(score).toContainText('to get a score');
    await expectAccessible(page);
    expect(errors).toEqual([]);
  });
});

test.describe('teach your own', () => {
  test('records takes, recognises the gesture, keeps it after reload and clears it', async ({
    page,
    errors,
  }) => {
    await openReady(page);
    await page.getByRole('tab', { name: 'Teach your own' }).click();
    const status = page.getByTestId('teach-status');
    const list = page.getByRole('list', { name: 'Your gestures' });

    for (const [name, clip] of [
      ['wave', 0],
      ['circle', 3],
    ] as const) {
      await page.getByLabel('New gesture').fill(name);
      await page.getByRole('button', { name: 'Add gesture' }).click();
      await expect(status).toContainText(`Recording “${name}”: take 1`);
      for (let take = 0; take < 3; take++) await injectClip(page, clip);
      await expect(list.getByRole('listitem').filter({ hasText: name })).toContainText('3/5 takes');
    }

    await injectClip(page, 0);
    await expect(status).toContainText('Recognised');
    await expect(status).toContainText('wave');
    await injectClip(page, 3);
    await expect(status).toContainText('circle');
    await expectAccessible(page);

    // Stored on this device: survives a reload.
    await page.reload();
    await page.waitForFunction(() => window.__gesturesignTest?.modelReady() === true);
    await page.getByRole('tab', { name: 'Teach your own' }).click();
    await expect(list.getByRole('listitem')).toHaveCount(2);

    await page.getByRole('button', { name: 'Clear all' }).click();
    await page.getByRole('button', { name: 'Delete all' }).click();
    await expect(list).toHaveCount(0);
    await page.reload();
    await page.waitForFunction(() => window.__gesturesignTest?.modelReady() === true);
    await page.getByRole('tab', { name: 'Teach your own' }).click();
    await expect(list).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  test('rejects duplicate and empty names', async ({ page }) => {
    await openReady(page);
    await page.getByRole('tab', { name: 'Teach your own' }).click();
    await page.getByRole('button', { name: 'Add gesture' }).click();
    await expect(page.getByText('Give the gesture a name.')).toBeVisible();
    await page.getByLabel('New gesture').fill('wave');
    await page.getByRole('button', { name: 'Add gesture' }).click();
    await page.getByLabel('New gesture').fill('Wave');
    await page.getByRole('button', { name: 'Add gesture' }).click();
    await expect(page.getByText('You already have a gesture with that name.')).toBeVisible();
  });
});
