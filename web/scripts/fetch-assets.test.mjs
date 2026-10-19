import { createHash } from 'node:crypto';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { downloadVerified, writeAtomic } from './fetch-assets.mjs';

const body = Buffer.from('a pinned landmark model');
const pinned = {
  file: 'test.task',
  url: 'https://storage.example/test.task',
  sha256: createHash('sha256').update(body).digest('hex'),
  bytes: body.length,
};

/** A fetch stand-in that serves `bytes` in two chunks. */
function serve(bytes, status = 200) {
  return () =>
    Promise.resolve(
      new Response(
        new ReadableStream({
          start(controller) {
            const half = Math.floor(bytes.length / 2);
            controller.enqueue(new Uint8Array(bytes.subarray(0, half)));
            controller.enqueue(new Uint8Array(bytes.subarray(half)));
            controller.close();
          },
        }),
        { status },
      ),
    );
}

describe('downloadVerified', () => {
  it('returns the body when its size and SHA-256 match the pins', async () => {
    expect(await downloadVerified(pinned, serve(body))).toEqual(body);
  });

  it('rejects a body with the right size but the wrong hash', async () => {
    const tampered = Buffer.from(body);
    tampered[0] ^= 1;
    await expect(downloadVerified(pinned, serve(tampered))).rejects.toThrow(/SHA-256/);
  });

  it('stops reading once the body is larger than pinned', async () => {
    const big = Buffer.concat([body, Buffer.alloc(1024)]);
    await expect(downloadVerified(pinned, serve(big))).rejects.toThrow(/larger than/);
  });

  it('rejects a body shorter than pinned', async () => {
    await expect(downloadVerified(pinned, serve(body.subarray(1)))).rejects.toThrow(/bytes/);
  });

  it('reports an HTTP error', async () => {
    await expect(downloadVerified(pinned, serve(body, 404))).rejects.toThrow(/HTTP 404/);
  });
});

describe('writeAtomic', () => {
  let dir;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'fetch-assets-'));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('replaces the destination and leaves no temporary file behind', async () => {
    const dest = join(dir, 'model.task');
    await writeFile(dest, 'old');
    await writeAtomic(dest, body);
    expect(await readFile(dest)).toEqual(body);
    expect(await readdir(dir)).toEqual(['model.task']);
  });
});
