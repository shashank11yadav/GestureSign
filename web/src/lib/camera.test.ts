import { describe, expect, it } from 'vitest';
import { CAMERA_MESSAGES, CameraError, classifyCameraError } from './camera';

const domError = (name: string) => new DOMException('x', name);

describe('classifyCameraError', () => {
  it('maps getUserMedia failures to what the visitor can do about them', () => {
    expect(classifyCameraError(domError('NotAllowedError'))).toBe('denied');
    expect(classifyCameraError(domError('SecurityError'))).toBe('denied');
    expect(classifyCameraError(domError('NotFoundError'))).toBe('none');
    expect(classifyCameraError(domError('OverconstrainedError'))).toBe('none');
    expect(classifyCameraError(domError('NotReadableError'))).toBe('busy');
    expect(classifyCameraError(new TypeError('insecure'))).toBe('unsupported');
    expect(classifyCameraError(new Error('?'))).toBe('failed');
    expect(classifyCameraError(new CameraError('busy', 'x'))).toBe('busy');
  });

  it('offers the replay path in every message', () => {
    for (const message of Object.values(CAMERA_MESSAGES)) {
      expect(message.body.toLowerCase()).toContain('replay');
    }
  });
});
