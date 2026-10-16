/**
 * Camera access. It is only requested after a click, and every failure is
 * mapped to a message plus the no-camera Replay path, so the page never goes
 * blank because a camera is missing, blocked or busy.
 */

export type CameraProblem = 'denied' | 'none' | 'busy' | 'unsupported' | 'failed';

export class CameraError extends Error {
  override name = 'CameraError';
  constructor(
    readonly problem: CameraProblem,
    message: string,
  ) {
    super(message);
  }
}

export const CAMERA_MESSAGES: Record<CameraProblem, { title: string; body: string }> = {
  denied: {
    title: 'Camera access was blocked',
    body: 'Allow the camera in your browser’s site settings and try again, or replay a recorded clip instead.',
  },
  none: {
    title: 'No camera found',
    body: 'Connect a camera and try again, or replay a recorded clip instead.',
  },
  busy: {
    title: 'The camera is in use',
    body: 'Another app or tab is using the camera. Close it and try again, or replay a recorded clip instead.',
  },
  unsupported: {
    title: 'This browser cannot share a camera here',
    body: 'Camera access needs a secure (https) page in a current browser. You can still replay a recorded clip.',
  },
  failed: {
    title: 'The camera could not start',
    body: 'Try again, or replay a recorded clip instead.',
  },
};

export function classifyCameraError(error: unknown): CameraProblem {
  if (error instanceof CameraError) return error.problem;
  const name = error instanceof Error ? error.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return 'denied';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return 'none';
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return 'busy';
    case 'TypeError':
      return 'unsupported';
    default:
      return 'failed';
  }
}

export async function openCamera(): Promise<MediaStream> {
  if (!('mediaDevices' in navigator) || typeof navigator.mediaDevices.getUserMedia !== 'function') {
    throw new CameraError('unsupported', 'getUserMedia is not available');
  }
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
      },
    });
  } catch (error) {
    const problem = classifyCameraError(error);
    throw new CameraError(problem, error instanceof Error ? error.message : String(error));
  }
}

export function stopStream(stream: MediaStream | null): void {
  for (const track of stream?.getTracks() ?? []) track.stop();
}
