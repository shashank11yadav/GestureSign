import { Camera, CameraOff, Film, Hand, LoaderCircle, Play, RotateCcw, Square } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { Session, SessionSnapshot } from '../app/session';
import { CAMERA_MESSAGES } from '../lib/camera';
import { drawLandmarks } from '../lib/overlay';
import { handsMissing } from '../ml/landmarks';
import type { FrameStatus } from '../ml/segmenter';
import { ReplayPicker } from './ReplayPicker';
import { Button } from './ui';

/**
 * Replay clips come from asl-signs, which was filmed on phones held upright,
 * so their landmarks are drawn in a portrait 9:16 box to keep the signer's
 * proportions (and the whole frame) on the stage.
 */
const REPLAY_ASPECT = 9 / 16;

const STATUS_TEXT: Record<FrameStatus, string> = {
  nothing: 'No one in view',
  'no-hands': 'Hands not in view',
  partial: 'Move back: shoulders out of view',
  crowded: 'More than one person in view',
  ok: 'Tracking',
};

function StatusPill({ snap }: { snap: SessionSnapshot }) {
  if (snap.source === 'off' || snap.source === 'starting' || snap.replay?.done) return null;
  const signing = snap.signing;
  const status = snap.status;
  const text = signing ? 'Reading sign…' : status ? STATUS_TEXT[status] : 'Waiting for landmarks…';
  const tone = signing
    ? 'bg-accent text-accent-ink'
    : status === 'ok'
      ? 'bg-emerald-500/90 text-white'
      : 'bg-black/55 text-white';
  return (
    <p
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium whitespace-nowrap shadow backdrop-blur ${tone}`}
      data-testid="tracking-status"
    >
      <span
        aria-hidden
        className={`size-2 rounded-full bg-current ${signing ? 'pulse-dot' : ''}`}
      />
      {text}
    </p>
  );
}

export function Stage({ session, snap }: { session: Session; snap: SessionSnapshot }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showReplay, setShowReplay] = useState(false);

  useEffect(() => {
    session.attachVideo(videoRef.current);
    return () => {
      session.attachVideo(null);
    };
  }, [session]);

  // Keep the canvas backing store at the element's size times the pixel ratio.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(canvas.clientWidth * ratio);
      canvas.height = Math.round(canvas.clientHeight * ratio);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    let pending: Float32Array | null = null;
    let raf = 0;
    const off = session.onFrame((frame, source) => {
      // Replays start and end with hands down (and a tail of empty frames so the
      // recogniser sees the sign end); keep the last signing pose on screen.
      if (source !== 'camera' && handsMissing(frame)) return;
      pending = frame;
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const ctx = canvasRef.current?.getContext('2d');
        const video = videoRef.current;
        if (!ctx || !pending) return;
        const aspect =
          source === 'camera' && video && video.videoWidth
            ? video.videoWidth / video.videoHeight
            : REPLAY_ASPECT;
        drawLandmarks(ctx, pending, undefined, { aspect });
      });
    });
    return () => {
      off();
      cancelAnimationFrame(raf);
    };
  }, [session]);

  // Clear the overlay when the source stops.
  useEffect(() => {
    if (snap.source === 'off') {
      const canvas = canvasRef.current;
      canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [snap.source]);

  const live = snap.source === 'camera';
  const replaying = snap.source === 'replay' || snap.source === 'injected';
  const problem = snap.cameraProblem ? CAMERA_MESSAGES[snap.cameraProblem] : null;

  const openReplay = () => {
    session.dismissCameraProblem();
    setShowReplay(true);
  };

  return (
    <section aria-labelledby="stage-heading" className="min-w-0">
      <h2 id="stage-heading" className="sr-only">
        Camera and landmarks
      </h2>
      <div className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl bg-stage text-stage-ink shadow-card ring-1 ring-black/5 dark:ring-white/10">
        <video
          ref={videoRef}
          className={`mirrored absolute inset-0 h-full w-full object-contain ${live ? '' : 'invisible'}`}
          muted
          playsInline
          aria-hidden
        />
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 h-full w-full ${live ? 'mirrored' : ''}`}
          role="img"
          aria-label="Tracked hands and upper body"
        />
        {!live && !replaying ? (
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgb(79_70_229/0.28),transparent_60%)]" />
        ) : null}

        <div className="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-2">
          <StatusPill snap={snap} />
          {live ? (
            <span className="hidden rounded-full bg-black/55 px-2.5 py-1 text-xs font-medium text-white backdrop-blur sm:inline">
              Live · on this device
            </span>
          ) : replaying && snap.replay ? (
            <span className="hidden max-w-[55%] truncate rounded-full bg-black/55 px-2.5 py-1 text-xs font-medium text-white backdrop-blur sm:inline">
              Replay · {snap.replay.clip.title}
            </span>
          ) : null}
        </div>

        {snap.source === 'off' && !problem ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center">
            <div className="grid size-14 place-items-center rounded-2xl bg-white/10 ring-1 ring-white/15">
              <Hand aria-hidden className="size-7" />
            </div>
            <div>
              <p className="text-lg font-semibold text-white sm:text-xl">
                {snap.paused ? 'Camera paused' : 'Sign to your camera'}
              </p>
              <p className="mx-auto mt-1 max-w-sm text-sm text-white/70">
                {snap.paused
                  ? 'The camera stopped when this tab was hidden.'
                  : 'Video stays on this device. Only hand and body landmarks are computed, in your browser.'}
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <Button
                variant="primary"
                size="lg"
                onClick={() => {
                  void session.startCamera();
                }}
              >
                <Camera aria-hidden className="size-5" />
                {snap.paused ? 'Resume camera' : 'Start camera'}
              </Button>
              <Button variant="onStage" size="lg" onClick={openReplay}>
                <Film aria-hidden className="size-5" />
                Replay a clip
              </Button>
            </div>
          </div>
        ) : null}

        {snap.source === 'starting' ? (
          <div
            role="status"
            className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-sm text-white/80"
          >
            <LoaderCircle aria-hidden className="size-7 animate-spin" />
            Waiting for the camera…
          </div>
        ) : null}

        {problem ? (
          <div className="absolute inset-0 flex items-center justify-center p-4 sm:p-6">
            <div
              role="alert"
              className="w-full max-w-md rounded-xl bg-white/95 p-4 text-sm text-zinc-900 shadow-xl sm:p-5"
              data-testid="camera-problem"
            >
              <p className="flex items-center gap-2 text-base font-semibold">
                <CameraOff aria-hidden className="size-5 text-red-700" />
                {problem.title}
              </p>
              <p className="mt-1.5 leading-relaxed text-zinc-700">{problem.body}</p>
              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  className="inline-flex h-10 items-center gap-2 rounded-lg bg-indigo-700 px-4 font-medium text-white hover:bg-indigo-600"
                  onClick={openReplay}
                >
                  <Film aria-hidden className="size-4" />
                  Replay a clip instead
                </button>
                <button
                  type="button"
                  className="inline-flex h-10 items-center gap-2 rounded-lg border border-zinc-300 px-4 font-medium hover:bg-zinc-100"
                  onClick={() => {
                    void session.startCamera();
                  }}
                >
                  <RotateCcw aria-hidden className="size-4" />
                  Try again
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {snap.replay?.done && snap.source === 'replay' ? (
          <div className="absolute inset-x-0 bottom-0 flex justify-center p-3">
            <Button
              variant="onStage"
              onClick={() => {
                if (snap.replay) session.playReplay(snap.replay.clip);
              }}
            >
              <RotateCcw aria-hidden className="size-4" />
              Clip finished · play again
            </Button>
          </div>
        ) : null}

        {snap.trackerError && live ? (
          <p
            role="alert"
            className="absolute inset-x-3 bottom-3 rounded-lg bg-black/70 px-3 py-2 text-sm text-white"
          >
            {snap.trackerError}
          </p>
        ) : null}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {live ? (
          <Button
            onClick={() => {
              session.stopCamera();
            }}
          >
            <Square aria-hidden className="size-4" />
            Stop camera
          </Button>
        ) : null}
        {replaying ? (
          <>
            <Button
              onClick={() => {
                session.stopReplay();
              }}
            >
              <Square aria-hidden className="size-4" />
              Stop replay
            </Button>
            {snap.replay ? (
              <Button
                onClick={() => {
                  if (snap.replay) session.playReplay(snap.replay.clip);
                }}
              >
                <Play aria-hidden className="size-4" />
                Play again
              </Button>
            ) : null}
          </>
        ) : null}
        {live || replaying ? (
          <Button
            variant="ghost"
            onClick={() => {
              setShowReplay((v) => !v);
            }}
            aria-expanded={showReplay}
          >
            <Film aria-hidden className="size-4" />
            {showReplay ? 'Hide clips' : 'Replay clips'}
          </Button>
        ) : null}
      </div>

      {showReplay ? (
        <ReplayPicker
          active={snap.replay?.clip.id ?? null}
          onPlay={(clip) => {
            session.playReplay(clip);
            // On phones the list sits between the stage and the results; get it out of the way.
            if (window.matchMedia('(max-width: 1023px)').matches) setShowReplay(false);
          }}
          onClose={() => {
            setShowReplay(false);
          }}
        />
      ) : null}
    </section>
  );
}
