/** Checks shared by the two workers' message handlers. */

/**
 * A dedicated worker only receives messages from the page that created it,
 * and those arrive with an empty origin. Anything else (which should never
 * happen) is ignored.
 */
export function isTrustedOrigin(origin: string, ownOrigin: string): boolean {
  return origin === '' || origin === ownOrigin;
}

export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function isCount(v: unknown, max: number): v is number {
  return typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= max;
}

/** The result of checking a message sent to a worker. */
export type Parsed<In, Out> =
  | { ok: true; message: In }
  /** `reply` tells the page its request failed, when the message says which request it was. */
  | { ok: false; reply: Out | null };
