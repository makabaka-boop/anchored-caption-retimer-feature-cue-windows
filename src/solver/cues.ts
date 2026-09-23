import type { Cue } from './solve';

export type ParseCuesResult =
  | { ok: true; cues: Cue[] }
  | { ok: false };

const MAX_CUES = 20_000;
const MAX_START = 86_400_000;
const MAX_DURATION = 60_000;
const MAX_TEXT = 200;

function isInt(v: unknown): v is number {
  return typeof v === 'number' && Number.isInteger(v);
}

/**
 * Strict import validation. Root object contains exactly `cues`, an array of
 * 1..20000 cue objects. Legacy cues contain exactly start/duration/text; a cue
 * may independently add integer earliest and/or latest. Any other shape, an
 * out-of-day bound or an inverted window rejects the complete import.
 */
export function parseCues(text: string): ParseCuesResult {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return { ok: false };
  }
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    return { ok: false };
  }
  const root = data as Record<string, unknown>;
  if (Object.keys(root).length !== 1 || !Array.isArray(root.cues)) {
    return { ok: false };
  }
  const raw = root.cues as unknown[];
  if (raw.length < 1 || raw.length > MAX_CUES) return { ok: false };

  const cues: Cue[] = new Array(raw.length);
  let prev = -1;
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i];
    if (typeof item !== 'object' || item === null || Array.isArray(item)) {
      return { ok: false };
    }
    const obj = item as Record<string, unknown>;
    const keys = Object.keys(obj);
    const allowed = ['start', 'duration', 'text', 'earliest', 'latest'];
    if (
      keys.length < 3 ||
      keys.length > 5 ||
      !keys.includes('start') ||
      !keys.includes('duration') ||
      !keys.includes('text') ||
      keys.some((key) => !allowed.includes(key))
    ) {
      return { ok: false };
    }

    const { start, duration, text: cueText, earliest, latest } = obj;
    if (!isInt(start) || start < 0 || start > MAX_START) return { ok: false };
    if (!isInt(duration) || duration < 1 || duration > MAX_DURATION) {
      return { ok: false };
    }
    if (typeof cueText !== 'string' || cueText.length < 1 || cueText.length > MAX_TEXT) {
      return { ok: false };
    }

    // Validate each independently before resolving omitted edges.
    if (earliest !== undefined) {
      if (!isInt(earliest) || earliest < 0 || earliest > MAX_START) return { ok: false };
    }
    if (latest !== undefined) {
      if (!isInt(latest) || latest < 0 || latest > MAX_START) return { ok: false };
    }
    if (
      (earliest ?? 0) > (latest ?? MAX_START)
    ) {
      return { ok: false };
    }

    if (start <= prev) return { ok: false }; // strictly increasing
    prev = start;

    const cue: Cue = { start, duration, text: cueText };
    if (earliest !== undefined) cue.earliest = earliest;
    if (latest !== undefined) cue.latest = latest;
    cues[i] = cue;
  }
  return { ok: true, cues };
}

export function toCuesJson(cues: ReadonlyArray<Cue>, starts: ReadonlyArray<number>): string {
  const out = cues.map((c, i) => {
    const cue: Cue = {
      start: starts[i],
      duration: c.duration,
      text: c.text,
    };
    // Preserve sparse old-style documents: emit each optional bound only when
    // the cue actually carried one.
    if (c.earliest !== undefined) cue.earliest = c.earliest;
    if (c.latest !== undefined) cue.latest = c.latest;
    return cue;
  });
  return JSON.stringify({ cues: out });
}
