// Maximum-retention repair for infeasible pin/window instances.
//
// When solve() reports infeasibility the operator may release fixed points.
// This module is a *pure analysis layer*: it never mutates the working draft,
// baseline, pin map or windows, never calls solve(), and in particular does not
// repeatedly delete a conflicting pin and re-solve.
//
// Geometry. With P[i] = sum_{k<i} duration[k] and y = pinStart - P[cueIndex],
// feasible cue starts correspond to a non-decreasing y. The start/end-of-day
// envelope gives 0 <= y <= C, C = daySpan - P[n-1]. A pin can be part of a
// joint solution only when it is inside that pin's own [earliest,latest]
// window and the global envelope. After mandatory releases, the largest
// jointly feasible set is a longest non-decreasing subsequence (LNDS) of y.
//
// Ties are broken by the complete release list (mandatory releases included,
// cue indices ascending) being lexicographically smallest, equivalent to the
// lexicographically largest retained index set.
//
// If the windows themselves admit no pin-free solution, releasing pins cannot
// help and no repair plan is generated. Complexity is O(k log k) time and
// O(k) space for k pins, plus an O(n) window scan.

import type { Cue, Pins } from './solve';

export const DAY_MS_REPAIR = 86_400_000;

export interface RepairInput {
  cues: ReadonlyArray<Cue>;
  pins?: Pins;
}

export interface RepairDetail {
  kind: 'repair';
  /** Pins outside their own window or the global duration envelope. */
  mandatoryReleased: number[];
  /** Cue indices kept, ascending — the chosen maximum-cardinality subset. */
  retained: number[];
  /** Complete release list in ascending cue-index order (mandatory + chosen). */
  released: number[];
  /** Pin map to install atomically when the repair is applied. */
  retainedPins: Map<number, number>;
  retainedCount: number;
  totalCount: number;
}

export type RepairAnalysis =
  | RepairDetail
  | {
      kind: 'unrecoverable';
      reason:
        | 'WINDOWS_INFEASIBLE'
        | 'TOO_LONG'
        | 'INVALID_WINDOW'
        | 'INVALID_PIN';
      conflictIndex?: number;
      requiredStart?: number;
      allowedLatest?: number;
    };

/** Revision identities of the state a repair plan was generated against. */
export interface RevisionId {
  draftRev: number;
  baseRev: number;
  pinsRev: number;
  windowsRev: number;
}

export interface RepairPlan extends RevisionId {
  retainedPins: Map<number, number>;
  retainedCount: number;
  totalCount: number;
  mandatoryReleased: number[];
  released: number[];
}

export type ApplyRepairResult =
  | { ok: true; pins: Map<number, number> }
  | { ok: false; reason: 'EXPIRED' };

export function sameRevision(a: RevisionId, b: RevisionId): boolean {
  return (
    a.draftRev === b.draftRev &&
    a.baseRev === b.baseRev &&
    a.pinsRev === b.pinsRev &&
    a.windowsRev === b.windowsRev
  );
}

// ---------------------------------------------------------------------------
// Segment tree with point maxima and range-max queries over value groups.
// The reconstruction merges positions into it as the remaining chain length
// drops; each leaf holds the largest position currently offered by a group.
// ---------------------------------------------------------------------------

class SegMaxRange {
  private readonly size: number;
  private readonly tree: Int32Array;

  constructor(n: number) {
    let s = 1;
    while (s < Math.max(1, n)) s <<= 1;
    this.size = s;
    this.tree = new Int32Array(2 * s).fill(-1);
  }

  set(i: number, v: number): void {
    let x = this.size + i;
    if (this.tree[x] >= v) return;
    this.tree[x] = v;
    for (x >>= 1; x > 0; x >>= 1) {
      const m = Math.max(this.tree[2 * x], this.tree[2 * x + 1]);
      if (this.tree[x] === m) break;
      this.tree[x] = m;
    }
  }

  /** Maximum over a half-open leaf range [lo, hi). */
  maxRange(lo: number, hi: number): number {
    if (lo >= hi || lo >= this.size) return -1;
    let l = this.size + lo;
    let r = this.size + Math.min(hi, this.size);
    let m = -1;
    while (l < r) {
      if (l & 1) m = Math.max(m, this.tree[l++]);
      if (r & 1) m = Math.max(m, this.tree[--r]);
      l >>= 1;
      r >>= 1;
    }
    return m;
  }
}

interface WindowEnvelope {
  prefix: number[];
  /** Tightest upper bound at i implied by i and every later window/day edge. */
  suffixUpper: number[];
  /** P[n-1]: duration consumed before the last start (the final duration need not fit past day end). */
  chainDuration: number;
  globalSlack: number;
  malformedIndex: number | null;
}

function windowOf(cue: Cue, daySpan: number): { earliest: number; latest: number } {
  return { earliest: cue.earliest ?? 0, latest: cue.latest ?? daySpan };
}

function buildEnvelope(
  cues: ReadonlyArray<Cue>,
  daySpan: number,
): WindowEnvelope | null {
  const n = cues.length;
  const prefix = new Array<number>(n);
  let malformedIndex: number | null = null;
  if (n === 0) {
    return {
      prefix,
      suffixUpper: [],
      chainDuration: 0,
      globalSlack: daySpan,
      malformedIndex: null,
    };
  }

  prefix[0] = 0;
  for (let i = 0; i + 1 < n; i++) prefix[i + 1] = prefix[i] + cues[i].duration;
  const chainDuration = prefix[n - 1];
  if (chainDuration > daySpan) return null;

  const suffixUpper = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    const w = windowOf(cues[i], daySpan);
    if (
      malformedIndex === null &&
      (!Number.isInteger(w.earliest) ||
        !Number.isInteger(w.latest) ||
        w.earliest < 0 ||
        w.latest < 0 ||
        w.earliest > daySpan ||
        w.latest > daySpan ||
        w.earliest > w.latest)
    ) {
      malformedIndex = i;
    }
    const safeLatest = Number.isInteger(w.latest)
      ? Math.max(0, Math.min(w.latest, daySpan))
      : 0;
    const own = Math.min(safeLatest, daySpan - (chainDuration - prefix[i]));
    suffixUpper[i] =
      i + 1 < n
        ? Math.min(own, suffixUpper[i + 1] - cues[i].duration)
        : own;
  }

  return {
    prefix,
    suffixUpper,
    chainDuration,
    globalSlack: daySpan - chainDuration,
    malformedIndex,
  };
}

interface WindowConflict {
  conflictIndex: number;
  requiredStart: number;
  allowedLatest: number;
}

/** Feasibility of the windows with no pins; same x-space envelope as solve(). */
function windowConflict(
  cues: ReadonlyArray<Cue>,
  env: WindowEnvelope,
): WindowConflict | null {
  const n = cues.length;
  let required = 0;
  for (let i = 0; i < n; i++) {
    const lower = Math.max(required, cues[i].earliest ?? 0);
    const cap = env.suffixUpper[i];
    if (lower > cap) {
      return { conflictIndex: i, requiredStart: lower, allowedLatest: cap };
    }
    required = lower + cues[i].duration;
  }
  return null;
}

/** Same contract as analyzeMaxRetention with a configurable day span. */
export function analyzeMaxRetentionWithSpan(
  cues: ReadonlyArray<Cue>,
  pins: Pins,
  daySpan: number,
): RepairAnalysis {
  const n = cues.length;
  const env = buildEnvelope(cues, daySpan);
  if (env === null) {
    return { kind: 'unrecoverable', reason: 'TOO_LONG' };
  }
  if (env.malformedIndex !== null) {
    return {
      kind: 'unrecoverable',
      reason: 'INVALID_WINDOW',
      conflictIndex: env.malformedIndex,
    };
  }
  if (env.chainDuration > daySpan) {
    return { kind: 'unrecoverable', reason: 'TOO_LONG' };
  }

  const conflict = windowConflict(cues, env);
  if (conflict !== null) {
    return {
      kind: 'unrecoverable',
      reason: 'WINDOWS_INFEASIBLE',
      ...conflict,
    };
  }

  // suffixUpper[i] is both the x-space cap for window feasibility and, after
  // subtracting P[i], the transformed monotone upper envelope B[i]. The lower
  // A envelope is one additional long array shared by pin filtering.
  const upperEnvelope = env.suffixUpper;
  const lowerEnvelope = new Array<number>(n);
  let runningLower = 0;
  for (let i = 0; i < n; i++) {
    runningLower = Math.max(runningLower, (cues[i].earliest ?? 0) - env.prefix[i]);
    lowerEnvelope[i] = runningLower;
  }

  interface Item {
    idx: number;
    start: number;
    y: number;
  }
  const valid: Item[] = [];
  const mandatory: number[] = [];

  for (const [idx, start] of pins) {
    const okIndex = Number.isInteger(idx) && idx >= 0 && idx < n;
    const pAt = okIndex ? env.prefix[idx] : 0;
    const inOwnWindow =
      okIndex &&
      Number.isInteger(start) &&
      start >= (cues[idx].earliest ?? 0) &&
      start <= (cues[idx].latest ?? daySpan);

    if (!okIndex || !Number.isInteger(start) || !inOwnWindow) {
      mandatory.push(idx);
    } else {
      const y = start - pAt;
      const transformedUpper = upperEnvelope[idx] - pAt;
      if (y < lowerEnvelope[idx] || y > transformedUpper) mandatory.push(idx);
      else valid.push({ idx, start, y });
    }
  }
  mandatory.sort((a, b) => a - b);
  // Pins arrive in map iteration order; the subsequence needs cue order.
  valid.sort((a, b) => a.idx - b.idx);
  const k = valid.length;

  // Forward tails pass gives the LNDS length L. The reverse pass records, for
  // every position, the LNDS length beginning there.
  const lenStart = new Int32Array(k);
  const tails: number[] = [];

  const insertUpperBound = (v: number): number => {
    let lo = 0;
    let hi = tails.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (tails[mid] > v) hi = mid;
      else lo = mid + 1;
    }
    if (lo === tails.length) tails.push(v);
    else tails[lo] = v;
    return lo + 1;
  };

  for (let i = 0; i < k; i++) insertUpperBound(valid[i].y);
  const L = tails.length;
  tails.length = 0;
  for (let i = k - 1; i >= 0; i--) lenStart[i] = insertUpperBound(-valid[i].y);

  // Value groups: positions sharing one y, groups numbered by ascending y so
  // the value constraint y >= curVal is a suffix of groups.
  const groupOf = new Int32Array(k);
  const groupId = new Map<number, number>();
  const groupY: number[] = [];
  for (let i = 0; i < k; i++) {
    const y = valid[i].y;
    let g = groupId.get(y);
    if (g === undefined) {
      g = groupY.length;
      groupId.set(y, g);
      groupY.push(y);
    }
    groupOf[i] = g;
  }
  groupY.sort((a, b) => a - b);
  const oldToNew = new Int32Array(groupY.length);
  groupY.forEach((y, pos) => oldToNew[groupId.get(y)!] = pos);
  for (let i = 0; i < k; i++) groupOf[i] = oldToNew[groupOf[i]];

  // Bucket positions by the LNDS length they begin. Reconstruction merges
  // buckets L, L-1, ..., need in descending order.
  const buckets: number[][] = Array.from({ length: L + 1 }, () => []);
  for (let i = 0; i < k; i++) buckets[lenStart[i]].push(i);

  // Greedy reconstruction of the lexicographically largest retained index set,
  // equivalent to the lexicographically smallest complete release list.
  const seg = new SegMaxRange(groupY.length);
  const chosen = new Int32Array(L);
  let curVal = -Infinity;
  for (let need = L, step = 0; need >= 1; need--, step++) {
    for (const pos of buckets[need]) seg.set(groupOf[pos], pos);
    // First group with y >= curVal.
    let lo = 0;
    let hi = groupY.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (groupY[mid] >= curVal) hi = mid;
      else lo = mid + 1;
    }
    const p = seg.maxRange(lo, groupY.length);
    if (p < 0) return { kind: 'unrecoverable', reason: 'WINDOWS_INFEASIBLE' };
    chosen[step] = p;
    curVal = valid[p].y;
  }

  const isChosen = new Uint8Array(k);
  for (let s = 0; s < L; s++) isChosen[chosen[s]] = 1;

  const retained: number[] = new Array(L);
  const retainedPins = new Map<number, number>();
  for (let s = 0; s < L; s++) {
    const item = valid[chosen[s]];
    retained[s] = item.idx;
    retainedPins.set(item.idx, item.start);
  }

  const optionalReleased: number[] = [];
  for (let i = 0; i < k; i++) {
    if (!isChosen[i]) optionalReleased.push(valid[i].idx);
  }

  // Merge two ascending lists into the complete release list.
  const released: number[] = new Array(mandatory.length + optionalReleased.length);
  let a = 0;
  let b = 0;
  while (a < mandatory.length || b < optionalReleased.length) {
    const takeA =
      b >= optionalReleased.length ||
      (a < mandatory.length && mandatory[a] <= optionalReleased[b]);
    if (takeA) released[a + b] = mandatory[a++];
    else released[a + b] = optionalReleased[b++];
  }

  return {
    kind: 'repair',
    mandatoryReleased: mandatory,
    retained,
    released,
    retainedPins,
    retainedCount: L,
    totalCount: pins.size,
  };
}

export function analyzeMaxRetention(input: RepairInput): RepairAnalysis {
  return analyzeMaxRetentionWithSpan(
    input.cues,
    input.pins ?? new Map(),
    DAY_MS_REPAIR,
  );
}

/** Freeze a generated analysis into an applicable, identity-carrying plan. */
export function buildRepairPlan(
  analysis: RepairAnalysis,
  id: RevisionId,
): RepairPlan | null {
  if (analysis.kind !== 'repair') return null;
  return {
    draftRev: id.draftRev,
    baseRev: id.baseRev,
    pinsRev: id.pinsRev,
    windowsRev: id.windowsRev,
    retainedPins: new Map(analysis.retainedPins),
    retainedCount: analysis.retainedCount,
    totalCount: analysis.totalCount,
    mandatoryReleased: analysis.mandatoryReleased.slice(),
    released: analysis.released.slice(),
  };
}

/**
 * Apply a plan only while the draft, baseline, pin and window revisions are
 * still the ones it was generated with. On mismatch the action is EXPIRED: it
 * reports the conflict and produces no partial modification. On success it
 * returns a fresh pin map for the caller to install in one replacement.
 */
export function applyRepair(
  plan: RepairPlan,
  current: RevisionId,
): ApplyRepairResult {
  if (!sameRevision(plan, current)) return { ok: false, reason: 'EXPIRED' };
  return { ok: true, pins: new Map(plan.retainedPins) };
}
