// Maximum-retention repair for infeasible pin/window systems.
//
// When solve() reports an infeasible pin set the operator can release fixed
// points. This module is a *pure analysis layer*: it never mutates the working
// draft, baseline, pin map or current error, and it never calls solve().
//
// Geometry. Let P[i] = Σ_{k<i} duration[k] and y[i] = x[i] - P[i]. Optional
// windows become transformed boxes low[i] <= y[i] <= high[i]. Close them under
// the non-decreasing chain:
//   L[i] = max_{j<=i} low[j]
//   R[i] = min_{j>=i} high[j].
// The windows themselves are jointly feasible iff L[i] <= R[i] for every i.
// If they are not, releasing pins cannot help and no plan is generated.
//
// With a feasible window snapshot, a pin at i with transformed value y is
// keepable by itself exactly when L[i] <= y <= R[i]. Any two keepable pins
// p<q are jointly compatible iff y[p] <= y[q]; the closures guarantee the
// cues between them can be filled. Thus maximum retention is an LNDS of y
// among keepable pins. Ties use the complete ascending release list, which is
// equivalent to the lexicographically largest retained index set.
//
// Complexity is O(n + k log k) time and O(n + k) space.

import type { Cue, FeasibilityConflict, Pins } from './solve';

export const DAY_MS_REPAIR = 86_400_000;

export interface RepairInput {
  cues: ReadonlyArray<Cue>;
  pins?: Pins;
}

export interface RepairDetail {
  kind: 'repair';
  /** Pins that violate even the propagated window envelope (or malformed). */
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
  | { kind: 'unrecoverable'; conflict: FeasibilityConflict };

/** Revision identities of the state a repair plan was generated against. */
export interface RevisionId {
  draftRev: number;
  baseRev: number;
  pinsRev: number;
  windowRev: number;
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
    a.windowRev === b.windowRev
  );
}

// ---------------------------------------------------------------------------
// Segment tree with point maxima and range-max queries over value groups.
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
    this.tree[x] = v;
    for (x >>= 1; x > 0; x >>= 1) {
      this.tree[x] = Math.max(this.tree[2 * x], this.tree[2 * x + 1]);
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

/** Same contract as analyzeMaxRetention with a configurable day span. */
export function analyzeMaxRetentionWithSpan(
  cues: ReadonlyArray<Cue>,
  pins: Pins,
  daySpan: number,
): RepairAnalysis {
  const n = cues.length;

  // Prefix durations P[i] = sum_{k < i} duration[k]. Keep the legacy no-window
  // path allocation-light so the full-pin performance case stays O(k log k)
  // without three extra n-sized arrays.
  const P = new Array<number>(n);
  let hasWindow = false;
  let rawLow: number[] | null = null;
  let rawHigh: number[] | null = null;
  let prefixSum = 0;
  for (let i = 0; i < n; i++) {
    P[i] = prefixSum;
    const cue = cues[i];
    if (cue.earliest !== undefined || cue.latest !== undefined) {
      if (!hasWindow) {
        hasWindow = true;
        rawLow = new Array<number>(n).fill(0);
        rawHigh = new Array<number>(n).fill(daySpan);
      }
    }
    const lo = cue.earliest === undefined ? 0 : cue.earliest;
    const hi = cue.latest === undefined ? daySpan : cue.latest;
    if (
      !Number.isInteger(lo) ||
      !Number.isInteger(hi) ||
      lo < 0 ||
      hi < lo ||
      hi > daySpan
    ) {
      return {
        kind: 'unrecoverable',
        conflict: {
          cueIndex: i,
          requiredStart: lo,
          allowedUpperBound: hi,
        },
      };
    }
    if (rawLow !== null && rawHigh !== null) {
      rawLow[i] = lo - prefixSum;
      rawHigh[i] = hi - prefixSum;
    }
    prefixSum += cue.duration;
  }

  let closedLow: number[] | Int32Array;
  let closedHigh: number[] | Int32Array;
  if (!hasWindow) {
    if (n > 0 && P[n - 1] > daySpan) {
      return {
        kind: 'unrecoverable',
        conflict: {
          cueIndex: n - 1,
          requiredStart: P[n - 1],
          allowedUpperBound: daySpan,
        },
      };
    }
    // The monotone transformed chain has the constant box [0, C].
    const C = n === 0 ? daySpan : daySpan - P[n - 1];
    closedLow = new Int32Array(n);
    closedHigh = new Int32Array(n).fill(C);
  } else {
    const low = rawLow!;
    const high = rawHigh!;
    let requiredY = 0;
    for (let i = 0; i < n; i++) {
      requiredY = Math.max(requiredY, low[i]);
      if (requiredY > high[i]) {
        return {
          kind: 'unrecoverable',
          conflict: {
            cueIndex: i,
            requiredStart: requiredY + P[i],
            allowedUpperBound: high[i] + P[i],
          },
        };
      }
    }
    // Close bounds under monotonicity, reusing the raw arrays.
    for (let i = 1; i < n; i++) {
      low[i] = Math.max(low[i - 1], low[i]);
    }
    for (let i = n - 2; i >= 0; i--) {
      high[i] = Math.min(high[i + 1], high[i]);
    }
    closedLow = low;
    closedHigh = high;
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
    if (
      !okIndex ||
      !Number.isInteger(start) ||
      start < 0 ||
      start > daySpan
    ) {
      mandatory.push(idx);
      continue;
    }
    const y = start - P[idx];
    if (y < closedLow[idx] || y > closedHigh[idx]) {
      mandatory.push(idx);
    } else {
      valid.push({ idx, start, y });
    }
  }
  mandatory.sort((a, b) => a - b);
  valid.sort((a, b) => a.idx - b.idx);
  const k = valid.length;

  // Forward/backward tail lengths for the LNDS.
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

  // Number equal-y groups in ascending order.
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

  const buckets: number[][] = Array.from({ length: L + 1 }, () => []);
  for (let i = 0; i < k; i++) buckets[lenStart[i]].push(i);

  // Greedily reconstruct the lexicographically largest retained index set.
  const seg = new SegMaxRange(groupY.length);
  const chosen = new Int32Array(L);
  let curVal = -Infinity;
  for (let need = L, step = 0; need >= 1; need--, step++) {
    for (const pos of buckets[need]) seg.set(groupOf[pos], pos);
    let lo = 0;
    let hi = groupY.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (groupY[mid] >= curVal) hi = mid;
      else lo = mid + 1;
    }
    const p = seg.maxRange(lo, groupY.length);
    if (p < 0) {
      return {
        kind: 'unrecoverable',
        conflict: { cueIndex: -1, requiredStart: 0, allowedUpperBound: -1 },
      };
    }
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
    windowRev: id.windowRev,
    retainedPins: new Map(analysis.retainedPins),
    retainedCount: analysis.retainedCount,
    totalCount: analysis.totalCount,
    mandatoryReleased: analysis.mandatoryReleased.slice(),
    released: analysis.released.slice(),
  };
}

/**
 * Apply a plan only while the draft, baseline and pin revisions are still the
 * ones it was generated with. On mismatch the action is EXPIRED: it reports
 * the conflict and produces no partial modification. On success it returns a
 * fresh pin map for the caller to install in one replacement.
 */
export function applyRepair(
  plan: RepairPlan,
  current: RevisionId,
): ApplyRepairResult {
  if (!sameRevision(plan, current)) return { ok: false, reason: 'EXPIRED' };
  return { ok: true, pins: new Map(plan.retainedPins) };
}
