// Maximum-retention repair for infeasible pin sets.
//
// When solve() reports INFEASIBLE the operator may release fixed points. This
// module is a *pure analysis layer*: it never mutates the working draft, the
// baseline, the pin map or the current error, and it never calls solve() (in
// particular it does not repeatedly delete a conflicting pin and re-solve).
//
// Geometry. With P[i] = sum_{k<i} duration[k] and y = pinStart - P[cueIndex],
// a pin set is jointly feasible iff every pin satisfies
//   0 <= y <= C,  C = daySpan - P[n-1]
// and the y values are non-decreasing in cue-index order. If P[n-1] > daySpan
// the cues cannot fit even with no pins, so no repair exists.
//
// Pins outside the constant box are mandatory releases. Among the remaining
// pins the largest jointly feasible set is a longest non-decreasing subsequence
// (LNDS) of y. Ties are broken by the full release list (mandatory releases
// included, cue indices ascending) being lexicographically smallest, which is
// equivalent to choosing the lexicographically largest retained index set.
//
// Complexity is O(k log k) time and O(k) space for k pins (plus the O(n)
// prefix-duration scan): two tails passes plus one per-group segment tree.

import type { Cue, Pins } from './solve';

export const DAY_MS_REPAIR = 86_400_000;

export interface RepairInput {
  cues: ReadonlyArray<Cue>;
  pins?: Pins;
}

export interface RepairDetail {
  kind: 'repair';
  /** Pins that violate the per-index envelope 0 <= y <= C (or are malformed). */
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
  | { kind: 'unrecoverable' };

/** Revision identities of the state a repair plan was generated against. */
export interface RevisionId {
  draftRev: number;
  baseRev: number;
  pinsRev: number;
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
    a.pinsRev === b.pinsRev
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

/** Same contract as analyzeMaxRetention with a configurable day span. */
export function analyzeMaxRetentionWithSpan(
  cues: ReadonlyArray<Cue>,
  pins: Pins,
  daySpan: number,
): RepairAnalysis {
  const n = cues.length;

  // Prefix durations P[i] = sum_{k < i} duration[k].
  const P = new Array<number>(n);
  if (n > 0) {
    P[0] = 0;
    for (let i = 0; i + 1 < n; i++) P[i + 1] = P[i] + cues[i].duration;
  }

  if (n > 0 && P[n - 1] > daySpan) {
    // The cues themselves do not fit in a day; releasing pins cannot help and
    // no plan is generated.
    return { kind: 'unrecoverable' };
  }

  const C = n === 0 ? daySpan : daySpan - P[n - 1];

  interface Item {
    idx: number;
    start: number;
    y: number;
  }
  const valid: Item[] = [];
  const mandatory: number[] = [];

  for (const [idx, start] of pins) {
    const okIndex = Number.isInteger(idx) && idx >= 0 && idx < n;
    const pAt = okIndex ? P[idx] : 0;
    if (
      !okIndex ||
      !Number.isInteger(start) ||
      start - pAt < 0 ||
      start - pAt > C
    ) {
      // Outside the constant feasible box (or structurally invalid): this pin
      // must be released no matter which subset survives.
      mandatory.push(idx);
    } else {
      valid.push({ idx, start, y: start - P[idx] });
    }
  }
  mandatory.sort((a, b) => a - b);
  // Pins arrive in map iteration order; the subsequence needs cue order.
  valid.sort((a, b) => a.idx - b.idx);
  const k = valid.length;

  // Forward tails pass gives the LNDS length L. The reverse pass records, for
  // every position, the LNDS length beginning there — also a non-decreasing
  // subsequence on the reversed negated values, i.e. LNDS of -y with an
  // upper-bound tails.
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

  // Bucket positions by the LNDS length they begin. When the reconstruction
  // needs a remaining length `need`, every position with lenStart >= need is
  // eligible — merging buckets L, L-1, ..., need in descending order.
  const buckets: number[][] = Array.from({ length: L + 1 }, () => []);
  for (let i = 0; i < k; i++) buckets[lenStart[i]].push(i);

  // Greedy reconstruction of the lexicographically largest retained index set,
  // equivalent to the lexicographically smallest complete release list (the
  // mandatory entries are identical across every tie).
  //
  // At remaining length `need` the usable positions are exactly those already
  // merged (lenStart >= need) with group >= firstGroup (y >= curVal). Each
  // group leaf stores its largest merged position, so the range maximum over
  // the suffix both certifies feasibility and returns the rightmost usable
  // position in O(log k). O(k log k) time, O(k) space overall.
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
    if (p < 0) return { kind: 'unrecoverable' };
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
