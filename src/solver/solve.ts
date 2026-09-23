// Solver: lock cues at fixed starts, keep each cue inside an optional edit
// window, and remove all adjacent overlaps.
//
// Variables x[i] are integer cue starts. The constraints are
//   0 <= x[i] <= daySpan
//   x[i+1] >= x[i] + duration[i]
//   earliest[i] <= x[i] <= latest[i]
//   pins force x[i] = start exactly.
// Objective: minimise sum |x[i] - base[i]|; ties use the lexicographically
// smallest complete start vector.
//
// With P[i] = sum_{k<i} duration[k] and y[i] = x[i] - P[i], the gap
// constraints become a non-decreasing y sequence. The day and edit windows
// become per-index intervals, so the problem is bounded L1 isotonic regression.
// PAVA pools adjacent fitted blocks; each block value is the lower weighted
// median of its raw targets, projected into the intersection of its member
// intervals. Weighted medians are maintained with meldable leftist heaps,
// giving O(n log n) time.

export const DAY_MS = 86_400_000;

export interface Cue {
  start: number;
  duration: number;
  text: string;
  /** Optional inclusive lower cut-in bound; absent means 0. */
  earliest?: number;
  /** Optional inclusive upper cut-in bound; absent means the end of day. */
  latest?: number;
}

/** cueIndex -> fixed integer start; one pin per cue, re-edit overwrites. */
export type Pins = ReadonlyMap<number, number>;

export interface SolveInput {
  cues: ReadonlyArray<Cue>;
  base: ReadonlyArray<number>;
  pins?: Pins;
}

export type InfeasibleResult =
  | {
      ok: false;
      reason: 'PIN_OUTSIDE_WINDOW';
      conflictIndex: number;
      requiredStart: number;
      allowedEarliest: number;
      allowedLatest: number;
    }
  | {
      ok: false;
      reason: 'WINDOW_CHAIN';
      conflictIndex: number;
      requiredStart: number;
      allowedLatest: number;
    }
  | {
      ok: false;
      reason: 'INVALID_WINDOW' | 'INVALID_PIN' | 'INFEASIBLE';
      conflictIndex?: number;
    };

export type SolveResult =
  | { ok: true; starts: number[]; cost: number }
  | InfeasibleResult;

export interface WindowBound {
  earliest: number;
  latest: number;
}

/** Fill absent optional cue windows with the whole-day closed interval. */
export function cueWindow(cue: Cue, daySpan: number = DAY_MS): WindowBound {
  return {
    earliest: cue.earliest ?? 0,
    latest: cue.latest ?? daySpan,
  };
}

/** Same contract as solve() with a configurable day span (tests use small U). */
export function solveWithSpan(
  cues: ReadonlyArray<Cue>,
  base: ReadonlyArray<number>,
  pins: Pins,
  daySpan: number,
): SolveResult {
  return solveCore(cues, base, pins, daySpan);
}

// ---------------------------------------------------------------------------
// Leftist heap of weighted, *unclamped* observations. Nodes are mutated in
// place and heaps are melded destructively (each observation belongs to one
// PAVA block).
// ---------------------------------------------------------------------------

interface HNode {
  value: number;
  weight: number;
  rank: number;
  left: HNode | null;
  right: HNode | null;
}

function makeNode(value: number, weight: number): HNode {
  return { value, weight, rank: 1, left: null, right: null };
}

/**
 * Destructive leftist heap meld.
 * mult = 1 gives a min heap, mult = -1 a max heap. Equal values can take
 * either branch; weight accounting, not heap order, defines the median.
 */
function meld(
  a: HNode | null,
  b: HNode | null,
  mult: 1 | -1,
): HNode | null {
  if (a === null) return b;
  if (b === null) return a;
  if (mult * a.value > mult * b.value) {
    const t = a;
    a = b;
    b = t;
  }
  a.right = meld(a.right, b, mult);
  const rl = a.left ? a.left.rank : 0;
  const rr = a.right ? a.right.rank : 0;
  if (rl < rr) {
    const t = a.left;
    a.left = a.right;
    a.right = t;
  }
  a.rank = (a.right ? a.right.rank : 0) + 1;
  return a;
}

function popLo(h: HNode | null): HNode | null {
  return meld(h ? h.left : null, h ? h.right : null, -1);
}

function popHi(h: HNode | null): HNode | null {
  return meld(h ? h.left : null, h ? h.right : null, 1);
}

// ---------------------------------------------------------------------------
// PAVA blocks.
// ---------------------------------------------------------------------------

interface Block {
  lo: HNode | null; // lower half (max heap), root is the lower median target
  hi: HNode | null; // upper half (min heap)
  wLo: number;
  wHi: number;
  total: number;
  lower: number; // intersection of member y intervals
  upper: number;
  value: number; // projected lower weighted median
  memberHead: EntryNode | null; // members in index order
  memberTail: EntryNode | null;
}

interface EntryNode {
  index: number;
  next: EntryNode | null;
}

function clamp(v: number, lower: number, upper: number): number {
  return Math.min(upper, Math.max(lower, v));
}

function blockMedian(blk: Block): number {
  return clamp(blk.lo!.value, blk.lower, blk.upper);
}

function singleton(
  index: number,
  target: number,
  lower: number,
  upper: number,
): Block {
  const head: EntryNode = { index, next: null };
  const blk: Block = {
    lo: makeNode(target, 1),
    hi: null,
    wLo: 1,
    wHi: 0,
    total: 1,
    lower,
    upper,
    value: 0,
    memberHead: head,
    memberTail: head,
  };
  blk.value = blockMedian(blk);
  return blk;
}

/** Merge PAVA block b into a (a precedes b). */
function mergeBlocks(a: Block, b: Block): Block {
  a.lo = meld(a.lo, b.lo, -1);
  a.hi = meld(a.hi, b.hi, 1);
  a.wLo += b.wLo;
  a.wHi += b.wHi;
  a.total += b.total;
  a.lower = Math.max(a.lower, b.lower);
  a.upper = Math.min(a.upper, b.upper);
  if (a.memberTail) a.memberTail.next = b.memberHead;
  else a.memberHead = b.memberHead;
  a.memberTail = b.memberTail;

  // Maintain the lower weighted median of raw targets: m is the smallest
  // value for which 2 * weight(<= m) >= total. lo holds values <= m and hi
  // holds values >= m; the block's feasible projection happens afterwards.
  for (;;) {
    if (2 * a.wLo < a.total) {
      const node = a.hi!;
      a.hi = popHi(a.hi);
      a.wHi -= node.weight;
      node.left = null;
      node.right = null;
      node.rank = 1;
      a.lo = meld(a.lo, node, -1);
      a.wLo += node.weight;
      continue;
    }
    if (a.lo !== null && 2 * (a.wLo - a.lo.weight) >= a.total) {
      const node = a.lo;
      a.lo = popLo(a.lo);
      a.wLo -= node.weight;
      node.left = null;
      node.right = null;
      node.rank = 1;
      a.hi = meld(a.hi, node, 1);
      a.wHi += node.weight;
      continue;
    }
    if (a.lo !== null && a.hi !== null && a.lo.value > a.hi.value) {
      const top = a.lo;
      const bot = a.hi;
      a.lo = popLo(a.lo);
      a.hi = popHi(a.hi);
      a.wLo -= top.weight;
      a.wHi -= bot.weight;
      top.left = top.right = null;
      top.rank = 1;
      bot.left = bot.right = null;
      bot.rank = 1;
      a.lo = meld(a.lo, bot, -1);
      a.wLo += bot.weight;
      a.hi = meld(a.hi, top, 1);
      a.wHi += top.weight;
      continue;
    }
    break;
  }

  a.value = blockMedian(a);
  return a;
}

export function solve(input: SolveInput): SolveResult {
  return solveCore(input.cues, input.base, input.pins ?? new Map(), DAY_MS);
}

function solveCore(
  cues: ReadonlyArray<Cue>,
  base: ReadonlyArray<number>,
  pins: Pins,
  daySpan: number,
): SolveResult {
  const n = cues.length;
  if (n === 0) return { ok: true, starts: [], cost: 0 };
  if (base.length !== n) return { ok: false, reason: 'INFEASIBLE' };

  // Prefix durations P[i] = sum_{k < i} duration[k].
  const P = new Array<number>(n);
  P[0] = 0;
  for (let i = 0; i + 1 < n; i++) P[i + 1] = P[i] + cues[i].duration;
  const suffixEnd = P[n - 1];

  const earliest = new Array<number>(n);
  const latest = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const w = cueWindow(cues[i], daySpan);
    if (
      !Number.isInteger(w.earliest) ||
      !Number.isInteger(w.latest) ||
      w.earliest < 0 ||
      w.latest < 0 ||
      w.earliest > daySpan ||
      w.latest > daySpan ||
      w.earliest > w.latest
    ) {
      return { ok: false, reason: 'INVALID_WINDOW', conflictIndex: i };
    }
    earliest[i] = w.earliest;
    latest[i] = w.latest;
  }

  const pinAt = new Array<number | null>(n).fill(null);
  for (const [idx, start] of pins) {
    if (!Number.isInteger(idx) || idx < 0 || idx >= n) {
      return { ok: false, reason: 'INVALID_PIN', conflictIndex: idx };
    }
    if (!Number.isInteger(start) || start < 0 || start > daySpan) {
      return { ok: false, reason: 'INVALID_PIN', conflictIndex: idx };
    }
    pinAt[idx] = start;
  }

  // Per-coordinate upper bounds including every later edit window transformed
  // back to this cue's x coordinate. The forward lower-envelope scan plus these
  // suffix upper bounds detects the first cue of a chain conflict.
  const suffixUpper = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    const own = Math.min(latest[i], daySpan - (suffixEnd - P[i]));
    suffixUpper[i] =
      i + 1 < n ? Math.min(own, suffixUpper[i + 1] - cues[i].duration) : own;
  }

  // One forward x-space feasibility pass. `required` is the earliest start
  // allowed by the start of day, prior durations, and prior fixed points.
  let required = 0;
  for (let i = 0; i < n; i++) {
    const cap = suffixUpper[i];
    const lower = Math.max(required, earliest[i]);
    const pin = pinAt[i];

    if (lower > cap) {
      return {
        ok: false,
        reason: 'WINDOW_CHAIN',
        conflictIndex: i,
        requiredStart: lower,
        allowedLatest: cap,
      };
    }

    if (pin !== null) {
      // A value outside this cue's own closed interval is a self-window
      // failure. The preceding window-only check already ensured the chain up
      // to this cue can pass at all.
      if (pin < earliest[i] || pin > latest[i]) {
        return {
          ok: false,
          reason: 'PIN_OUTSIDE_WINDOW',
          conflictIndex: i,
          requiredStart: pin,
          allowedEarliest: earliest[i],
          allowedLatest: latest[i],
        };
      }
      if (pin < lower || pin > cap) {
        // The pin is in its own edit window but incompatible with prior
        // durations or with the future-window/day suffix transformed here.
        return {
          ok: false,
          reason: 'WINDOW_CHAIN',
          conflictIndex: i,
          requiredStart: lower,
          allowedLatest: cap,
        };
      }
      required = pin + cues[i].duration;
    } else {
      required = lower + cues[i].duration;
    }
  }

  // Bounded L1 isotonic regression in y-space. Exact pins are simply singleton
  // intervals [pinY, pinY]; feasibility above guarantees pooled blocks never
  // need an empty intersection. A pin therefore dominates every pooled block
  // without a special oversized observation.
  const b = new Array<number>(n);
  const lowerY = new Array<number>(n);
  const upperY = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    b[i] = base[i] - P[i];
    lowerY[i] = Math.max(0, earliest[i] - P[i]);
    upperY[i] = suffixUpper[i] - P[i];
    const pin = pinAt[i];
    if (pin !== null) {
      const v = pin - P[i];
      lowerY[i] = upperY[i] = v;
    }
  }

  const stack: Block[] = [];
  for (let i = 0; i < n; i++) {
    let blk = singleton(i, b[i], lowerY[i], upperY[i]);
    while (stack.length > 0) {
      const top = stack[stack.length - 1];
      if (top.value <= blk.value) break;
      stack.pop();
      blk = mergeBlocks(top, blk);
    }
    stack.push(blk);
  }

  const y = new Array<number>(n);
  let cost = 0;
  for (const blk of stack) {
    if (blk.lower > blk.upper) {
      return {
        ok: false,
        reason: 'WINDOW_CHAIN',
        conflictIndex: blk.memberHead?.index ?? 0,
        requiredStart: 0,
        allowedLatest: 0,
      };
    }
    let entry = blk.memberHead;
    while (entry !== null) {
      const i = entry.index;
      y[i] = blk.value;
      cost += Math.abs(blk.value - b[i]);
      entry = entry.next;
    }
  }

  const starts = new Array<number>(n);
  for (let i = 0; i < n; i++) starts[i] = y[i] + P[i];

  // Defensive verification (contract checks; should never fire after the
  // explicit feasibility pass).
  for (let i = 0; i < n; i++) {
    if (!Number.isInteger(starts[i]) || starts[i] < 0 || starts[i] > daySpan) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
    if (i > 0 && starts[i] < starts[i - 1] + cues[i - 1].duration) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
    if (starts[i] < earliest[i] || starts[i] > latest[i]) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
    if (pinAt[i] !== null && starts[i] !== pinAt[i]) {
      return { ok: false, reason: 'INFEASIBLE' };
    }
  }

  return { ok: true, starts, cost };
}
