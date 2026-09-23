// Solver: move cues just enough to remove overlaps while honouring exact pins
// and optional per-cue shot-cut windows.
//
// Variables x[i] are integer cue starts. Constraints are
//   earliest[i] <= x[i] <= latest[i], where omitted bounds default to the day
//   x[i+1] >= x[i] + duration[i]
//   x[i] = start for every pinned cue.
// The objective is min Σ|x[i] - base[i]|; ties use the lexicographically
// smallest complete start vector.
//
// With P[i] = Σ_{k<i} duration[k] and y[i] = x[i] - P[i], gap constraints
// become y non-decreasing. Windows become per-index boxes
//   earliest[i] - P[i] <= y[i] <= latest[i] - P[i],
// so the problem is bounded L1 isotonic regression with a few infinite-weight
// pinned observations. A block PAVA solves it directly: adjacent fitted blocks
// merge while their clipped lower weighted medians decrease. Bounds are kept as
// the block's lower maximum / upper minimum; no unconstrained answer is later
// trimmed cue-by-cue.

export const DAY_MS = 86_400_000;

export interface Cue {
  start: number;
  duration: number;
  text: string;
  /** Optional inclusive lower shot-cut bound; omitted means 0. */
  earliest?: number;
  /** Optional inclusive upper shot-cut bound; omitted means the day end. */
  latest?: number;
}

/** cueIndex -> fixed integer start; one pin per cue, re-edit overwrites. */
export type Pins = ReadonlyMap<number, number>;

export interface SolveInput {
  cues: ReadonlyArray<Cue>;
  base: ReadonlyArray<number>;
  pins?: Pins;
}

export interface FeasibilityConflict {
  /** First cue whose lower envelope cannot meet its inclusive upper bound. */
  cueIndex: number;
  /** Minimum start forced there by the cue itself and every preceding cue. */
  requiredStart: number;
  /** Inclusive maximum start permitted at that cue. */
  allowedUpperBound: number;
}

export type SolveResult =
  | { ok: true; starts: number[]; cost: number }
  | { ok: false; reason: 'INVALID_WINDOW'; cueIndex: number }
  | { ok: false; reason: 'INVALID_PIN'; cueIndex: number; pinStart: number }
  | {
      ok: false;
      reason: 'WINDOW_INFEASIBLE';
      conflict: FeasibilityConflict;
    }
  | {
      ok: false;
      reason: 'PIN_OUTSIDE_WINDOW';
      pinStart: number;
      conflict: FeasibilityConflict;
    }
  | {
      ok: false;
      reason: 'WINDOW_CHAIN_CONFLICT';
      pinStart?: number;
      conflict: FeasibilityConflict;
    };

/** Same contract as solve() with a configurable day span (tests use small U). */
export function solveWithSpan(
  cues: ReadonlyArray<Cue>,
  base: ReadonlyArray<number>,
  pins: Pins,
  daySpan: number,
): SolveResult {
  return solveCore(cues, base, pins, daySpan);
}

interface ResolvedWindow {
  low: number;
  high: number;
}

function resolveWindow(
  cue: Cue,
  daySpan: number,
): { ok: true; window: ResolvedWindow } | { ok: false } {
  const low = cue.earliest === undefined ? 0 : cue.earliest;
  const high = cue.latest === undefined ? daySpan : cue.latest;
  if (
    !Number.isInteger(low) ||
    !Number.isInteger(high) ||
    low < 0 ||
    high < low ||
    high > daySpan
  ) {
    return { ok: false };
  }
  return { ok: true, window: { low, high } };
}

type ScanReason =
  | 'WINDOW_INFEASIBLE'
  | 'PIN_OUTSIDE_WINDOW'
  | 'WINDOW_CHAIN_CONFLICT';

interface ScanConflict extends FeasibilityConflict {
  reason: ScanReason;
  pinStart?: number;
}

/** Earliest conflict in the start-space lower-envelope scan. */
function scanWindows(
  cues: ReadonlyArray<Cue>,
  windows: ReadonlyArray<ResolvedWindow>,
): ScanConflict | null {
  let required = 0;
  for (let i = 0; i < cues.length; i++) {
    required = Math.max(required, windows[i].low);
    if (required > windows[i].high) {
      return {
        reason: 'WINDOW_INFEASIBLE',
        cueIndex: i,
        requiredStart: required,
        allowedUpperBound: windows[i].high,
      };
    }
    required += cues[i].duration;
  }
  return null;
}

function scanPins(
  cues: ReadonlyArray<Cue>,
  windows: ReadonlyArray<ResolvedWindow>,
  pinAt: ReadonlyArray<number | null>,
): ScanConflict | null {
  let required = 0;
  let lastPinStart: number | null = null;
  for (let i = 0; i < cues.length; i++) {
    required = Math.max(required, windows[i].low);
    const pin = pinAt[i];

    // A value outside the cue's own raw [earliest, latest] is first classified
    // as a self-window violation, even if earlier pins have already made the
    // propagated interval tighter.
    if (pin !== null && (pin < windows[i].low || pin > windows[i].high)) {
      return {
        reason: 'PIN_OUTSIDE_WINDOW',
        cueIndex: i,
        requiredStart: required,
        allowedUpperBound: windows[i].high,
        pinStart: pin,
      };
    }

    // The pin itself is inside its raw window but cannot be placed there after
    // duration propagation (or a previous pin forces this later cue out).
    if (pin !== null ? pin < required : required > windows[i].high) {
      return {
        reason: 'WINDOW_CHAIN_CONFLICT',
        cueIndex: i,
        requiredStart: required,
        allowedUpperBound: windows[i].high,
        pinStart: pin ?? lastPinStart!,
      };
    }

    const value = pin === null ? required : pin;
    if (pin !== null) lastPinStart = pin;
    required = value + cues[i].duration;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Leftist heap of weighted observations. Nodes are mutated in place; heaps
// are melded destructively (every observation belongs to exactly one block).
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

/** Destructive leftist heap meld. mult = 1 is a min heap, -1 a max heap. */
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
// Bounded PAVA blocks.
// ---------------------------------------------------------------------------

interface Block {
  lo: HNode | null; // lower half (max heap), root is the lower weighted median
  hi: HNode | null; // upper half (min heap)
  wLo: number;
  wHi: number;
  total: number;
  boxLow: number; // max low bound in this contiguous block (transformed)
  boxHigh: number; // min high bound in this contiguous block (transformed)
  frozen: number | null; // exact pin value when a pin belongs to the block
  memberHead: EntryNode | null;
  memberTail: EntryNode | null;
}

interface EntryNode {
  index: number;
  next: EntryNode | null;
}

function singleton(
  index: number,
  value: number,
  weight: number,
  boxLow: number,
  boxHigh: number,
  frozen: number | null,
): Block {
  const head: EntryNode = { index, next: null };
  return {
    lo: makeNode(value, weight),
    hi: null,
    wLo: weight,
    wHi: 0,
    total: weight,
    boxLow,
    boxHigh,
    frozen,
    memberHead: head,
    memberTail: head,
  };
}

function clamp(v: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, v));
}

/** A block's optimum is its lower weighted median projected onto its box. */
function blockValue(block: Block): number {
  return clamp(block.lo!.value, block.boxLow, block.boxHigh);
}

/** Merge PAVA block b into a (a precedes b). */
function mergeBlocks(a: Block, b: Block): Block {
  a.lo = meld(a.lo, b.lo, -1);
  a.hi = meld(a.hi, b.hi, 1);
  a.wLo += b.wLo;
  a.wHi += b.wHi;
  a.total += b.total;
  a.boxLow = Math.max(a.boxLow, b.boxLow);
  a.boxHigh = Math.min(a.boxHigh, b.boxHigh);
  if (a.frozen === null) a.frozen = b.frozen;
  else if (b.frozen !== null && b.frozen !== a.frozen) {
    // Feasibility pre-scans prevent decreasing exact pins from reaching here.
    a.frozen = b.frozen;
  }
  if (a.memberTail) a.memberTail.next = b.memberHead;
  else a.memberHead = b.memberHead;
  a.memberTail = b.memberTail;

  // Maintain the unconstrained lower weighted median in lo. Bounds are applied
  // only when reading blockValue(), which is exact for a common block box.
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
  return a;
}

export function solve(input: SolveInput): SolveResult {
  return solveCore(input.cues, input.base, input.pins ?? new Map(), DAY_MS);
}

function failure(conflict: ScanConflict): SolveResult {
  if (conflict.reason === 'WINDOW_INFEASIBLE') {
    const {
      reason,
      cueIndex,
      requiredStart,
      allowedUpperBound,
    } = conflict;
    return {
      ok: false,
      reason,
      conflict: { cueIndex, requiredStart, allowedUpperBound },
    };
  }
  if (conflict.reason === 'PIN_OUTSIDE_WINDOW') {
    const { reason, cueIndex, requiredStart, allowedUpperBound, pinStart } =
      conflict;
    return {
      ok: false,
      reason,
      pinStart: pinStart!,
      conflict: { cueIndex, requiredStart, allowedUpperBound },
    };
  }
  const { reason, cueIndex, requiredStart, allowedUpperBound, pinStart } =
    conflict;
  return {
    ok: false,
    reason,
    ...(pinStart === undefined ? {} : { pinStart }),
    conflict: { cueIndex, requiredStart, allowedUpperBound },
  };
}

function solveCore(
  cues: ReadonlyArray<Cue>,
  base: ReadonlyArray<number>,
  pins: Pins,
  daySpan: number,
): SolveResult {
  const n = cues.length;
  if (n === 0) return { ok: true, starts: [], cost: 0 };

  const windows: ResolvedWindow[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const resolved = resolveWindow(cues[i], daySpan);
    if (!resolved.ok) return { ok: false, reason: 'INVALID_WINDOW', cueIndex: i };
    windows[i] = resolved.window;
  }

  const pinAt = new Array<number | null>(n).fill(null);
  for (const [idx, start] of pins) {
    if (
      !Number.isInteger(idx) ||
      idx < 0 ||
      idx >= n ||
      !Number.isInteger(start) ||
      start < 0 ||
      start > daySpan
    ) {
      return {
        ok: false,
        reason: 'INVALID_PIN',
        cueIndex: Number.isInteger(idx) ? idx : -1,
        pinStart: Number.isInteger(start) ? start : Number.NaN,
      };
    }
    pinAt[idx] = start;
  }

  const windowConflict = scanWindows(cues, windows);
  if (windowConflict) return failure(windowConflict);
  const pinConflict = scanPins(cues, windows, pinAt);
  if (pinConflict) return failure(pinConflict);

  // Prefix durations and transformed targets/bounds.
  const P = new Array<number>(n);
  P[0] = 0;
  for (let i = 0; i + 1 < n; i++) P[i + 1] = P[i] + cues[i].duration;
  const b = new Array<number>(n);
  const lowY = new Array<number>(n);
  const highY = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    b[i] = base[i] - P[i];
    lowY[i] = windows[i].low - P[i];
    highY[i] = windows[i].high - P[i];
  }

  // A pin outweighs all ordinary observations combined, forcing its block's
  // unconstrained weighted median to the exact transformed pin value.
  const pinWeight = n + 1;
  const stack: Block[] = [];
  const push = (i: number): void => {
    const pin = pinAt[i];
    let blk =
      pin === null
        ? singleton(i, b[i], 1, lowY[i], highY[i], null)
        : singleton(
            i,
            pin - P[i],
            pinWeight,
            lowY[i],
            highY[i],
            pin - P[i],
          );
    while (stack.length > 0) {
      const top = stack[stack.length - 1];
      if (blockValue(top) <= blockValue(blk)) break;
      stack.pop();
      blk = mergeBlocks(top, blk);
    }
    stack.push(blk);
  };
  for (let i = 0; i < n; i++) push(i);

  const y = new Array<number>(n);
  let cost = 0;
  for (const blk of stack) {
    const value = blockValue(blk);
    let entry = blk.memberHead;
    while (entry !== null) {
      const i = entry.index;
      y[i] = value;
      cost += Math.abs(value - b[i]);
      entry = entry.next;
    }
  }

  const starts = new Array<number>(n);
  for (let i = 0; i < n; i++) starts[i] = y[i] + P[i];

  // Defensive contract verification.
  for (let i = 0; i < n; i++) {
    if (
      !Number.isInteger(starts[i]) ||
      starts[i] < windows[i].low ||
      starts[i] > windows[i].high
    ) {
      return { ok: false, reason: 'INVALID_WINDOW', cueIndex: i };
    }
    if (i > 0 && starts[i] < starts[i - 1] + cues[i - 1].duration) {
      return { ok: false, reason: 'INVALID_WINDOW', cueIndex: i };
    }
    if (pinAt[i] !== null && starts[i] !== pinAt[i]) {
      return {
        ok: false,
        reason: 'INVALID_PIN',
        cueIndex: i,
        pinStart: pinAt[i]!,
      };
    }
  }

  return { ok: true, starts, cost };
}
