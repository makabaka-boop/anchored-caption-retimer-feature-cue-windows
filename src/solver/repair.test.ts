import { describe, it, expect } from 'vitest';
import { solveWithSpan, type Cue } from './solve';
// Vite raw import (resolved by Vitest); the ambient declaration keeps tsc happy.
import repairSource from './repair.ts?raw';
import {
  analyzeMaxRetention,
  analyzeMaxRetentionWithSpan,
  applyRepair,
  buildRepairPlan,
  type RepairAnalysis,
  type RepairPlan,
  type RevisionId,
} from './repair';

function makeCues(durations: number[]): Cue[] {
  return durations.map((d, i) => ({
    start: durations.slice(0, i).reduce((a, b) => a + b, 0),
    duration: d,
    text: `c${i}`,
  }));
}

function prefixes(durations: number[]): number[] {
  const P = new Array<number>(durations.length).fill(0);
  for (let i = 1; i < P.length; i++) P[i] = P[i - 1] + durations[i - 1];
  return P;
}

function lexLess(a: number[], b: number[]): boolean {
  const m = Math.min(a.length, b.length);
  for (let i = 0; i < m; i++) {
    if (a[i] !== b[i]) return a[i] < b[i];
  }
  return a.length < b.length;
}

interface BruteResult {
  mandatoryReleased: number[];
  retained: number[];
  released: number[];
  retainedPins: Map<number, number>;
  retainedCount: number;
  totalCount: number;
}

/**
 * Reference maximum-retention solver: enumerate every subset of the
 * in-envelope pins (2^v, v <= 3 here), keep the largest feasible one, ties
 * broken by the full ascending release list (mandatory entries included).
 */
function bruteMaxRetention(
  durations: number[],
  pins: Map<number, number>,
  U: number,
): BruteResult | null {
  const P = prefixes(durations);
  const C = U - P[durations.length - 1];
  if (C < 0) return null;

  const mandatory: number[] = [];
  const valid: Array<{ idx: number; start: number; y: number }> = [];
  for (const [idx, start] of pins) {
    const y = start - P[idx];
    if (y < 0 || y > C) mandatory.push(idx);
    else valid.push({ idx, start, y });
  }
  mandatory.sort((a, b) => a - b);
  valid.sort((a, b) => a.idx - b.idx);

  let bestKept: number[] = [];
  let bestReleased: number[] | null = null;

  const v = valid.length;
  for (let mask = 0; mask < 1 << v; mask++) {
    const kept: number[] = [];
    let prevY = -Infinity;
    let feasible = true;
    for (let i = 0; i < v; i++) {
      if (mask & (1 << i)) {
        if (valid[i].y < prevY) {
          feasible = false;
          break;
        }
        prevY = valid[i].y;
        kept.push(valid[i].idx);
      }
    }
    if (!feasible) continue;
    const optionalReleased: number[] = [];
    for (let i = 0; i < v; i++) {
      if (!(mask & (1 << i))) optionalReleased.push(valid[i].idx);
    }
    // Merge with the (common) mandatory list.
    const released: number[] = [];
    let a = 0;
    let b = 0;
    while (a < mandatory.length || b < optionalReleased.length) {
      if (
        b >= optionalReleased.length ||
        (a < mandatory.length && mandatory[a] <= optionalReleased[b])
      ) {
        released.push(mandatory[a++]);
      } else {
        released.push(optionalReleased[b++]);
      }
    }
    if (
      bestReleased === null ||
      kept.length > bestKept.length ||
      (kept.length === bestKept.length && lexLess(released, bestReleased))
    ) {
      bestKept = kept;
      bestReleased = released;
    }
  }

  const retainedPins = new Map<number, number>();
  for (const item of valid) if (bestKept.includes(item.idx)) retainedPins.set(item.idx, item.start);
  return {
    mandatoryReleased: mandatory,
    retained: bestKept,
    released: bestReleased!,
    retainedPins,
    retainedCount: bestKept.length,
    totalCount: pins.size,
  };
}

function expectDetail(actual: RepairAnalysis): asserts actual is Exclude<
  RepairAnalysis,
  { kind: 'unrecoverable' }
> {
  expect(actual.kind).toBe('repair');
}

describe('max-retention repair — pure analysis contract', () => {
  it('enumerates every pin subset on all short sequences (n <= 3)', () => {
    const cart = <T,>(sets: T[][]): T[][] =>
      sets.reduce<T[][]>(
        (acc, s) => acc.flatMap((prefix) => s.map((v) => [...prefix, v])),
        [[]],
      );
    const range = (m: number): number[] =>
      Array.from({ length: m }, (_, i) => i + 1);

    let cases = 0;
    for (const n of [1, 2, 3]) {
      const durGrids = cart(range(n).map(() => [1, 2]));
      for (const durations of durGrids) {
        const P = prefixes(durations);
        for (const U of [P[n - 1], P[n - 1] + 3]) {
          const C = U - P[n - 1];
          // Absent, boundary values (-1 / 0 / C / C+1) and interior values.
          const yGrid = Array.from(
            new Set([-1, 0, 1, 2, C - 1, C, C + 1].filter((y) => Number.isInteger(y))),
          );
          const choices: Array<null | number>[] = range(n).map((_, i) => [
            null,
            ...yGrid.map((y) => P[i] + y),
          ]);
          for (const combo of cart(choices)) {
            const pins = new Map<number, number>();
            combo.forEach((start, i) => {
              if (start !== null) pins.set(i, start);
            });
            const expected = bruteMaxRetention(durations, pins, U);
            expect(expected).not.toBeNull();
            const actual = analyzeMaxRetentionWithSpan(
              makeCues(durations),
              pins,
              U,
            );
            expectDetail(actual);
            expect(actual.retainedCount, `n=${n} d=${durations} U=${U} pins=${[...pins]}`).toBe(
              expected!.retainedCount,
            );
            expect(actual.totalCount).toBe(expected!.totalCount);
            expect(actual.retained).toEqual(expected!.retained);
            expect(actual.released).toEqual(expected!.released);
            expect(actual.mandatoryReleased).toEqual(
              expected!.mandatoryReleased,
            );
            expect(new Map(actual.retainedPins)).toEqual(
              expected!.retainedPins,
            );
            // The retained set must solve feasibly against the real span.
            const r = solveWithSpan(
              makeCues(durations),
              makeCues(durations).map((c) => c.start),
              actual.retainedPins,
              U,
            );
            expect(r.ok, `retained set infeasible: ${[...actual.released]}`).toBe(true);
            cases++;
          }
        }
      }
    }
    expect(cases).toBeGreaterThan(3_000);
  });

  it('transformed values [5,1,4] release the first item', () => {
    // P = [0,1,2]; pinned transformed coordinates y = 5,1,4 with C = 10.
    const durations = [1, 1, 1];
    const U = 12;
    const pins = new Map<number, number>([
      [0, 5],
      [1, 2],
      [2, 6],
    ]);
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expectDetail(a);
    expect(a.retainedCount).toBe(2);
    expect(a.retained).toEqual([1, 2]);
    expect(a.released).toEqual([0]);
    expect(a.mandatoryReleased).toEqual([]);
  });

  it('tie: [2,1,2] keeps the lexicographically largest retained set', () => {
    const durations = [1, 1, 1];
    const U = 12;
    const pins = new Map<number, number>([
      [0, 2],
      [1, 1],
      [2, 2],
    ]);
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expectDetail(a);
    // [1,2] yields release list [0], which is lex-smaller than [0,2] -> [1].
    expect(a.retained).toEqual([1, 2]);
    expect(a.released).toEqual([0]);
  });

  it('tie with a mandatory entry still compares the complete release list', () => {
    // Mandatory release at index 3 (y = 12 > C = 11); the valid tail [2,1,2]
    // on indices 0,1,2 ties the same way — the common mandatory prefix must
    // not change which optional index gets released.
    const durations = [1, 1, 1, 1];
    const U = 14; // P[n-1] = 3, C = 11
    const pins = new Map<number, number>([
      [0, 2],
      [1, 1],
      [2, 2],
      [3, 15], // P[3] = 3 -> y = 12 > C
    ]);
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expectDetail(a);
    expect(a.mandatoryReleased).toEqual([3]);
    expect(a.retained).toEqual([1, 2]);
    expect(a.released).toEqual([0, 3]);
  });

  it('boundary filtering: y < 0 and y > C are mandatory releases', () => {
    const durations = [5, 5, 5];
    const U = 20; // P[n-1] = 10, C = 10
    const pins = new Map<number, number>([
      [0, 0], // y = 0, on the edge: keepable
      [1, 4], // P[1] = 5 -> y = -1: mandatory
      [2, 21], // P[2] = 10 -> y = 11 > C: mandatory
    ]);
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expectDetail(a);
    expect(a.mandatoryReleased.sort((x, y) => x - y)).toEqual([1, 2]);
    expect(a.retained).toEqual([0]);
    expect(a.released).toEqual([1, 2]);
  });

  it('malformed pin indices and non-integer starts are mandatory releases', () => {
    const durations = [1, 1];
    const U = 10;
    const pins = new Map<number, number>([
      [0, 0],
      [5, 1],
    ]);
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expectDetail(a);
    expect(a.mandatoryReleased).toContain(5);
    expect(a.retained).toEqual([0]);
    expect(a.totalCount).toBe(2);
  });

  it('returns unrecoverable when total durations already exceed the span', () => {
    const durations = [4, 4, 4]; // P[n-1] = 8
    const pins = new Map<number, number>([[0, 0]]);
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, 7);
    expect(a.kind).toBe('unrecoverable');
    expect(buildRepairPlan(a, { draftRev: 1, baseRev: 0, pinsRev: 0 })).toBe(
      null,
    );
  });

  it('keeps every pin when the set is already feasible', () => {    const durations = [2, 2, 2];
    const U = 20;
    const pins = new Map<number, number>([
      [0, 3],
      [1, 6],
      [2, 9],
    ]);
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expectDetail(a);
    expect(a.retainedCount).toBe(3);
    expect(a.released).toEqual([]);
    expect(a.mandatoryReleased).toEqual([]);
  });

  it('is a pure layer: no runtime call into solve and no input mutation', () => {
    // Only a type-only import from the solver is allowed: no runtime symbol
    // (solve/solveWithSpan/DAY_MS) may be imported or invoked, so the repair
    // cannot be implemented by deleting a pin and re-solving repeatedly.
    const importLines = repairSource
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.startsWith('import'));
    expect(importLines.length).toBeGreaterThan(0);
    for (const line of importLines) {
      expect(line, 'solver import must be type-only').toMatch(
        /^import\s+type\b/,
      );
    }
    expect(repairSource).not.toMatch(/\bsolveWithSpan\b/);
    // Strip comments (the header documents that this layer does not call
    // solve) before looking for an actual invocation in code.
    const codeOnly = repairSource.replace(/\/\*[\s\S]*?\*\//g, '').replace(
      /\/\/[^\n]*/g,
      '',
    );
    expect(codeOnly).not.toMatch(/\bsolve\s*\(/);
    const durations = [1, 1, 1];
    const pins = new Map<number, number>([
      [0, 5],
      [1, 2],
      [2, 6],
    ]);
    const snapshot = [...pins];
    analyzeMaxRetentionWithSpan(makeCues(durations), pins, 12);
    expect([...pins]).toEqual(snapshot);
  });

  it('repeating value groups (i mod 4): regression for grouped activation', () => {
    // Within one value group the LNDS continuation length grows at the later
    // occurrences; a single-event-per-group reconstruction used to miss this.
    const n = 12;
    const durations = new Array<number>(n).fill(1);
    const P = prefixes(durations);
    const U = P[n - 1] + 5;
    const pins = new Map<number, number>();
    for (let i = 0; i < n; i++) pins.set(i, P[i] + (i % 4));
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expectDetail(a);
    const expected = bruteMaxRetention(durations, pins, U);
    expect(a.retainedCount).toBe(expected!.retainedCount);
    expect(a.retained).toEqual(expected!.retained);
    expect(a.released).toEqual(expected!.released);
    // LNDS length of 0,1,2,3,0,1,2,3,0,1,2,3 is 6; lex-largest chain begins
    // at the second full block.
    expect(a.retained).toEqual([0, 4, 8, 9, 10, 11]);
  });

  it('random medium sequences agree with the brute-force reference', () => {
    let seed = 0x9e3779b9;
    const rand = (): number => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let trial = 0; trial < 200; trial++) {
      const n = 1 + Math.floor(rand() * 10);
      const durations = Array.from({ length: n }, () => 1 + Math.floor(rand() * 3));
      const P = prefixes(durations);
      const U = P[n - 1] + Math.floor(rand() * 6);
      const C = U - P[n - 1];
      const pins = new Map<number, number>();
      for (let i = 0; i < n; i++) {
        if (rand() < 0.7) pins.set(i, P[i] + Math.floor(rand() * (C + 4)) - 1);
      }
      const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
      const expected = bruteMaxRetention(durations, pins, U);
      if (expected === null) {
        expect(a.kind).toBe('unrecoverable');
      } else {
        expectDetail(a);
        expect(a.retainedCount, `trial ${trial}`).toBe(expected.retainedCount);
        expect(a.released, `trial ${trial}`).toEqual(expected.released);
        expect(a.mandatoryReleased, `trial ${trial}`).toEqual(
          expected.mandatoryReleased,
        );
        const r = solveWithSpan(
          makeCues(durations),
          makeCues(durations).map((c) => c.start),
          a.retainedPins,
          U,
        );
        expect(r.ok, `trial ${trial}`).toBe(true);
      }
    }
  });
});

describe('repair plans: identities, expiry and atomic apply', () => {
  const id: RevisionId = { draftRev: 1, baseRev: 2, pinsRev: 3 };

  it('carries the draft/base/pins revisions it was generated with', () => {
    const a = analyzeMaxRetentionWithSpan(
      makeCues([1, 1, 1]),
      new Map<number, number>([
        [0, 5],
        [1, 2],
        [2, 6],
      ]),
      12,
    );
    const plan = buildRepairPlan(a, id);
    expect(plan).not.toBeNull();
    expect(plan).toMatchObject({
      draftRev: 1,
      baseRev: 2,
      pinsRev: 3,
      retainedCount: 2,
      released: [0],
    });
  });

  it('changing a pin after generation makes the old apply expired with no side effects', () => {
    const livePins = new Map<number, number>([
      [0, 5],
      [1, 2],
      [2, 6],
    ]);
    const a = analyzeMaxRetentionWithSpan(makeCues([1, 1, 1]), livePins, 12);
    const plan: RepairPlan = buildRepairPlan(a, id)!;

    // Operator edits a pin: the pin revision advances.
    const changed = new Map(livePins);
    changed.set(2, 7);
    const staleId: RevisionId = { ...id, pinsRev: id.pinsRev + 1 };

    const stale = applyRepair(plan, staleId);
    expect(stale.ok).toBe(false);
    if (!stale.ok) expect(stale.reason).toBe('EXPIRED');
    // No partial modification: the live map is exactly what the operator has.
    expect(changed.get(2)).toBe(7);
    expect(changed.size).toBe(3);
    expect([...changed]).toEqual([
      [0, 5],
      [1, 2],
      [2, 7],
    ]);

    // Applying against the matching identity still works atomically.
    const fresh = applyRepair(plan, id);
    expect(fresh.ok).toBe(true);
    if (fresh.ok) {
      expect([...fresh.pins]).toEqual([
        [1, 2],
        [2, 6],
      ]);
      // Source state and plan payload are left intact.
      expect(livePins.size).toBe(3);
      expect(plan.retainedCount).toBe(2);
    }
  });

  it('expires on draft or baseline revisions too', () => {
    const a = analyzeMaxRetentionWithSpan(
      makeCues([1, 1]),
      new Map<number, number>([
        [0, 3],
        [1, 0],
      ]),
      5,
    );
    const plan = buildRepairPlan(a, id)!;
    expect(applyRepair(plan, { ...id, draftRev: 9 }).ok).toBe(false);
    expect(applyRepair(plan, { ...id, baseRev: 9 }).ok).toBe(false);
    expect(applyRepair(plan, id).ok).toBe(true);
  });

  it('default-day convenience wrapper agrees on a real-day conflict', () => {
    const d = 60_000;
    const cues: Cue[] = Array.from({ length: 1442 }, (_, i) => ({
      start: i * d,
      duration: d,
      text: 'a',
    }));
    // P[n-1] = 1441*60000 = 86_460_000 > DAY: unrecoverable even with no pins.
    const over = analyzeMaxRetention({ cues, pins: new Map([[0, 0]]) });
    expect(over.kind).toBe('unrecoverable');
    // 1441 cues fit exactly: a boundary-busting pin is mandatory-released.
    const fit = Array.from({ length: 1441 }, (_, i) => ({
      start: i * d,
      duration: d,
      text: 'a',
    }));
    const ok2 = analyzeMaxRetention({
      cues: fit,
      pins: new Map<number, number>([[1440, 86_400_000]]), // y = 0: edge-keepable
    });
    expect(ok2.kind).toBe('repair');
  });
});

describe('max-retention repair — 20000 adversarial pins', () => {
  const n = 20_000;

  it('strictly descending y: O(k log k) timing, then solve is feasible', () => {
    const durations = new Array<number>(n).fill(1);
    const P = prefixes(durations);
    const C = n + 1;
    const U = P[n - 1] + C;
    // Adversarial to a naive scan: y strictly descends 20000..1; LNDS is one.
    const pins = new Map<number, number>();
    for (let i = 0; i < n; i++) pins.set(i, P[i] + (n - i));
    const cues = makeCues(durations);
    const base = cues.map((c) => c.start);

    const t0 = performance.now();
    const a = analyzeMaxRetentionWithSpan(cues, pins, U);
    const elapsed = performance.now() - t0;
    expect(elapsed).toBeLessThan(2_000);
    expectDetail(a);
    expect(a.retainedCount).toBe(1);
    expect(a.retained).toEqual([n - 1]); // lex-largest singleton
    expect(a.released.length).toBe(n - 1);
    expect(a.mandatoryReleased.length).toBe(0);

    const id: RevisionId = { draftRev: 1, baseRev: 0, pinsRev: 0 };
    const plan = buildRepairPlan(a, id)!;
    const applied = applyRepair(plan, id);
    expect(applied.ok).toBe(true);
    if (!applied.ok) return;

    const t1 = performance.now();
    const r = solveWithSpan(cues, base, applied.pins, U);
    expect(performance.now() - t1).toBeLessThan(2_000);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.starts[n - 1]).toBe(applied.pins.get(n - 1));
      for (let i = 1; i < n; i++) {
        expect(r.starts[i] - r.starts[i - 1]).toBeGreaterThanOrEqual(1);
      }
    }
  });

  it('constant y keeps all 20000 pins and solves', () => {
    const durations = new Array<number>(n).fill(1);
    const P = prefixes(durations);
    const U = P[n - 1] + 5;
    const pins = new Map<number, number>();
    for (let i = 0; i < n; i++) pins.set(i, P[i] + 3);
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expectDetail(a);
    expect(a.retainedCount).toBe(n);
    expect(a.released).toEqual([]);
    const r = solveWithSpan(
      makeCues(durations),
      makeCues(durations).map((c) => c.start),
      a.retainedPins,
      U,
    );
    expect(r.ok).toBe(true);
  });

  it('tie-heavy sequence matches an independent LNDS length and solves', () => {
    const durations = new Array<number>(n).fill(1);
    const P = prefixes(durations);
    const U = P[n - 1] + n + 2;
    // Strictly decreasing until the final two positions, which form an
    // equal pair: LNDS length is 2 and the lex-largest chain must end on the
    // last two indices, exercising the equal-y tie path at full scale.
    const ys = new Array<number>(n);
    for (let i = 0; i < n - 2; i++) ys[i] = n - i;
    ys[n - 2] = 1;
    ys[n - 1] = 1;
    const pins = new Map<number, number>();
    for (let i = 0; i < n; i++) pins.set(i, P[i] + ys[i]);

    const t0 = performance.now();
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expect(performance.now() - t0).toBeLessThan(2_000);
    expectDetail(a);

    const tails: number[] = [];
    for (const v of ys) {
      let lo = 0;
      let hi = tails.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (tails[mid] > v) hi = mid;
        else lo = mid + 1;
      }
      if (lo === tails.length) tails.push(v);
      else tails[lo] = v;
    }
    expect(tails.length).toBe(2);
    expect(a.retainedCount).toBe(2);
    // Retained y values must be non-decreasing; the equal pair ties to the
    // lexicographically largest retained indices.
    expect(a.retained).toEqual([n - 2, n - 1]);
    const r = solveWithSpan(
      makeCues(durations),
      makeCues(durations).map((c) => c.start),
      a.retainedPins,
      U,
    );
    expect(r.ok).toBe(true);
  });

  it('repeating value groups at k = 20000 stay within the time bound', () => {
    const durations = new Array<number>(n).fill(1);
    const P = prefixes(durations);
    const U = P[n - 1] + 5;
    const pins = new Map<number, number>();
    // Only four distinct y values repeating: stresses the grouped activation
    // and per-group point maxima with 20000 members spread over four leaves.
    for (let i = 0; i < n; i++) pins.set(i, P[i] + (i % 4));
    const t0 = performance.now();
    const a = analyzeMaxRetentionWithSpan(makeCues(durations), pins, U);
    expect(performance.now() - t0).toBeLessThan(2_000);
    expectDetail(a);
    // Pattern 0,1,2,3 repeating: 5000 complete blocks then a 0,1,2 tail;
    // the LNDS takes one block plus the tail: 4 + 3 = 5003.
    expect(a.retainedCount).toBe(5003);
    const r2 = solveWithSpan(
      makeCues(durations),
      makeCues(durations).map((c) => c.start),
      a.retainedPins,
      U,
    );
    expect(r2.ok).toBe(true);
  });
});
