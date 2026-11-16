// Core iterative refinement for linear systems.
//
// The idea: given an approximate solution x0 to A x = b, compute the residual
// r = b - A x0, solve A d = r for the correction d, and set x1 = x0 + d. Repeating
// this drives the residual down until it is limited by floating point rounding.
//
// We deliberately do NOT solve the correction system ourselves. The caller supplies
// a `solve` callback that solves A y = rhs for y. That keeps this module free of any
// particular factorisation and lets callers reuse a cached LU/Cholesky factor across
// refinement steps, which is the whole point of refinement: the expensive
// factorisation is done once, and each refinement step is only a triangular solve.

export class SolveError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SolveError';
  }
}

function assertMatrix(A) {
  if (!Array.isArray(A) || A.length === 0) {
    throw new SolveError('matrix must be a non-empty array of rows');
  }
  const n = A.length;
  for (let i = 0; i < n; i++) {
    const row = A[i];
    if (!Array.isArray(row) || row.length !== n) {
      throw new SolveError('matrix must be square');
    }
    for (let j = 0; j < n; j++) {
      if (typeof row[j] !== 'number' || !Number.isFinite(row[j])) {
        throw new SolveError('matrix entries must be finite numbers');
      }
    }
  }
  return n;
}

function assertVector(v, n, label) {
  if (!Array.isArray(v) || v.length !== n) {
    throw new SolveError(`${label} must be an array of length ${n}`);
  }
  for (let i = 0; i < n; i++) {
    if (typeof v[i] !== 'number' || !Number.isFinite(v[i])) {
      throw new SolveError(`${label} entries must be finite numbers`);
    }
  }
}

function matVec(A, x) {
  const n = A.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = A[i];
    for (let j = 0; j < n; j++) {
      s += row[j] * x[j];
    }
    out[i] = s;
  }
  return out;
}

function residual(A, x, b) {
  // Compute r = b - A x in one pass without forming A x separately, to avoid
  // an extra allocation on the hot path.
  const n = A.length;
  const r = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    const row = A[i];
    for (let j = 0; j < n; j++) {
      s -= row[j] * x[j];
    }
    r[i] = s;
  }
  return r;
}

function normInf(v) {
  let m = 0;
  for (let i = 0; i < v.length; i++) {
    const a = Math.abs(v[i]);
    if (a > m) m = a;
  }
  return m;
}

/**
 * Refine an approximate solution to A x = b by repeated residual correction.
 *
 * @param {number[][]} A square matrix, row-major.
 * @param {number[]} b right-hand side.
 * @param {number[]} x0 initial approximate solution.
 * @param {(rhs: number[]) => number[]} solve solves A y = rhs for y.
 * @param {object} [options]
 * @param {number} [options.maxIterations=10] maximum number of correction passes.
 * @param {number} [options.tolerance=1e-12] stop when the infinity norm of the
 *   residual drops to or below this value.
 * @returns {{ solution: number[], residualNorm: number, iterations: number }}
 *
 * The returned `iterations` counts correction passes actually performed, so 0
 * means the initial guess already satisfied the tolerance.
 */
export function refine(A, b, x0, solve, options = {}) {
  const n = assertMatrix(A);
  assertVector(b, n, 'b');
  assertVector(x0, n, 'x0');
  if (typeof solve !== 'function') {
    throw new SolveError('solve must be a function');
  }
  const maxIterations = options.maxIterations === undefined ? 10 : options.maxIterations;
  const tolerance = options.tolerance === undefined ? 1e-12 : options.tolerance;
  if (!Number.isInteger(maxIterations) || maxIterations < 0) {
    throw new SolveError('maxIterations must be a non-negative integer');
  }
  if (typeof tolerance !== 'number' || !(tolerance >= 0) || !Number.isFinite(tolerance)) {
    throw new SolveError('tolerance must be a finite non-negative number');
  }

  let x = x0.slice();
  let r = residual(A, x, b);
  let rn = normInf(r);
  let iterations = 0;

  while (iterations < maxIterations && rn > tolerance) {
    const d = solve(r);
    assertVector(d, n, 'solve result');
    for (let i = 0; i < n; i++) {
      x[i] += d[i];
    }
    r = residual(A, x, b);
    const next = normInf(r);
    iterations++;
    // If the residual stops shrinking, further passes cannot help: the
    // correction is now dominated by rounding noise. Bail out rather than
    // burning iterations and possibly oscillating.
    if (!(next < rn)) {
      rn = next;
      break;
    }
    rn = next;
  }

  return { solution: x, residualNorm: rn, iterations };
}

/**
 * Convenience wrapper: build the residual-correcting `solve` callback from a
 * plain one-shot solver. Use this when you cannot reuse a factorisation; if you
 * can, pass the cached factorisation directly to `refine` instead.
 *
 * @param {number[][]} A square matrix, row-major.
 * @param {number[]} b right-hand side.
 * @param {number[]} x0 initial approximate solution.
 * @param {(A: number[][], rhs: number[]) => number[]} solveSystem solves A y = rhs.
 * @param {object} [options] forwarded to `refine`.
 * @returns {{ solution: number[], residualNorm: number, iterations: number }}
 */
export function refineSystem(A, b, x0, solveSystem, options = {}) {
  if (typeof solveSystem !== 'function') {
    throw new SolveError('solveSystem must be a function');
  }
  return refine(A, b, x0, (rhs) => solveSystem(A, rhs), options);
}
