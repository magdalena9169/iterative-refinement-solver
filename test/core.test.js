import test from 'node:test';
import assert from 'node:assert/strict';
import { refine, refineSystem, SolveError } from '../src/core.js';

// A tiny dense linear solver used only by the tests. Gaussian elimination with
// partial pivoting; no third-party code.
function gauss(A, b) {
  const n = A.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (M[piv][col] === 0) throw new Error('singular');
    if (piv !== col) {
      const t = M[piv];
      M[piv] = M[col];
      M[col] = t;
    }
    const p = M[col][col];
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / p;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

function matVec(A, x) {
  return A.map((row) => row.reduce((s, v, j) => s + v * x[j], 0));
}

function infNorm(v) {
  return v.reduce((m, a) => Math.max(m, Math.abs(a)), 0);
}

// A deliberately ill-conditioned 2x2 system. Direct elimination gives an
// answer with a visible residual; refinement should shrink it.
const A = [
  [1, 1],
  [1, 1 + 1e-8],
];
const b = [2, 2 + 1e-8];

function exactSolve() {
  return [1, 1];
}

test('refine improves a slightly wrong initial guess', () => {
  const x0 = [1.001, 0.999];
  const res = refine(A, b, x0, (rhs) => gauss(A, rhs), { tolerance: 1e-12 });
  assert.ok(res.iterations >= 1);
  assert.ok(res.residualNorm <= 1e-12);
  assert.ok(Math.abs(res.solution[0] - 1) < 1e-6);
  assert.ok(Math.abs(res.solution[1] - 1) < 1e-6);
});

test('refine returns zero iterations when the guess already satisfies tolerance', () => {
  const res = refine(A, b, [1, 1], (rhs) => gauss(A, rhs), { tolerance: 1e-12 });
  assert.equal(res.iterations, 0);
  assert.equal(res.residualNorm, 0);
  assert.deepEqual(res.solution, [1, 1]);
});

test('refine stops when the residual stops shrinking', () => {
  // A solver that always returns the zero correction cannot reduce the
  // residual, so refinement must stop after a single pass instead of looping.
  const x0 = [0, 0];
  const res = refine(A, b, x0, () => [0, 0], { maxIterations: 50 });
  assert.equal(res.iterations, 1);
  assert.deepEqual(res.solution, [0, 0]);
});

test('refine respects maxIterations', () => {
  const res = refine(A, b, [0, 0], (rhs) => gauss(A, rhs), { maxIterations: 1, tolerance: 0 });
  assert.equal(res.iterations, 1);
});

test('refine handles maxIterations of zero', () => {
  const res = refine(A, b, [0, 0], (rhs) => gauss(A, rhs), { maxIterations: 0 });
  assert.equal(res.iterations, 0);
  assert.deepEqual(res.solution, [0, 0]);
  assert.ok(res.residualNorm > 0);
});

test('refine does not mutate its inputs', () => {
  const x0 = [0.5, 0.5];
  const snapshot = x0.slice();
  refine(A, b, x0, (rhs) => gauss(A, rhs));
  assert.deepEqual(x0, snapshot);
});

test('refine rejects a non-square matrix', () => {
  assert.throws(() => refine([[1, 2, 3]], [1], [0], (r) => r), SolveError);
});

test('refine rejects a mismatched right-hand side', () => {
  assert.throws(() => refine(A, [1], [0, 0], (r) => r), SolveError);
});

test('refine rejects a mismatched initial guess', () => {
  assert.throws(() => refine(A, b, [0], (r) => r), SolveError);
});

test('refine rejects a non-finite entry', () => {
  assert.throws(() => refine([[1, NaN], [0, 1]], [1, 1], [0, 0], (r) => r), SolveError);
});

test('refine rejects a bad solve result length', () => {
  assert.throws(() => refine(A, b, [0, 0], () => [1]), SolveError);
});

test('refine rejects a negative maxIterations', () => {
  assert.throws(() => refine(A, b, [0, 0], (r) => r, { maxIterations: -1 }), SolveError);
});

test('refine rejects a negative tolerance', () => {
  assert.throws(() => refine(A, b, [0, 0], (r) => r, { tolerance: -1 }), SolveError);
});

test('refineSystem wires a one-shot solver through', () => {
  const res = refineSystem(A, b, [1.001, 0.999], (M, rhs) => gauss(M, rhs), { tolerance: 1e-12 });
  assert.ok(res.residualNorm <= 1e-12);
  assert.ok(Math.abs(res.solution[0] - 1) < 1e-6);
});

test('refineSystem rejects a non-function solver', () => {
  assert.throws(() => refineSystem(A, b, [0, 0], 42), SolveError);
});

test('refine drives a larger system to a small residual', () => {
  const M = [
    [4, 1, 0],
    [1, 4, 1],
    [0, 1, 4],
  ];
  const rhs = matVec(M, [1, 2, 3]);
  const res = refine(M, rhs, [1.01, 1.99, 3.02], (r) => gauss(M, r), { tolerance: 1e-12 });
  assert.ok(res.residualNorm <= 1e-12);
  assert.ok(infNorm(matVec(M, res.solution).map((v, i) => v - rhs[i])) <= 1e-12);
});
