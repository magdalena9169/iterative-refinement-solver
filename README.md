# iterative-refinement-solver

Given an approximate solution to a linear system `A x = b`, this library repeatedly
substitutes the residual back through a solver to drive the error down.

```js
import { refine } from './src/index.js';

const A = [
  [1, 1],
  [1, 1 + 1e-8],
];
const b = [2, 2 + 1e-8];

// Any function that solves A y = rhs for y works here. Reuse a cached
// factorisation inside this callback if you have one.
const solve = (rhs) => {
  // ... your linear solve ...
  return rhs;
};

const { solution, residualNorm, iterations } = refine(A, b, [1.001, 0.999], solve, {
  maxIterations: 10,
  tolerance: 1e-12,
});
```

`refineSystem(A, b, x0, solveSystem, options)` is a thin wrapper for the case where
the solver takes the matrix as its first argument: it adapts `solveSystem(A, rhs)`
into the `solve(rhs)` callback that `refine` expects.

## Why this exists

Iterative refinement is useful when the cost of factorising `A` dominates the cost
of solving, which is the usual situation for dense systems. Factor once, then each
refinement pass is only a triangular solve against the stored factorisation, so the
marginal cost per digit of accuracy is small. The trade-off is that refinement
cannot rescue a solve that is already dominated by rounding error: once the
residual stops shrinking, more passes only shuffle noise around, which is why the
loop exits early when the residual fails to decrease.

## Exports

- `refine(A, b, x0, solve, options)` — returns `{ solution, residualNorm, iterations }`.
- `refineSystem(A, b, x0, solveSystem, options)` — same result, adapting a
  matrix-taking solver.
- `SolveError` — thrown for malformed input, bad options, or a `solve` result of
  the wrong length.

## The awkward edge

The library never forms or factorises `A` itself; it only calls your `solve`. That
means a `solve` that silently returns a wrong answer of the right length will be
trusted, and refinement will happily iterate on garbage. Validate your solver in
your own tests. The other edge is the early-exit rule: if a pass produces a
residual that is not strictly smaller than the previous one, refinement stops, so
`iterations` can be smaller than `maxIterations` even when `tolerance` was never
reached. The returned `residualNorm` tells you which case you are in.
